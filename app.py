# app.py
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, time
import base64
import threading
import random
import pickle

from config import Config
from models import db, Teacher, Department, Batch, Subject, Student, FaceEmbedding, Attendance
from face_utils import embedder, liveness
from face_utils.recognition import FaceMatcher

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def seed_default_data():
    """Create default departments and batches if they don't exist."""
    # Departments
    dept_names = ['Computer Science', 'Electronics', 'Mechanical', 'Civil']
    for name in dept_names:
        if not Department.query.filter_by(name=name).first():
            dept = Department(name=name)
            db.session.add(dept)
            db.session.commit()
            # For each department, create batches (e.g., 2022, 2023, 2024)
            for year in ['2022', '2023', '2024']:
                batch_name = f"{year} Batch"
                if not Batch.query.filter_by(name=batch_name, department_id=dept.id).first():
                    batch = Batch(name=batch_name, department_id=dept.id)
                    db.session.add(batch)
    db.session.commit()

class TeacherUser(UserMixin):
    def __init__(self, teacher):
        self.id = teacher.id
        self.username = teacher.username

    @staticmethod
    def get(user_id):
        teacher = Teacher.query.get(int(user_id))
        return TeacherUser(teacher) if teacher else None

@login_manager.user_loader
def load_user(user_id):
    return TeacherUser.get(user_id)

# Initialize the face matcher
face_matcher = FaceMatcher(db, embedder)  # we'll define this

# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if Teacher.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
        else:
            hashed = generate_password_hash(password)
            teacher = Teacher(username=username, password_hash=hashed)
            db.session.add(teacher)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        teacher = Teacher.query.filter_by(username=username).first()
        if teacher and check_password_hash(teacher.password_hash, password):
            user = TeacherUser(teacher)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        dept_id = request.form.get('department')
        batch_id = request.form.get('batch')
        subject_id = request.form.get('subject')
        new_subject_name = request.form.get('new_subject', '').strip()

        # Validate required selections
        if not dept_id or not batch_id:
            flash('Please select both department and batch.', 'warning')
            return redirect(url_for('dashboard'))

        # Save to session
        session['dept_id'] = dept_id
        session['batch_id'] = batch_id

        # Handle subject selection/creation
        if new_subject_name:
            # Check if subject already exists for this batch
            existing = Subject.query.filter_by(
                name=new_subject_name, 
                batch_id=batch_id
            ).first()
            if existing:
                session['subject_id'] = existing.id
                flash(f'Using existing subject: {existing.name}', 'info')
            else:
                new_sub = Subject(name=new_subject_name, batch_id=batch_id)
                db.session.add(new_sub)
                db.session.commit()
                session['subject_id'] = new_sub.id
                flash(f'New subject "{new_subject_name}" created.', 'success')
        elif subject_id:
            # Use selected existing subject
            session['subject_id'] = subject_id
        else:
            flash('Please select or create a subject.', 'warning')
            return redirect(url_for('dashboard'))

        return redirect(url_for('attendance'))

    # GET: Show selection form with current values
    departments = Department.query.all()
    batches = Batch.query.all()
    subjects = Subject.query.all()
    return render_template(
        'dashboard.html',
        departments=departments,
        batches=batches,
        subjects=subjects
    )

@app.route('/students')
@login_required
def students():
    # Get students for the current teacher's selection? Actually teacher selects department/batch
    dept_id = session.get('dept_id')
    batch_id = session.get('batch_id')
    if not dept_id or not batch_id:
        flash('Please select department and batch first', 'warning')
        return redirect(url_for('dashboard'))
    dept = Department.query.get(dept_id)
    batch = Batch.query.get(batch_id)
    students = Student.query.filter_by(department_id=dept_id, batch_id=batch_id).all()
    return render_template('students.html', students=students, dept=dept, batch=batch)

@app.route('/students/add', methods=['POST'])
@login_required
def add_student():
    roll_no = request.form['roll_no']
    name = request.form['name']
    dept_id = session.get('dept_id')
    batch_id = session.get('batch_id')
    if not dept_id or not batch_id:
        flash('Please select department and batch first', 'warning')
        return redirect(url_for('dashboard'))
    # Check if roll already exists
    if Student.query.get(roll_no):
        flash('Student with this roll number already exists', 'danger')
        return redirect(url_for('students'))
    student = Student(roll_no=roll_no, name=name, department_id=dept_id, batch_id=batch_id)
    db.session.add(student)
    db.session.commit()
    flash(f'Student {name} added', 'success')
    return redirect(url_for('students'))

@app.route('/students/delete/<roll_no>')
@login_required
def delete_student(roll_no):
    student = Student.query.get(roll_no)
    if student:
        db.session.delete(student)
        db.session.commit()
        flash(f'Student {roll_no} deleted', 'success')
    else:
        flash('Student not found', 'danger')
    return redirect(url_for('students'))

@app.route('/enroll/<roll_no>', methods=['GET', 'POST'])
@login_required
def enroll_student(roll_no):
    student = Student.query.get(roll_no)
    if not student:
        flash('Student not found', 'danger')
        return redirect(url_for('students'))
    if request.method == 'POST':
        # Process uploaded images
        files = request.files.getlist('photos')
        if not files:
            flash('No files uploaded', 'warning')
            return redirect(url_for('enroll_student', roll_no=roll_no))
        saved = 0
        for file in files:
            if file and allowed_file(file.filename):
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                emb, msg = embedder.extract_embedding(img_rgb)
                if emb is not None:
                    # Save embedding
                    emb_blob = serialize_embedding(emb)
                    embedding = FaceEmbedding(student_roll_no=roll_no, embedding_blob=emb_blob)
                    db.session.add(embedding)
                    db.session.commit()
                    saved += 1
                else:
                    flash(f'Failed to extract embedding from {file.filename}: {msg}', 'warning')
        if saved:
            flash(f'{saved} face photo(s) enrolled for {student.name}', 'success')
        return redirect(url_for('students'))
    # GET: show enrollment page with current count
    count = FaceEmbedding.query.filter_by(student_roll_no=roll_no).count()
    return render_template('enroll.html', student=student, count=count)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# Helper to serialize embeddings for storage
def serialize_embedding(emb):
    return emb.astype(np.float32).tobytes()

def deserialize_embedding(data):
    return np.frombuffer(data, dtype=np.float32)

@app.route('/attendance')
@login_required
def attendance():
    # Ensure selection is present
    dept_id = session.get('dept_id')
    batch_id = session.get('batch_id')
    subject_id = session.get('subject_id')
    if not (dept_id and batch_id and subject_id):
        flash('Please select department, batch, and subject first', 'warning')
        return redirect(url_for('dashboard'))
    subject = Subject.query.get(subject_id)
    return render_template('attendance.html', subject=subject)

@app.route('/attendance/process_frame', methods=['POST'])
@login_required
def process_frame():
    """
    Receive a base64 encoded frame from the frontend, perform face detection,
    liveness challenge, and face recognition. Returns JSON with status.
    """
    # Get the frame data
    data = request.get_json()
    image_data = data.get('image')
    if not image_data:
        return jsonify({'error': 'No image data'}), 400

    # Decode base64
    try:
        # Remove header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Get current session state (stored in server memory per user)
    user_id = current_user.id
    if user_id not in app.config.get('SESSION_STATES', {}):
        # Initialize state
        app.config['SESSION_STATES'][user_id] = {
            'phase': 'challenge',
            'challenge': random.choice(['BLINK', 'TURN_LEFT', 'TURN_RIGHT']),
            'frames': [],  # store recent frames for liveness
            'result': None,
            'result_frames': 0,
            'last_marked': []
        }
    state = app.config['SESSION_STATES'][user_id]

    # Process frame
    result = process_attendance_frame(img_rgb, state, session['subject_id'])
    return jsonify(result)

def process_attendance_frame(img_rgb, state, subject_id):
    """
    Main logic for processing one frame. Returns dict for frontend.
    """
    # Face detection
    bbox = embedder.get_face_bbox(img_rgb)
    if bbox is None:
        # No face detected
        return {'status': 'no_face', 'phase': state['phase'], 'challenge': state['challenge']}

    # Analyze liveness
    frame_data = liveness.analyse_frame(img_rgb)  # we need to adapt liveness.py to work with RGB
    # liveness.analyse_frame expects RGB and returns FrameData object
    # We'll assume liveness.py is updated to have analyse_frame that works with RGB.

    # Update state frames buffer (keep last 10 frames for blink/turn detection)
    state['frames'].append(frame_data)
    if len(state['frames']) > 10:
        state['frames'].pop(0)

    if state['phase'] == 'challenge':
        # Evaluate challenge using the buffered frames
        challenge = state['challenge']
        passed, msg = liveness.evaluate_challenge(challenge, state['frames'])
        if passed:
            # Challenge passed, now recognize face
            emb, emb_msg = embedder.extract_embedding(img_rgb)
            if emb is None:
                return {'status': 'embedding_failed', 'message': emb_msg, 'phase': 'result', 'success': False}
            # Match face against students in the current batch
            match = face_matcher.find_match(emb, subject_id, threshold=embedder.get_threshold())
            if match['matched']:
                # Mark attendance
                student = Student.query.get(match['roll_no'])
                if student:
                    # Check if already marked today for this subject
                    today = date.today()
                    existing = Attendance.query.filter_by(
                        student_roll_no=student.roll_no,
                        subject_id=subject_id,
                        date=today
                    ).first()
                    if existing:
                        marked = False
                        msg = f"{student.name} already marked present today"
                    else:
                        att = Attendance(
                            student_roll_no=student.roll_no,
                            subject_id=subject_id,
                            date=today,
                            time=datetime.now().time(),
                            method='face'
                        )
                        db.session.add(att)
                        db.session.commit()
                        marked = True
                        msg = f"{student.name} marked present"
                    # Update state to show result
                    state['phase'] = 'result'
                    state['result'] = {
                        'success': True,
                        'name': student.name,
                        'roll_no': student.roll_no,
                        'confidence': match['confidence'],
                        'marked': marked,
                        'message': msg
                    }
                    state['result_frames'] = 60  # show result for ~2 seconds (assuming 30 fps)
                    return {
                        'status': 'success',
                        'name': student.name,
                        'roll_no': student.roll_no,
                        'confidence': match['confidence'],
                        'marked': marked,
                        'message': msg
                    }
            else:
                # No match
                state['phase'] = 'result'
                state['result'] = {
                    'success': False,
                    'name': 'Unknown',
                    'roll_no': None,
                    'confidence': match['confidence'],
                    'marked': False,
                    'message': 'Face not recognized'
                }
                state['result_frames'] = 60
                return {
                    'status': 'no_match',
                    'name': 'Unknown',
                    'confidence': match['confidence'],
                    'message': 'Face not recognized'
                }
        else:
            # Still in challenge phase
            return {
                'status': 'challenge_active',
                'phase': 'challenge',
                'challenge': challenge,
                'message': msg,
                'face_detected': True
            }
    elif state['phase'] == 'result':
        # Displaying result, decrement counter
        state['result_frames'] -= 1
        if state['result_frames'] <= 0:
            # Reset to challenge
            state['phase'] = 'challenge'
            state['challenge'] = random.choice(['BLINK', 'TURN_LEFT', 'TURN_RIGHT'])
            state['frames'] = []
            state['result'] = None
            return {'status': 'reset'}
        else:
            # Still showing result
            return {'status': 'result', **state['result']}
    else:
        return {'status': 'unknown'}

# Global dictionary for states (should be per user, maybe in memory)
app.config['SESSION_STATES'] = {}

@app.route('/reports')
@login_required
def reports():
    subject_id = session.get('subject_id')
    if not subject_id:
        flash('Please select a subject first', 'warning')
        return redirect(url_for('dashboard'))
    subject = Subject.query.get(subject_id)
    # Get attendance for this subject, optionally with date filters
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    query = Attendance.query.filter_by(subject_id=subject_id)
    if from_date:
        query = query.filter(Attendance.date >= datetime.strptime(from_date, '%Y-%m-%d').date())
    if to_date:
        query = query.filter(Attendance.date <= datetime.strptime(to_date, '%Y-%m-%d').date())
    records = query.all()
    # Build summary per student
    summary = {}
    for rec in records:
        key = rec.student_roll_no
        if key not in summary:
            student = Student.query.get(key)
            summary[key] = {'name': student.name, 'count': 0}
        summary[key]['count'] += 1
    summary_list = [{'roll_no': k, 'name': v['name'], 'present': v['count']} for k, v in summary.items()]
    # Optionally, all students in the batch
    dept_id = session.get('dept_id')
    batch_id = session.get('batch_id')
    if dept_id and batch_id:
        all_students = Student.query.filter_by(department_id=dept_id, batch_id=batch_id).all()
        for student in all_students:
            if student.roll_no not in summary:
                summary_list.append({'roll_no': student.roll_no, 'name': student.name, 'present': 0})
    return render_template('reports.html', records=records, summary=summary_list, subject=subject)

# For downloading CSV
@app.route('/reports/download')
@login_required
def download_report():
    subject_id = session.get('subject_id')
    if not subject_id:
        flash('Please select a subject first', 'warning')
        return redirect(url_for('dashboard'))
    # Similar to above, get records and convert to CSV
    # We'll implement later
    pass

# ----------------------------------------------------------------------
# Start the app
# ----------------------------------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if not exist
        seed_default_data()  # Seed departments and batches
    app.run(debug=True)