import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import pandas as pd
import cv2
from PIL import Image
import io

from config import Config
from models import db, User, Department, Batch, Subject, Student, Attendance
from utils import embedder, face_db, liveness, attendance as att_module

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Ensure database tables exist
with app.app_context():
    db.create_all()
    # Create default department and batch if not exist (optional)
    # For demo, we can create some sample data via a separate script

# Helper to get current user from session
def get_current_user():
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_teacher', methods=['GET', 'POST'])
def register_teacher():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if len(username) < 3 or len(password) < 6:
            flash('Username must be at least 3 characters, password at least 6.')
            return redirect(url_for('register_teacher'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register_teacher'))
        user = User(username=username, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register_teacher.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Logged in successfully.')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    # Get departments for dropdown
    departments = Department.query.all()
    # For simplicity, if no departments exist, create some sample
    if not departments:
        # Create sample departments, batches, subjects (only once)
        dept1 = Department(name='Computer Science')
        dept2 = Department(name='Electronics')
        db.session.add_all([dept1, dept2])
        db.session.commit()
        batch1 = Batch(name='3rd Year', department_id=dept1.id)
        batch2 = Batch(name='2nd Semester', department_id=dept1.id)
        db.session.add_all([batch1, batch2])
        db.session.commit()
        # Assign subjects to teacher (current user)
        subj1 = Subject(name='Machine Learning', batch_id=batch1.id, teacher_id=user.id)
        subj2 = Subject(name='Database Systems', batch_id=batch2.id, teacher_id=user.id)
        db.session.add_all([subj1, subj2])
        db.session.commit()
        departments = Department.query.all()
    return render_template('dashboard.html', departments=departments, user=user)

@app.route('/get_batches/<int:dept_id>')
def get_batches(dept_id):
    batches = Batch.query.filter_by(department_id=dept_id).all()
    return jsonify([{'id': b.id, 'name': b.name} for b in batches])

@app.route('/get_subjects/<int:batch_id>')
def get_subjects(batch_id):
    user = get_current_user()
    subjects = Subject.query.filter_by(batch_id=batch_id, teacher_id=user.id).all()
    return jsonify([{'id': s.id, 'name': s.name} for s in subjects])

@app.route('/register_student', methods=['GET', 'POST'])
def register_student():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST':
        roll_no = request.form['roll_no'].strip()
        name = request.form['name'].strip()
        batch_id = int(request.form['batch_id'])
        # Check if student already exists in this batch
        existing = Student.query.filter_by(roll_no=roll_no, batch_id=batch_id).first()
        if existing:
            flash(f'Student with roll number {roll_no} already exists in this batch.')
            return redirect(url_for('register_student'))
        # Get face image
        if 'face_image' not in request.files:
            flash('No face image uploaded.')
            return redirect(url_for('register_student'))
        file = request.files['face_image']
        if file.filename == '':
            flash('No image selected.')
            return redirect(url_for('register_student'))
        # Read image file
        try:
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            flash(f'Error reading image: {e}')
            return redirect(url_for('register_student'))
        # Extract embedding
        emb, msg = embedder.extract_embedding(img_rgb)
        if emb is None:
            flash(f'Face embedding failed: {msg}')
            return redirect(url_for('register_student'))
        # Register in face_db
        ok, msg = face_db.register_student(roll_no, name)
        if not ok:
            flash(msg)
            return redirect(url_for('register_student'))
        ok, msg = face_db.add_embedding(roll_no, emb)
        if not ok:
            flash(msg)
            # Rollback face_db registration? We could delete it, but okay.
            return redirect(url_for('register_student'))
        # Save to Student table
        student = Student(roll_no=roll_no, name=name, batch_id=batch_id)
        db.session.add(student)
        db.session.commit()
        flash(f'Student {name} registered successfully with {face_db.get_student(roll_no)["n_photos"]} photo(s).')
        return redirect(url_for('dashboard'))
    # GET: display form with batches for this teacher
    batches = Batch.query.all()  # Or filter by teacher's departments?
    return render_template('register_student.html', batches=batches)

@app.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST':
        subject_id = int(request.form['subject_id'])
        # Get the subject and its batch
        subject = Subject.query.get(subject_id)
        if not subject:
            flash('Invalid subject.')
            return redirect(url_for('mark_attendance'))
        # Get the uploaded image (from webcam or file)
        if 'image' not in request.files:
            flash('No image uploaded.')
            return redirect(url_for('mark_attendance'))
        file = request.files['image']
        if file.filename == '':
            flash('No image selected.')
            return redirect(url_for('mark_attendance'))
        # Read image
        try:
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            flash(f'Error reading image: {e}')
            return redirect(url_for('mark_attendance'))
        # Liveness check (optional)
        # if not liveness.perform_liveness(img_rgb):  # We'll need to implement a simple check
        #     flash('Liveness check failed.')
        #     return redirect(url_for('mark_attendance'))
        # Extract embedding
        emb, msg = embedder.extract_embedding(img_rgb)
        if emb is None:
            flash(f'No face detected: {msg}')
            return redirect(url_for('mark_attendance'))
        # Get all students in this batch (to limit matching)
        batch_students = Student.query.filter_by(batch_id=subject.batch_id).all()
        batch_roll_nos = [s.roll_no for s in batch_students]
        # Custom match function: we'll use face_db.find_match but restrict to batch_roll_nos
        match = find_match_in_rollnos(emb, batch_roll_nos)
        if not match['matched']:
            flash(match['message'])
            return redirect(url_for('mark_attendance'))
        # Student found, record attendance
        student = Student.query.filter_by(roll_no=match['roll_no'], batch_id=subject.batch_id).first()
        if not student:
            flash('Student not found in database.')
            return redirect(url_for('mark_attendance'))
        # Check if already marked today for this subject
        today = date.today()
        existing = Attendance.query.filter_by(student_id=student.id, subject_id=subject_id, date=today).first()
        if existing:
            flash(f'{student.name} already marked present today.')
            return redirect(url_for('mark_attendance'))
        # Record attendance
        att = Attendance(student_id=student.id, subject_id=subject_id, date=today, marked_by=user.id)
        db.session.add(att)
        db.session.commit()
        flash(f'Attendance marked for {student.name} ({match["confidence"]}% confidence).')
        return redirect(url_for('mark_attendance'))
    # GET: display form with subjects for teacher
    subjects = Subject.query.filter_by(teacher_id=user.id).all()
    return render_template('mark_attendance.html', subjects=subjects)

def find_match_in_rollnos(query_emb, roll_nos, threshold=None):
    """Similar to face_db.find_match but only considers given roll_nos."""
    # Load face_db dictionary
    db_dict = face_db._load_db()  # internal function
    # Filter to those roll_nos that exist in db
    candidates = {r: db_dict[r] for r in roll_nos if r in db_dict and db_dict[r]['mean_emb'] is not None}
    if not candidates:
        return {'matched': False, 'roll_no': None, 'name': 'Unknown', 'confidence': 0.0,
                'message': 'No enrolled students in this batch.'}
    # Use threshold from embedder if not provided
    if threshold is None:
        threshold = embedder.get_threshold()
    best_sim = -1.0
    best_roll = None
    q = query_emb.astype(np.float32)
    for roll_no, data in candidates.items():
        # Similarity to mean embedding
        sim = float(np.dot(q, data['mean_emb']))
        # Also check individual embeddings
        for emb in data['embeddings']:
            s = float(np.dot(q, emb.astype(np.float32)))
            if s > sim:
                sim = s
        if sim > best_sim:
            best_sim = sim
            best_roll = roll_no
    confidence = round(best_sim * 100, 1)
    if best_sim < threshold:
        return {'matched': False, 'roll_no': None, 'name': 'Unknown', 'confidence': confidence,
                'message': f'No match found (best score {confidence}%).'}
    student = db_dict[best_roll]
    return {'matched': True, 'roll_no': best_roll, 'name': student['name'], 'confidence': confidence,
            'message': 'OK'}

@app.route('/reports', methods=['GET', 'POST'])
def reports():
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST':
        subject_id = int(request.form['subject_id'])
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        # Query attendance
        query = Attendance.query.filter_by(subject_id=subject_id)
        if start_date:
            query = query.filter(Attendance.date >= datetime.strptime(start_date, '%Y-%m-%d').date())
        if end_date:
            query = query.filter(Attendance.date <= datetime.strptime(end_date, '%Y-%m-%d').date())
        records = query.order_by(Attendance.date.desc()).all()
        # Get subject name
        subject = Subject.query.get(subject_id)
        return render_template('reports.html', records=records, subject=subject, start_date=start_date, end_date=end_date, subjects=Subject.query.filter_by(teacher_id=user.id).all())
    # GET: show form with subjects
    subjects = Subject.query.filter_by(teacher_id=user.id).all()
    return render_template('reports.html', subjects=subjects, records=[])

if __name__ == '__main__':
    app.run(debug=True)