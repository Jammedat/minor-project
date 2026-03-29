"""
database.py — Unified SQLite database layer

All student data is scoped by department + batch.
"""

import sqlite3
import os
import hashlib
import pickle
import numpy as np
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "faceattend.db")


# ── Connection ────────────────────────────────────────────────────────────────

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    with get_db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS teachers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name          TEXT DEFAULT ""
            );

            CREATE TABLE IF NOT EXISTS departments (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS batches (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
                UNIQUE(name, department_id)
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                department_id INTEGER NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
                UNIQUE(name, department_id)
            );

            CREATE TABLE IF NOT EXISTS students (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no       TEXT NOT NULL,
                name          TEXT NOT NULL,
                department_id INTEGER NOT NULL REFERENCES departments(id),
                batch_id      INTEGER NOT NULL REFERENCES batches(id),
                registered_on TEXT,
                face_photos   INTEGER DEFAULT 0,
                iris_photos   INTEGER DEFAULT 0,
                UNIQUE(roll_no, department_id, batch_id)
            );

            -- Individual raw embeddings for re-computation
            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                emb_data   BLOB NOT NULL,
                emb_type   TEXT NOT NULL DEFAULT "face"
            );

            -- Pre-computed mean embedding per student per type (fast matching)
            CREATE TABLE IF NOT EXISTS mean_embeddings (
                student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                emb_data   BLOB NOT NULL,
                emb_type   TEXT NOT NULL DEFAULT "face",
                PRIMARY KEY (student_id, emb_type)
            );

            -- Attendance — UNIQUE(student, subject, date) prevents duplicates
            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id),
                subject_id INTEGER NOT NULL REFERENCES subjects(id),
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                method     TEXT NOT NULL DEFAULT "face",
                UNIQUE(student_id, subject_id, date)
            );
        ''')
        _seed(conn)


def _seed(conn):
    """Insert default departments, batches, and subjects if not present."""
    depts = [
        ('BCT',            ['Data Structures & Algorithms','Database Management',
                                          'Operating Systems','Artificial Intelligence']),
        ('BEI', ['Digital Electronics','Database Management','Microprocessors',
                                          'Data Structures & Algorithms']),
        ('BME',                  ['Fluid Mechanics', 'Engineering Materials']),
        ('BCE',                       ['Structural Analysis','Hydraulics',
                                          'Transportation Engineering']),
    ]
    batches = ['078', '079', '080', '081']

    for dept_name, subjects in depts:
        conn.execute('INSERT OR IGNORE INTO departments (name) VALUES (?)', (dept_name,))
        row = conn.execute('SELECT id FROM departments WHERE name=?', (dept_name,)).fetchone()
        if row:
            did = row['id']
            for b in batches:
                conn.execute('INSERT OR IGNORE INTO batches (name, department_id) VALUES (?,?)', (b, did))
            for s in subjects:
                conn.execute('INSERT OR IGNORE INTO subjects (name, department_id) VALUES (?,?)', (s, did))


# ── Helper: serialize / deserialize numpy arrays ──────────────────────────────

def _emb_to_blob(arr: np.ndarray) -> bytes:
    return pickle.dumps(arr.astype(np.float32))

def _blob_to_emb(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


# ══════════════════════════════════════════════════════════════════════════════
#  TEACHER AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_teacher(username: str, password: str, name: str = "") -> tuple:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO teachers (username, password_hash, name) VALUES (?,?,?)',
                (username, _hash(password), name.strip())
            )
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def authenticate_teacher(username: str, password: str):
    """Returns teacher row dict or None."""
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM teachers WHERE username=? AND password_hash=?',
            (username, _hash(password))
        ).fetchone()
    return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
#  DEPARTMENTS / BATCHES / SUBJECTS
# ══════════════════════════════════════════════════════════════════════════════

def get_departments():
    with get_db() as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM departments ORDER BY name').fetchall()]


def add_department(name: str) -> tuple:
    try:
        with get_db() as conn:
            conn.execute('INSERT INTO departments (name) VALUES (?)', (name.strip(),))
        return True, "Department added."
    except sqlite3.IntegrityError:
        return False, "Department already exists."


def get_batches(department_id: int):
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            'SELECT * FROM batches WHERE department_id=? ORDER BY name DESC',
            (department_id,)
        ).fetchall()]


def add_batch(name: str, department_id: int) -> tuple:
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO batches (name, department_id) VALUES (?,?)',
                (name.strip(), department_id)
            )
        return True, "Batch added."
    except sqlite3.IntegrityError:
        return False, "Batch already exists for this department."


def get_subjects(department_id: int):
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            'SELECT * FROM subjects WHERE department_id=? ORDER BY name',
            (department_id,)
        ).fetchall()]


def add_subject(name: str, department_id: int) -> tuple:
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO subjects (name, department_id) VALUES (?,?)',
                (name.strip(), department_id)
            )
        return True, "Subject added."
    except sqlite3.IntegrityError:
        return False, "Subject already exists for this department."


def get_subject(subject_id: int):
    with get_db() as conn:
        row = conn.execute('SELECT * FROM subjects WHERE id=?', (subject_id,)).fetchone()
    return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENTS  (scoped by department_id + batch_id)
# ══════════════════════════════════════════════════════════════════════════════

def get_students(department_id: int, batch_id: int):
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            '''SELECT s.*, d.name AS dept_name, b.name AS batch_name
               FROM students s
               JOIN departments d ON d.id = s.department_id
               JOIN batches     b ON b.id = s.batch_id
               WHERE s.department_id=? AND s.batch_id=?
               ORDER BY s.roll_no''',
            (department_id, batch_id)
        ).fetchall()]


def get_student(student_id: int):
    with get_db() as conn:
        row = conn.execute('SELECT * FROM students WHERE id=?', (student_id,)).fetchone()
    return dict(row) if row else None


def register_student(roll_no: str, name: str, department_id: int, batch_id: int) -> tuple:
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with get_db() as conn:
            conn.execute(
                '''INSERT INTO students (roll_no, name, department_id, batch_id, registered_on)
                   VALUES (?,?,?,?,?)''',
                (str(roll_no).strip(), name.strip(), department_id, batch_id, now)
            )
        return True, "Student registered successfully."
    except sqlite3.IntegrityError:
        return False, f"Roll number {roll_no} already exists in this batch."


def update_student_name(student_id: int, new_name: str) -> tuple:
    with get_db() as conn:
        conn.execute('UPDATE students SET name=? WHERE id=?', (new_name.strip(), student_id))
    return True, "Name updated."


def delete_student(student_id: int) -> tuple:
    with get_db() as conn:
        conn.execute('DELETE FROM students WHERE id=?', (student_id,))
    return True, "Student deleted."


# ── Embeddings ────────────────────────────────────────────────────────────────

def add_embedding(student_id: int, embedding: np.ndarray, emb_type: str = "face") -> tuple:
    """Add one embedding and recompute mean. emb_type: 'face' or 'iris'."""
    with get_db() as conn:
        # Insert raw embedding
        conn.execute(
            'INSERT INTO embeddings (student_id, emb_data, emb_type) VALUES (?,?,?)',
            (student_id, _emb_to_blob(embedding), emb_type)
        )

        # Count photos and recompute mean
        rows = conn.execute(
            'SELECT emb_data FROM embeddings WHERE student_id=? AND emb_type=?',
            (student_id, emb_type)
        ).fetchall()

        embs = np.array([_blob_to_emb(r['emb_data']) for r in rows])
        mean = embs.mean(axis=0)
        norm = np.linalg.norm(mean)
        mean_emb = (mean / (norm + 1e-8)).astype(np.float32)

        conn.execute(
            '''INSERT INTO mean_embeddings (student_id, emb_data, emb_type)
               VALUES (?,?,?)
               ON CONFLICT(student_id, emb_type) DO UPDATE SET emb_data=excluded.emb_data''',
            (student_id, _emb_to_blob(mean_emb), emb_type)
        )

        # Update photo count on student
        col = 'face_photos' if emb_type == 'face' else 'iris_photos'
        conn.execute(f'UPDATE students SET {col}=? WHERE id=?', (len(rows), student_id))

    return True, f"Photo {len(rows)} enrolled ({emb_type})."


def clear_embeddings(student_id: int, emb_type: str = "face") -> tuple:
    with get_db() as conn:
        conn.execute(
            'DELETE FROM embeddings WHERE student_id=? AND emb_type=?',
            (student_id, emb_type)
        )
        conn.execute(
            'DELETE FROM mean_embeddings WHERE student_id=? AND emb_type=?',
            (student_id, emb_type)
        )
        col = 'face_photos' if emb_type == 'face' else 'iris_photos'
        conn.execute(f'UPDATE students SET {col}=0 WHERE id=?', (student_id,))
    return True, f"All {emb_type} photos cleared."


def is_enrolled(student_id: int, emb_type: str = "face", min_photos: int = 3) -> bool:
    with get_db() as conn:
        col = 'face_photos' if emb_type == 'face' else 'iris_photos'
        row = conn.execute(f'SELECT {col} FROM students WHERE id=?', (student_id,)).fetchone()
    return row and row[0] >= min_photos


# ── Face / Iris Matching (scoped to dept + batch) ─────────────────────────────

def find_match(query_emb: np.ndarray, department_id: int, batch_id: int,
               emb_type: str = "face", threshold: float = None) -> dict:
    """
    Match query embedding against students in a specific department + batch only.
    Returns best match above threshold or a no-match dict.
    """
    if threshold is None:
        try:
            import embedder
            threshold = embedder.get_threshold()
        except Exception:
            threshold = 0.55

    with get_db() as conn:
        rows = conn.execute(
            '''SELECT s.id, s.roll_no, s.name, me.emb_data
               FROM mean_embeddings me
               JOIN students s ON s.id = me.student_id
               WHERE s.department_id=? AND s.batch_id=? AND me.emb_type=?''',
            (department_id, batch_id, emb_type)
        ).fetchall()

    if not rows:
        return {
            "matched": False, "student_id": None, "roll_no": None,
            "name": "Unknown", "confidence": 0.0,
            "message": "No enrolled students found in this batch.",
        }

    q = query_emb.astype(np.float32)
    best_sim  = -1.0
    best_row  = None

    for row in rows:
        mean_emb = _blob_to_emb(row['emb_data'])
        sim = float(np.dot(q, mean_emb))
        if sim > best_sim:
            best_sim = sim
            best_row = row

    confidence = round(best_sim * 100, 1)

    if best_row is None or best_sim < threshold:
        return {
            "matched": False, "student_id": None, "roll_no": None,
            "name": "Unknown", "confidence": confidence,
            "message": f"No match found (best score: {confidence}%).",
        }

    return {
        "matched": True,
        "student_id": best_row['id'],
        "roll_no":    best_row['roll_no'],
        "name":       best_row['name'],
        "confidence": confidence,
        "message":    "Match found.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════

def mark_attendance(student_id: int, subject_id: int, method: str = "face") -> tuple:
    """
    Mark attendance for a student today for a given subject.
    UNIQUE constraint on (student_id, subject_id, date) prevents duplicates.
    Returns (True, message) or (False, reason).
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now   = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        with get_db() as conn:
            conn.execute(
                'INSERT INTO attendance (student_id, subject_id, date, time, method) VALUES (?,?,?,?,?)',
                (student_id, subject_id, today, now, method)
            )
        return True, "Attendance marked."
    except sqlite3.IntegrityError:
        return False, "Already marked present today."


def is_marked_today(student_id: int, subject_id: int) -> bool:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    with get_db() as conn:
        row = conn.execute(
            'SELECT 1 FROM attendance WHERE student_id=? AND subject_id=? AND date=?',
            (student_id, subject_id, today)
        ).fetchone()
    return row is not None


def get_today_attendance(subject_id: int, department_id: int, batch_id: int) -> list:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            '''SELECT a.id, a.date, a.time, a.method,
                      s.roll_no, s.name
               FROM attendance a
               JOIN students s ON s.id = a.student_id
               WHERE a.subject_id=? AND a.date=?
                 AND s.department_id=? AND s.batch_id=?
               ORDER BY a.time''',
            (subject_id, today, department_id, batch_id)
        ).fetchall()]


def get_attendance_report(subject_id: int, department_id: int, batch_id: int,
                          date_from: str = None, date_to: str = None) -> list:
    query = '''
        SELECT a.id, a.date, a.time, a.method,
               s.roll_no, s.name,
               sub.name AS subject_name,
               d.name   AS dept_name,
               b.name   AS batch_name
        FROM attendance a
        JOIN students   s   ON s.id   = a.student_id
        JOIN subjects   sub ON sub.id = a.subject_id
        JOIN departments d  ON d.id   = s.department_id
        JOIN batches     b  ON b.id   = s.batch_id
        WHERE a.subject_id=?
          AND s.department_id=? AND s.batch_id=?
    '''
    params = [subject_id, department_id, batch_id]

    if date_from:
        query += ' AND a.date >= ?'
        params.append(date_from)
    if date_to:
        query += ' AND a.date <= ?'
        params.append(date_to)

    query += ' ORDER BY a.date DESC, a.time DESC'

    with get_db() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_attendance_summary(subject_id: int, department_id: int, batch_id: int) -> list:
    """Per-student total days present."""
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            '''SELECT s.roll_no, s.name, COUNT(a.id) AS days_present
               FROM students s
               LEFT JOIN attendance a ON a.student_id=s.id AND a.subject_id=?
               WHERE s.department_id=? AND s.batch_id=?
               GROUP BY s.id
               ORDER BY s.roll_no''',
            (subject_id, department_id, batch_id)
        ).fetchall()]
