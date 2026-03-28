# FaceAttend — Flask Biometric Attendance System

Face & iris recognition attendance system built with Flask + SQLite.

---

## Project Layout

```
minor-project-main/
├── app.py                  ← NEW Flask app (replaces Streamlit app.py)
├── database.py             ← NEW SQLite database layer
├── iris_embedder.py        ← NEW iris extraction using eye cascade
├── embedder.py             ← Keep as-is (face embedding extractor)
├── liveness.py             ← Keep as-is
├── model.py                ← Keep as-is
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── students.html
│   ├── enroll.html
│   ├── attendance_mark.html
│   └── attendance_report.html
├── static/
│   ├── css/style.css
│   └── js/attendance.js
├── data/                   ← Auto-created on first run
│   ├── faceattend.db       ← SQLite database
│   └── models/             ← Put your trained model files here
│       ├── face_embed_model.pth
│       └── face_embed_config.json
├── requirements.txt
└── saved_model/            ← Already exists in your project
```

---

## Setup

### 1. Copy model files
```bash
mkdir -p data/models
cp saved_model/face_embed_model.pth  data/models/
cp saved_model/face_embed_config.json data/models/
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```
Open http://localhost:5000 in your browser.

---

## Teacher Flow

1. **Sign Up** — create a teacher account at `/signup`
2. **Dashboard** — select Department → Batch → Subject
3. **Students** — register students, enroll face & iris biometrics
4. **Mark Attendance** — camera runs automatically; students just look at it
5. **Reports** — view & export attendance as CSV

---

## How Attendance Works

- Go to **Mark Attendance**
- Click **Start** — the webcam activates
- The system scans a frame every 2 seconds (adjustable)
- It only matches against students in the **selected department + batch**
- When a face is recognised, attendance is marked automatically
- Each student can only be marked **once per day per subject**
- Switch between **Face** and **Iris** mode using the toggle

---

## Database Schema

| Table             | Purpose                                     |
|-------------------|---------------------------------------------|
| `teachers`        | Teacher login credentials                   |
| `departments`     | CS, ECE, Mechanical, Civil, etc.            |
| `batches`         | 2021-25, 2022-26, etc. (per department)     |
| `subjects`        | DBMS, OS, etc. (per department)             |
| `students`        | Roll no, name, dept, batch                  |
| `embeddings`      | Raw face/iris embeddings per student        |
| `mean_embeddings` | Averaged embedding for fast matching        |
| `attendance`      | Date, time, subject, method; unique per day |

---

## Enrollment Requirements
- Minimum **3 photos** for reliable face recognition
- Minimum **3 photos** for reliable iris recognition
- More photos = higher accuracy

---

## Notes
- The system uses your existing `embedder.py` (FaceEmbedNet) for face recognition
- Iris recognition uses the same model but applied to the cropped eye region
- No separate iris model training is needed
- Student data is scoped by department + batch — the matcher ONLY looks at students in the active context, making it fast even with thousands of students
