"""
app.py — Flask web application
Face & Iris Recognition Attendance System
"""

import os
import io
import base64
import datetime
import csv

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, Response, send_file
)

import database as db
import embedder
import iris_embedder

# ── App setup ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "face-attend-secret-2024")
app.config["SESSION_COOKIE_HTTPONLY"] = True

# Initialise DB on startup
with app.app_context():
    db.init_db()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "teacher_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def context_required(f):
    """Require dept / batch / subject to be selected."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "teacher_id" not in session:
            return redirect(url_for("login"))
        if not session.get("dept_id") or not session.get("batch_id") or not session.get("subject_id"):
            flash("Please select department, batch and subject first.", "info")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


def decode_image(b64: str) -> np.ndarray:
    """Decode a base64 data-URL or raw base64 string to RGB numpy array."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if "teacher_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        teacher  = db.authenticate_teacher(username, password)
        if teacher:
            session["teacher_id"]       = teacher["id"]
            session["teacher_username"] = teacher["username"]
            session["teacher_name"]     = teacher["name"] or teacher["username"]
            # Clear any old context
            for k in ("dept_id","dept_name","batch_id","batch_name","subject_id","subject_name"):
                session.pop(k, None)
            flash(f"Welcome back, {session['teacher_name']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        name     = request.form.get("name", "").strip()
        ok, msg  = db.register_teacher(username, password, name)
        if ok:
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        flash(msg, "danger")
    return render_template("login.html", signup=True)


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD — Select Dept / Batch / Subject
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        dept_id    = int(request.form["dept_id"])
        batch_id   = int(request.form["batch_id"])
        subject_id = int(request.form["subject_id"])

        depts    = {d["id"]: d["name"] for d in db.get_departments()}
        batches  = {b["id"]: b["name"] for b in db.get_batches(dept_id)}
        subjects = {s["id"]: s["name"] for s in db.get_subjects(dept_id)}

        session["dept_id"]      = dept_id
        session["dept_name"]    = depts.get(dept_id, "")
        session["batch_id"]     = batch_id
        session["batch_name"]   = batches.get(batch_id, "")
        session["subject_id"]   = subject_id
        session["subject_name"] = subjects.get(subject_id, "")

        flash(f"Context set: {session['dept_name']} / {session['batch_name']} / {session['subject_name']}", "success")
        return redirect(url_for("students"))

    departments = db.get_departments()
    return render_template("dashboard.html", departments=departments)


# ══════════════════════════════════════════════════════════════════════════════
#  API — Departments / Batches / Subjects  (for dynamic dropdowns + adding new)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/departments", methods=["GET", "POST"])
@login_required
def api_departments():
    if request.method == "POST":
        data = request.get_json()
        ok, msg = db.add_department(data.get("name", ""))
        return jsonify({"ok": ok, "message": msg})
    return jsonify(db.get_departments())


@app.route("/api/departments/<int:dept_id>/batches", methods=["GET"])
@login_required
def api_batches(dept_id):
    return jsonify(db.get_batches(dept_id))


@app.route("/api/departments/<int:dept_id>/subjects", methods=["GET"])
@login_required
def api_subjects(dept_id):
    return jsonify(db.get_subjects(dept_id))


@app.route("/api/batches", methods=["POST"])
@login_required
def api_add_batch():
    data = request.get_json()
    ok, msg = db.add_batch(data.get("name", ""), data.get("department_id"))
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/subjects", methods=["POST"])
@login_required
def api_add_subject():
    data = request.get_json()
    ok, msg = db.add_subject(data.get("name", ""), data.get("department_id"))
    return jsonify({"ok": ok, "message": msg})


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/students")
@context_required
def students():
    student_list = db.get_students(session["dept_id"], session["batch_id"])
    return render_template("students.html", students=student_list)


@app.route("/api/students", methods=["POST"])
@context_required
def api_register_student():
    data = request.get_json()
    ok, msg = db.register_student(
        roll_no=data.get("roll_no", ""),
        name=data.get("name", ""),
        department_id=session["dept_id"],
        batch_id=session["batch_id"],
    )
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/students/<int:student_id>", methods=["PUT", "DELETE"])
@context_required
def api_student(student_id):
    if request.method == "PUT":
        data = request.get_json()
        ok, msg = db.update_student_name(student_id, data.get("name", ""))
        return jsonify({"ok": ok, "message": msg})
    else:
        ok, msg = db.delete_student(student_id)
        return jsonify({"ok": ok, "message": msg})


@app.route("/api/students/<int:student_id>/clear-embeddings", methods=["POST"])
@context_required
def api_clear_embeddings(student_id):
    data     = request.get_json() or {}
    emb_type = data.get("type", "face")
    ok, msg  = db.clear_embeddings(student_id, emb_type)
    return jsonify({"ok": ok, "message": msg})


# ── Enrollment endpoints (capture from webcam or upload) ──────────────────────

@app.route("/students/<int:student_id>/enroll")
@context_required
def enroll_student(student_id):
    student = db.get_student(student_id)
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("students"))
    return render_template("enroll.html", student=student)


@app.route("/api/students/<int:student_id>/enroll", methods=["POST"])
@context_required
def api_enroll(student_id):
    data     = request.get_json()
    emb_type = data.get("type", "face")   # 'face' or 'iris'
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"ok": False, "message": "No image provided."})

    try:
        img_rgb = decode_image(image_b64)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Image decode error: {e}"})

    if emb_type == "face":
        emb, msg = embedder.extract_embedding(img_rgb)
    else:
        emb, msg = iris_embedder.extract_iris_embedding(img_rgb)

    if emb is None:
        return jsonify({"ok": False, "message": msg})

    ok, msg = db.add_embedding(student_id, emb, emb_type)
    student = db.get_student(student_id)

    return jsonify({
        "ok":        ok,
        "message":   msg,
        "face_photos": student["face_photos"],
        "iris_photos": student["iris_photos"],
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENDANCE — Video / Live Mode
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/attendance/mark")
@context_required
def attendance_mark():
    today_records = db.get_today_attendance(
        session["subject_id"], session["dept_id"], session["batch_id"]
    )
    total_students = len(db.get_students(session["dept_id"], session["batch_id"]))
    return render_template("attendance_mark.html",
                           today_records=today_records,
                           total_students=total_students)


@app.route("/api/attendance/frame", methods=["POST"])
@context_required
def api_attendance_frame():
    """
    Process one video frame for attendance marking.
    Called every ~2 seconds from the frontend camera feed.
    """
    data      = request.get_json()
    image_b64 = data.get("image", "")
    mode      = data.get("mode", "face")   # 'face' or 'iris'

    if not image_b64:
        return jsonify({"ok": False, "message": "No image."})

    try:
        img_rgb = decode_image(image_b64)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Decode error: {e}"})

    # Extract embedding
    if mode == "face":
        emb, msg = embedder.extract_embedding(img_rgb)
    else:
        emb, msg = iris_embedder.extract_iris_embedding(img_rgb)

    if emb is None:
        return jsonify({"ok": False, "detected": False, "message": msg})

    # Find match scoped to this dept + batch only
    match = db.find_match(
        query_emb=emb,
        department_id=session["dept_id"],
        batch_id=session["batch_id"],
        emb_type=mode,
    )

    if not match["matched"]:
        return jsonify({
            "ok":        False,
            "detected":  True,
            "matched":   False,
            "message":   match["message"],
            "confidence": match["confidence"],
        })

    student_id = match["student_id"]
    subject_id = session["subject_id"]

    already = db.is_marked_today(student_id, subject_id)
    if already:
        return jsonify({
            "ok":         True,
            "detected":   True,
            "matched":    True,
            "already":    True,
            "name":       match["name"],
            "roll_no":    match["roll_no"],
            "confidence": match["confidence"],
            "message":    f"{match['name']} already marked today.",
        })

    ok, msg = db.mark_attendance(student_id, subject_id, method=mode)
    return jsonify({
        "ok":         ok,
        "detected":   True,
        "matched":    True,
        "already":    False,
        "new_mark":   ok,
        "name":       match["name"],
        "roll_no":    match["roll_no"],
        "confidence": match["confidence"],
        "message":    msg,
    })


@app.route("/api/attendance/today")
@context_required
def api_today_attendance():
    records = db.get_today_attendance(
        session["subject_id"], session["dept_id"], session["batch_id"]
    )
    return jsonify(records)


# ══════════════════════════════════════════════════════════════════════════════
#  ATTENDANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/attendance/report")
@context_required
def attendance_report():
    date_from = request.args.get("from", "")
    date_to   = request.args.get("to",   "")

    records  = db.get_attendance_report(
        session["subject_id"], session["dept_id"], session["batch_id"],
        date_from or None, date_to or None,
    )
    summary  = db.get_attendance_summary(
        session["subject_id"], session["dept_id"], session["batch_id"]
    )
    students = db.get_students(session["dept_id"], session["batch_id"])

    return render_template("attendance_report.html",
                           records=records,
                           summary=summary,
                           total_students=len(students),
                           date_from=date_from,
                           date_to=date_to)


@app.route("/attendance/export")
@context_required
def attendance_export():
    date_from = request.args.get("from", "")
    date_to   = request.args.get("to",   "")

    records = db.get_attendance_report(
        session["subject_id"], session["dept_id"], session["batch_id"],
        date_from or None, date_to or None,
    )

    def generate():
        fieldnames = ["date", "time", "roll_no", "name", "subject_name",
                      "dept_name", "batch_name", "method"]
        yield ",".join(fieldnames) + "\n"
        for r in records:
            row = [str(r.get(f, "")) for f in fieldnames]
            yield ",".join(row) + "\n"

    filename = (
        f"attendance_{session['subject_name']}_{session['batch_name']}"
        f"_{date_from or 'all'}.csv"
    ).replace(" ", "_")

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return render_template("login.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
