"""student.py — Student records"""
import pickle, os, datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STUDENT_FILE = os.path.join(BASE_DIR, "data", "students.pkl")

def _ensure(): os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

def load_students():
    _ensure()
    if os.path.exists(STUDENT_FILE):
        with open(STUDENT_FILE, "rb") as f: return pickle.load(f)
    return []

def save_students(s):
    _ensure()
    with open(STUDENT_FILE, "wb") as f: pickle.dump(s, f)

def add_student(roll_no, name):
    students = load_students()
    if any(str(s["roll_no"]) == str(roll_no) for s in students):
        return False, f"Roll {roll_no} already exists"
    students.append({
        "roll_no": str(roll_no),
        "name": name,
        "registered_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "biometrics": []
    })
    save_students(students)
    return True, "Student added"

def mark_biometric(roll_no, biometric):
    students = load_students()
    for s in students:
        if str(s["roll_no"]) == str(roll_no):
            if biometric not in s.get("biometrics", []):
                s.setdefault("biometrics", []).append(biometric)
            save_students(students)
            return True, "Updated"
    return False, "Student not found"

def delete_student(roll_no):
    students = load_students()
    new = [s for s in students if str(s["roll_no"]) != str(roll_no)]
    if len(new) == len(students): return False, "Not found"
    save_students(new)
    return True, "Deleted"

def get_student(roll_no):
    return next((s for s in load_students() if str(s["roll_no"]) == str(roll_no)), None)
