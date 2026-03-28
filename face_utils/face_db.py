"""
face_db.py — Student face embedding database
Stores embeddings as pickle. No retraining ever needed.
"""

import os
import pickle
import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "face_db.pkl")


def _ensure():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _load_db() -> dict:
    _ensure()
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_db(db: dict):
    _ensure()
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


# ── CRUD ──────────────────────────────────────────────────────────────────────

def register_student(roll_no: str, name: str) -> tuple:
    roll_no = str(roll_no).strip()
    name    = name.strip()
    db = _load_db()
    if roll_no in db:
        return False, f"Roll number {roll_no} is already registered."
    db[roll_no] = {
        "roll_no":    roll_no,
        "name":       name,
        "embeddings": [],
        "mean_emb":   None,
        "n_photos":   0,
        "registered": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save_db(db)
    return True, f"Student '{name}' registered successfully."


def update_student(roll_no: str, new_name: str) -> tuple:
    db = _load_db()
    if str(roll_no) not in db:
        return False, "Student not found."
    db[str(roll_no)]["name"] = new_name.strip()
    _save_db(db)
    return True, "Name updated successfully."


def delete_student(roll_no: str) -> tuple:
    db = _load_db()
    if str(roll_no) not in db:
        return False, "Student not found."
    del db[str(roll_no)]
    _save_db(db)
    return True, "Student deleted."


def add_embedding(roll_no: str, embedding: np.ndarray) -> tuple:
    """Add one face embedding. Recomputes mean embedding automatically."""
    db = _load_db()
    if str(roll_no) not in db:
        return False, "Student not found. Register the student first."
    db[str(roll_no)]["embeddings"].append(embedding.astype(np.float32))
    db[str(roll_no)]["n_photos"] = len(db[str(roll_no)]["embeddings"])
    # Recompute L2-normalised mean embedding
    embs = np.array(db[str(roll_no)]["embeddings"])
    mean = embs.mean(axis=0)
    norm = np.linalg.norm(mean)
    db[str(roll_no)]["mean_emb"] = (mean / (norm + 1e-8)).astype(np.float32)
    _save_db(db)
    return True, f"Photo {db[str(roll_no)]['n_photos']} enrolled."


def clear_embeddings(roll_no: str) -> tuple:
    """Remove all enrolled photos for a student (allow re-enrollment)."""
    db = _load_db()
    if str(roll_no) not in db:
        return False, "Student not found."
    db[str(roll_no)]["embeddings"] = []
    db[str(roll_no)]["mean_emb"]   = None
    db[str(roll_no)]["n_photos"]   = 0
    _save_db(db)
    return True, "All photos cleared. Please re-enroll the student."


def get_all_students() -> list:
    return list(_load_db().values())


def get_student(roll_no: str):
    return _load_db().get(str(roll_no))


def is_enrolled(roll_no: str) -> bool:
    s = get_student(roll_no)
    return s is not None and s["n_photos"] >= 3


# ── Face matching ─────────────────────────────────────────────────────────────

def find_match(query_emb: np.ndarray, threshold: float = None) -> dict:
    """
    Compare query embedding against all enrolled students.
    Uses cosine similarity (dot product of L2-normalised vectors).
    Returns best match above threshold, or a no-match dict.
    """
    if threshold is None:
        import embedder
        threshold = embedder.get_threshold()

    db       = _load_db()
    enrolled = {k: v for k, v in db.items() if v["mean_emb"] is not None}

    if not enrolled:
        return {
            "matched":    False,
            "roll_no":    None,
            "name":       "Unknown",
            "confidence": 0.0,
            "message":    "No students enrolled. Please enroll students first.",
        }

    best_sim  = -1.0
    best_roll = None

    q = query_emb.astype(np.float32)

    for roll_no, data in enrolled.items():
        # Similarity to mean embedding
        sim = float(np.dot(q, data["mean_emb"]))
        # Also check individual embeddings — take the best
        for emb in data["embeddings"]:
            s = float(np.dot(q, emb.astype(np.float32)))
            if s > sim:
                sim = s
        if sim > best_sim:
            best_sim  = sim
            best_roll = roll_no

    confidence = round(best_sim * 100, 1)

    if best_roll is None or best_sim < threshold:
        return {
            "matched":    False,
            "roll_no":    None,
            "name":       "Unknown",
            "confidence": confidence,
            "message":    f"No match found (best score: {confidence}%). "
                          "Face not enrolled or image unclear.",
        }

    student = db[best_roll]
    return {
        "matched":    True,
        "roll_no":    best_roll,
        "name":       student["name"],
        "confidence": confidence,
        "message":    "OK",
    }