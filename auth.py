"""
auth.py — Teacher authentication
SHA-256 password hashing, CSV storage.
"""

import hashlib
import os
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEACHER_FILE  = os.path.join(BASE_DIR, "data", "teachers.csv")


def _ensure():
    os.makedirs(os.path.dirname(TEACHER_FILE), exist_ok=True)
    if not os.path.exists(TEACHER_FILE):
        pd.DataFrame(columns=["username", "password_hash"]).to_csv(
            TEACHER_FILE, index=False)


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def load_teachers() -> pd.DataFrame:
    _ensure()
    return pd.read_csv(TEACHER_FILE)


def register_teacher(username: str, password: str) -> tuple:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    df = load_teachers()
    if username in df["username"].values:
        return False, "Username already exists."
    new_row = pd.DataFrame([[username, _hash(password)]],
                           columns=["username", "password_hash"])
    pd.concat([df, new_row], ignore_index=True).to_csv(TEACHER_FILE, index=False)
    return True, "Registration successful."


def authenticate(username: str, password: str) -> bool:
    df = load_teachers()
    if username not in df["username"].values:
        return False
    stored = df.loc[df["username"] == username, "password_hash"].iloc[0]
    return stored == _hash(password)