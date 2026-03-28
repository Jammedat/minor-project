"""
attendance.py — Attendance tracking
Stores one CSV per subject in data/attendance/.
"""

import os
import datetime
import io
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATT_DIR  = os.path.join(BASE_DIR, "data", "attendance")

COLUMNS = ["Date", "RollNo", "Name", "Time", "Subject", "Method"]


def _ensure():
    os.makedirs(ATT_DIR, exist_ok=True)


def _path(subject: str) -> str:
    # Sanitise subject name so it's safe as a filename
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in subject).strip()
    return os.path.join(ATT_DIR, f"attendance_{safe}.csv")


def load_attendance(subject: str) -> pd.DataFrame:
    _ensure()
    fp = _path(subject)
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        # Ensure all expected columns exist
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=COLUMNS)


def save_attendance(subject: str, df: pd.DataFrame):
    _ensure()
    df.to_csv(_path(subject), index=False)


def mark_attendance(subject: str, roll_no: str, name: str,
                    method: str = "face") -> tuple:
    """
    Mark attendance for one student today.
    Returns (True, "Marked <name>") or (False, "already marked") — never raises.
    """
    df    = load_attendance(subject)
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now   = datetime.datetime.now().strftime("%H:%M:%S")

    already = (
        (df["Date"].astype(str) == today) &
        (df["RollNo"].astype(str) == str(roll_no))
    ).any()

    if already:
        return False, f"{name} is already marked present today."

    new_row = pd.DataFrame(
        [[today, str(roll_no), name, now, subject, method]],
        columns=COLUMNS,
    )
    df = pd.concat([df, new_row], ignore_index=True)
    save_attendance(subject, df)
    return True, f"Marked {name} present."


def get_all_subjects() -> list:
    _ensure()
    files = [f for f in os.listdir(ATT_DIR) if f.endswith(".csv")]
    return sorted([
        f.replace("attendance_", "").replace(".csv", "")
        for f in files
    ])


def load_all_attendance() -> pd.DataFrame:
    subjects = get_all_subjects()
    frames   = [load_attendance(s) for s in subjects]
    frames   = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")
        ws = writer.sheets["Attendance"]
        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4
    return buf.getvalue()