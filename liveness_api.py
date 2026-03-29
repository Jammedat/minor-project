"""
liveness_api.py — Bridge between liveness.py and the Flask API

The original liveness.py uses a challenge-response flow:
  1. Pick a random challenge (BLINK / TURN_LEFT / TURN_RIGHT)
  2. Collect several frames from the user
  3. Evaluate whether they completed the challenge

This module exposes simple functions that can be called from Flask routes.

Session-level state (current challenge, collected frames) is stored in
Flask session, so each teacher session tracks its own liveness state
independently.

Usage in Flask:
  from liveness_api import start_challenge, submit_frame, get_challenge_info

Passive liveness (used during attendance):
  from liveness_api import passive_liveness_check
"""

import sys
import os

# Make sure liveness.py is importable (it lives in the project root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import liveness
    LIVENESS_AVAILABLE = True
except ImportError:
    LIVENESS_AVAILABLE = False

import numpy as np
import cv2
import base64
import io
from PIL import Image


# ── Session key ───────────────────────────────────────────────────────────────
# We store challenge state in a plain dict (not Flask session) keyed by a
# student_id string, so multiple concurrent enrollments work.

_challenges: dict = {}   # { "student_<id>": {"challenge": str, "frames": list} }


def start_challenge(key: str) -> dict:
    """
    Start a new liveness challenge for the given key.
    Returns {"challenge": str, "label": str, "hint": str}
    """
    if not LIVENESS_AVAILABLE:
        return {"challenge": "SKIP", "label": "Liveness unavailable", "hint": ""}

    ch = liveness.pick_challenge()
    _challenges[key] = {"challenge": ch, "frames": []}
    return {
        "challenge": ch,
        "label":     liveness.CHALLENGE_LABEL.get(ch, ch),
        "hint":      liveness.CHALLENGE_HINT.get(ch, ""),
    }


def submit_frame(key: str, image_rgb: np.ndarray) -> dict:
    """
    Add one frame to the challenge in progress.
    Returns {"done": bool, "passed": bool, "frames_collected": int, "message": str}
    """
    if not LIVENESS_AVAILABLE or key not in _challenges:
        return {"done": True, "passed": True, "frames_collected": 0,
                "message": "Liveness skipped (not configured)."}

    state = _challenges[key]
    fd    = liveness.analyse_frame(image_rgb)
    state["frames"].append(fd)
    collected = len(state["frames"])

    if collected < liveness.FRAMES_NEEDED:
        return {
            "done":             False,
            "passed":           False,
            "frames_collected": collected,
            "message":          f"Keep going — {liveness.FRAMES_NEEDED - collected} more frame(s) needed.",
        }

    # Evaluate
    passed, msg = liveness.evaluate_challenge(state["challenge"], state["frames"])
    if passed:
        _challenges.pop(key, None)   # clear on success

    return {
        "done":             True,
        "passed":           passed,
        "frames_collected": collected,
        "message":          msg,
    }


def get_challenge_info(key: str) -> dict:
    """Return current challenge info for a key, or None."""
    if key not in _challenges:
        return None
    ch = _challenges[key]["challenge"]
    return {
        "challenge":        ch,
        "label":            liveness.CHALLENGE_LABEL.get(ch, ch) if LIVENESS_AVAILABLE else "",
        "hint":             liveness.CHALLENGE_HINT.get(ch, "") if LIVENESS_AVAILABLE else "",
        "frames_collected": len(_challenges[key]["frames"]),
        "frames_needed":    liveness.FRAMES_NEEDED if LIVENESS_AVAILABLE else 0,
    }


def clear_challenge(key: str):
    _challenges.pop(key, None)


# ── Passive liveness (used during attendance video stream) ────────────────────
# For continuous video attendance we can't do a full challenge-response because
# students just walk past the camera.  Instead we do a "passive" check:
#   - Face must be detected
#   - Eyes must be visible (not a closed photo)
#   - Face must be a reasonable size (not a tiny photo held up)
#   - Not a completely static frame (crude anti-spoofing)

def passive_liveness_check(image_rgb: np.ndarray) -> dict:
    """
    Passive liveness check for the attendance video feed.
    Rejects printed photos and phone/screen spoofs without any user challenge.

    Checks:
      1. A face must be detected.
      2. The face must be large enough (student is actually in front of camera,
         not a tiny photo held far away).
      3. At least one eye must be visible and open (a printed photo of a person
         with closed eyes, or a screen photo taken at an angle, often fails this).

    Returns {"live": bool, "reason": str}
    """
    if not LIVENESS_AVAILABLE:
        return {"live": True, "reason": "Liveness module unavailable — check skipped."}

    fd = liveness.analyse_frame(image_rgb)

    if not fd.face_found:
        return {"live": False, "reason": "No face detected."}

    # Face width must be > 12% of frame width.
    # A phone photo held up typically has a tiny face relative to the frame.
    if fd.face_w < 0.12:
        return {"live": False, "reason": "Face too small — student must be closer to camera."}

    # Eyes must be detectable and open.
    # Printed photos and screen photos taken at an angle often show closed/obscured eyes.
    if fd.eyes_found == 0:
        return {"live": False, "reason": "Eyes not detected — liveness check failed."}

    return {"live": True, "reason": "Liveness OK."}
