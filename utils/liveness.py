"""
liveness.py — Active challenge-response liveness detection

KEY INSIGHT — MIRROR IMAGE FIX:
  Webcam shows a MIRRORED image (like a selfie camera).
  When student turns head to THEIR LEFT:
    - Their face moves to the RIGHT side of the mirrored frame
    - face_cx INCREASES
  When student turns head to THEIR RIGHT:
    - Their face moves to the LEFT side of the mirrored frame
    - face_cx DECREASES

  So TURN_LEFT  -> max(cx) > baseline  (face moves right in frame)
     TURN_RIGHT -> min(cx) < baseline  (face moves left in frame)
  This is the SAME direction as natural expectation in a mirror.
  We simply require ANY clear shift, then label the direction by
  which way cx moved — no directional confusion.

For blink: Haar eye cascade only detects OPEN eyes.
  eyes_found == 0  =>  eyes closed
  eyes_found >= 1  =>  eyes open
"""

import cv2
import numpy as np
import random
from dataclasses import dataclass

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

CHALLENGES    = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]
FRAMES_NEEDED = 5

CHALLENGE_LABEL = {
    "BLINK":       "😉  Blink your eyes",
    "TURN_LEFT":   "⬅️  Turn your head LEFT",
    "TURN_RIGHT":  "➡️  Turn your head RIGHT",
}
CHALLENGE_HINT = {
    "BLINK":
        "Look at camera. Close BOTH eyes fully for at least one photo, "
        "then open them again. Capture photos at each step.",
    "TURN_LEFT":
        "In the mirror you see yourself — turn YOUR head to YOUR LEFT "
        "(your left ear moves toward the camera side). "
        "Start straight, then turn clearly.",
    "TURN_RIGHT":
        "In the mirror you see yourself — turn YOUR head to YOUR RIGHT "
        "(your right ear moves toward the camera side). "
        "Start straight, then turn clearly.",
}


@dataclass
class FrameData:
    face_found: bool  = False
    eyes_found: int   = 0      # 0 = closed/not detected, >=1 = open
    face_cx:    float = 0.0    # face centre-x as fraction of image width (0..1)
    face_w:     float = 0.0    # face width as fraction of image width (0..1)


def analyse_frame(image_rgb: np.ndarray) -> FrameData:
    result = FrameData()
    if image_rgb is None or image_rgb.size == 0:
        return result

    img_h, img_w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Lenient detection — works at normal 50-80cm webcam distance
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return result

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    result.face_found = True
    result.face_cx    = (fx + fw / 2) / max(img_w, 1)   # 0..1
    result.face_w     = fw / max(img_w, 1)               # 0..1

    # Eye detection — upper 55% of face only
    upper_h = int(fh * 0.55)
    upper   = gray[fy: fy + upper_h, fx: fx + fw]
    eyes    = _eye_cascade.detectMultiScale(
        upper,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(10, 10),
    )
    result.eyes_found = len(eyes)
    return result


def evaluate_challenge(challenge: str, frames: list) -> tuple:
    valid = [f for f in frames if f.face_found]

    if len(valid) < 2:
        return False, (
            "Face not detected in enough frames. "
            "Make sure your face is well-lit and fully visible."
        )

    # ── BLINK ─────────────────────────────────────────────────────────────────
    if challenge == "BLINK":
        n_closed = sum(1 for f in valid if f.eyes_found == 0)
        n_open   = sum(1 for f in valid if f.eyes_found >= 1)

        if n_closed >= 1 and n_open >= 1:
            return True, "Blink confirmed ✓"
        if n_closed == 0:
            return False, (
                "Blink not detected. Close your eyes FULLY for at least one capture."
            )
        return False, (
            "Eyes not seen open. Start with eyes open, then close them."
        )

    # ── HEAD TURN (mirror-aware) ───────────────────────────────────────────────
    #
    # MIRROR LOGIC:
    #   Student turns LEFT  → face moves to frame-RIGHT  → cx goes UP
    #   Student turns RIGHT → face moves to frame-LEFT   → cx goes DOWN
    #
    # We measure: did cx shift enough? And did it shift the right way?
    #
    cx_values = [f.face_cx for f in valid]
    fw_values = [f.face_w  for f in valid]

    cx_min = min(cx_values)
    cx_max = max(cx_values)
    cx_range = cx_max - cx_min
    fw_range = max(fw_values) - min(fw_values)

    # Any significant movement
    turned = cx_range >= 0.07 or fw_range >= 0.05

    # Use first frame as baseline (student is straight at start)
    baseline = cx_values[0]

    if challenge == "TURN_LEFT":
        # Mirror: turning left → cx increases (face moves right in frame)
        correct = cx_max > baseline + 0.05
    else:  # TURN_RIGHT
        # Mirror: turning right → cx decreases (face moves left in frame)
        correct = cx_min < baseline - 0.05

    if turned and correct:
        return True, "Head turn confirmed ✓"

    if turned and not correct:
        # They moved but wrong way — likely confused by mirror
        # Be forgiving: accept any clear turn >= 0.10 regardless of direction
        # (some cameras don't mirror, some students are confused)
        if cx_range >= 0.10 or fw_range >= 0.08:
            return True, "Head turn confirmed ✓"
        return False, (
            "You turned but in the unexpected direction. "
            "Try turning the other way, or turn MORE clearly."
        )

    return False, (
        f"Head movement too small ({cx_range*100:.0f}% of frame detected, need ~7%). "
        "Turn your head MORE clearly and take photos mid-turn."
    )


def pick_challenge() -> str:
    return random.choice(CHALLENGES)