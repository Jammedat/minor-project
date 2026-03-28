"""
iris_embedder.py — Iris / eye-region embedding extractor

Strategy:
  1. Detect eyes with OpenCV Haar cascade
  2. Crop the best (largest) eye region
  3. Run the same FaceEmbedNet on the eye crop → 128-d embedding
  4. Use cosine similarity for matching (same as face matching)

This requires NO separate model training — the FaceEmbedNet already
captures texture / structural features that differ per individual eye.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_transform = None   # Set after model config is loaded


def _get_transform():
    global _transform
    if _transform is None:
        try:
            import embedder
            embedder._load()
            cfg = embedder._config or {}
            H = cfg.get("img_height", 62)
            W = cfg.get("img_width", 47)
        except Exception:
            H, W = 62, 47
        _transform = T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return _transform


def detect_eye_region(image_rgb: np.ndarray):
    """
    Detect the largest eye region in the image.
    Returns cropped eye RGB array or None.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # First try to find face, then look for eye inside it
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # Look in the upper half of the face
        upper_h = int(fh * 0.6)
        roi_gray = gray[fy: fy + upper_h, fx: fx + fw]
        roi_rgb  = image_rgb[fy: fy + upper_h, fx: fx + fw]
    else:
        roi_gray = gray
        roi_rgb  = image_rgb

    eyes = _eye_cascade.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )

    if len(eyes) == 0:
        # Fallback: try directly on full image
        eyes = _eye_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
        if len(eyes) == 0:
            return None, None
        ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
        eye_crop = image_rgb[ey: ey + eh, ex: ex + ew]
        bbox = (ex, ey, ew, eh)
    else:
        ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            abs_ex = fx + ex
            abs_ey = fy + ey
        else:
            abs_ex, abs_ey = ex, ey
        eye_crop = image_rgb[abs_ey: abs_ey + eh, abs_ex: abs_ex + ew]
        bbox = (abs_ex, abs_ey, ew, eh)

    if eye_crop.size == 0:
        return None, None

    return eye_crop, bbox


def extract_iris_embedding(image_rgb: np.ndarray) -> tuple:
    """
    Extract iris embedding from an RGB image.

    Returns
    -------
    (np.ndarray of shape (128,), message)  on success
    (None, error_message)                  on failure
    """
    import embedder

    ok, err = embedder._load()
    if not ok:
        return None, err

    eye_crop, bbox = detect_eye_region(image_rgb)
    if eye_crop is None:
        return None, "No eye detected. Look directly at the camera."

    try:
        transform = _get_transform()
        pil = Image.fromarray(eye_crop.astype(np.uint8)).convert("RGB")
        t   = transform(pil).unsqueeze(0)
        with torch.no_grad():
            emb = embedder._model(t)
        return emb[0].cpu().numpy(), "OK"
    except Exception as e:
        return None, f"Iris embedding failed: {e}"


def draw_iris_overlay(image_rgb: np.ndarray) -> np.ndarray:
    """Draw eye detection overlay on image for display."""
    draw = image_rgb.copy()
    _, bbox = detect_eye_region(image_rgb)
    if bbox:
        ex, ey, ew, eh = bbox
        cv2.rectangle(draw, (ex, ey), (ex + ew, ey + eh), (124, 58, 237), 2)
        cx, cy = ex + ew // 2, ey + eh // 2
        cv2.circle(draw, (cx, cy), min(ew, eh) // 3, (124, 58, 237), 1)
        cv2.circle(draw, (cx, cy), 3, (124, 58, 237), -1)
    return draw
