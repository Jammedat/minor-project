"""
embedder.py — FaceEmbedNet loader + face embedding extractor
Loads the trained model from the Jupyter notebook output.

Required files (produced by train_from_scratch.ipynb Cell 7):
  data/models/face_embed_model.pth
  data/models/face_embed_config.json
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "data", "models", "face_embed_model.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "data", "models", "face_embed_config.json")

# Module-level cache so the model is only loaded once per session
_model     = None
_config    = None
_transform = None

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ── Model architecture — MUST match the notebook exactly ─────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FaceEmbedNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=False),
        )
        # Auto-compute flat size from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 62, 47)
            flat  = self.features(dummy).view(1, -1).shape[1]
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedder(x)
        return F.normalize(x, p=2, dim=1)


# ── Model loading ─────────────────────────────────────────────────────────────

def _load() -> tuple:
    """Load model into module-level cache. Returns (ok, error_message)."""
    global _model, _config, _transform

    if _model is not None:
        return True, ""

    if not os.path.exists(MODEL_PATH):
        return False, (
            "Model file not found: data/models/face_embed_model.pth\n"
            "Steps to fix:\n"
            "1. Run all cells in train_from_scratch.ipynb\n"
            "2. Copy saved_model/face_embed_model.pth  → data/models/\n"
            "3. Copy saved_model/face_embed_config.json → data/models/"
        )

    if not os.path.exists(CONFIG_PATH):
        return False, (
            "Config file not found: data/models/face_embed_config.json\n"
            "Copy saved_model/face_embed_config.json → data/models/"
        )

    try:
        with open(CONFIG_PATH, "r") as f:
            _config = json.load(f)

        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        _model = FaceEmbedNet(embedding_dim=_config.get("embedding_dim", 128))
        _model.load_state_dict(checkpoint["state_dict"])
        _model.eval()

        H = _config.get("img_height", 62)
        W = _config.get("img_width",  47)
        _transform = T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return True, ""

    except Exception as e:
        return False, f"Error loading model: {e}"


# ── Public API ────────────────────────────────────────────────────────────────

def model_ready() -> tuple:
    """Returns (bool, error_message). Call this to check if model is loaded."""
    return _load()


def get_threshold() -> float:
    """Return the best cosine-similarity threshold from the config."""
    _load()
    return float(_config.get("best_threshold", 0.55)) if _config else 0.55


def get_face_bbox(image_rgb: np.ndarray):
    """
    Detect the largest face in an RGB image.
    Returns (x, y, w, h) or None if no face found.
    """
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        return None
    return tuple(map(int, max(faces, key=lambda f: f[2] * f[3])))


def extract_embedding(image_rgb: np.ndarray) -> tuple:
    """
    Detect face → crop → run through FaceEmbedNet → return 128-d embedding.

    Parameters
    ----------
    image_rgb : np.ndarray  (H x W x 3, uint8, RGB)

    Returns
    -------
    (np.ndarray of shape (128,), message)  on success
    (None, error_message)                  on failure
    """
    ok, err = _load()
    if not ok:
        return None, err

    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_rgb = image_rgb[y:y+h, x:x+w]
    else:
        # No face detected — use the full image
        # (useful for pre-cropped enrollment photos)
        face_rgb = image_rgb

    try:
        pil = Image.fromarray(face_rgb.astype(np.uint8)).convert("RGB")
        t   = _transform(pil).unsqueeze(0)  # (1, 3, H, W)
        with torch.no_grad():
            emb = _model(t)
        return emb[0].cpu().numpy(), "OK"
    except Exception as e:
        return None, f"Embedding failed: {e}"