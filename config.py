"""
DetectAI - Configuration & System Constants
Centralized settings for deepfake detection pipeline, model paths,
ensemble weights, video sampling rates, and forensic parameters.
"""

import os
from pathlib import Path
import cv2

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Storage and assets directories
MODEL_DIR = BASE_DIR / "model"
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DB_PATH = BASE_DIR / "history.db"

# Create directories if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model file paths
CNN_MODEL_PATH = MODEL_DIR / "model.h5"
RF_MODEL_PATH = MODEL_DIR / "rf_model.pkl"

# Face detection parameters
LOCAL_CASCADE = MODEL_DIR / "haarcascade_frontalface_default.xml"
if LOCAL_CASCADE.exists():
    HAAR_CASCADE_PATH = str(LOCAL_CASCADE)
elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and os.path.exists(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"):
    HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
else:
    HAAR_CASCADE_PATH = str(LOCAL_CASCADE)
FACE_MARGIN_RATIO = 0.25  # 25% boundary expansion around detected face
TARGET_SIZE = (224, 224)  # Standard input resolution for CNN & forensic extractor

# Hybrid Ensemble Weights (Must sum to 1.0)
CNN_WEIGHT = 0.60
RF_WEIGHT = 0.40

# Default classification threshold (probability >= threshold -> FAKE)
DEFAULT_DECISION_THRESHOLD = 0.50

# Video pipeline sampling
VIDEO_SAMPLE_INTERVAL = 10  # Process 1 frame every 10 frames
MAX_VIDEO_FRAMES = 60       # Cap on total frames to analyze per video

# File constraints
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)

# Known file signature magic bytes for security verification
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"ftyp": "video/mp4",
    b"RIFF": "video/avi",
    b"\x00\x00\x00\x14ftypqt": "video/quicktime",
    b"\x00\x00\x00\x20ftyp": "video/mp4",
}

# Branding & Theme Tokens
APP_NAME = "DetectAI"
APP_TAGLINE = "See through the fake."
APP_VERSION = "2.0.0"

THEME_COLORS = {
    "purple_900": "#2e1065",
    "purple_700": "#6d28d9",
    "purple_500": "#8b5cf6",
    "lavender": "#c4b5fd",
    "lilac": "#e9d5ff",
    "bg_dark": "#140a2b",
    "card": "#1f1240",
    "real_green": "#10b981",
    "fake_red": "#ef4444",
}
