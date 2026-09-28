"""Central configuration for the room-recognition project."""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATABASE_DIR = PROJECT_ROOT / "room_database"

# The database records this name so inference cannot silently use another model.
MODEL_NAME = os.getenv("ROOM_MODEL_NAME", "facebook/dinov2-small")
_REQUESTED_DEVICE = os.getenv("ROOM_DEVICE", "cuda")
DEVICE = _REQUESTED_DEVICE if not _REQUESTED_DEVICE.startswith("cuda") or torch.cuda.is_available() else "cpu"
INFERENCE_BATCH_SIZE = int(os.getenv("ROOM_BATCH_SIZE", "16" if DEVICE.startswith("cuda") else "4"))

# Reference-video sampling and quality filtering.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
FRAME_SAMPLE_INTERVAL = float(os.getenv("ROOM_SAMPLE_INTERVAL", "0.5"))  # seconds
MIN_BLUR_SCORE = float(os.getenv("ROOM_MIN_BLUR", "60.0"))
MIN_BRIGHTNESS = float(os.getenv("ROOM_MIN_BRIGHTNESS", "22.0"))
MAX_DARK_PIXEL_RATIO = float(os.getenv("ROOM_MAX_DARK_RATIO", "0.92"))
DARK_PIXEL_THRESHOLD = int(os.getenv("ROOM_DARK_PIXEL_THRESHOLD", "25"))
DUPLICATE_SIMILARITY = float(os.getenv("ROOM_DUPLICATE_SIMILARITY", "0.992"))

# Live retrieval and open-set recognition.
CAMERA_ID = int(os.getenv("ROOM_CAMERA_ID", "0"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("ROOM_PROCESS_EVERY", "3"))
TOP_K = int(os.getenv("ROOM_TOP_K", "10"))
ROOM_MATCH_COUNT = int(os.getenv("ROOM_MATCH_COUNT", "3"))
MIN_SIMILARITY = float(os.getenv("ROOM_MIN_SIMILARITY", "0.54"))
MIN_SCORE_MARGIN = float(os.getenv("ROOM_MIN_SCORE_MARGIN", "0.015"))
PROTOTYPE_WEIGHT = float(os.getenv("ROOM_PROTOTYPE_WEIGHT", "0.20"))
MATCH_MODE = os.getenv("ROOM_MATCH_MODE", "frames")  # frames, prototypes, hybrid

# Temporal stability. Counts are processed/inference frames, not display frames.
PREDICTION_WINDOW = int(os.getenv("ROOM_PREDICTION_WINDOW", "15"))
SWITCH_CONFIRMATION_FRAMES = int(os.getenv("ROOM_SWITCH_FRAMES", "5"))
UNKNOWN_CONFIRMATION_FRAMES = int(os.getenv("ROOM_UNKNOWN_FRAMES", "8"))

# Camera display.
WINDOW_NAME = "Real-time Room Recognition"
CAMERA_WIDTH = int(os.getenv("ROOM_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("ROOM_CAMERA_HEIGHT", "720"))
DISPLAY_WIDTH = int(os.getenv("ROOM_DISPLAY_WIDTH", "1280"))  # 0 keeps native width
