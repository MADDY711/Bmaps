"""
============================================
Configuration file for AI Blind Assistant
============================================
Central place for all settings and constants.
Modify values here instead of in individual modules.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (check both current and parent directory)
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ============================================
# Camera Settings
# ============================================
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
try:
    # If it's a digit (like "0"), convert to int for local webcam
    CAMERA_INDEX = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
except:
    CAMERA_INDEX = CAMERA_SOURCE

FRAME_WIDTH = 640             # Frame width in pixels (lower = faster)
FRAME_HEIGHT = 480            # Frame height in pixels (lower = faster)
TARGET_FPS = 30               # Target frames per second

# ============================================
# YOLOv8 Settings
# ============================================
YOLO_MODEL = "yolov8n-seg.pt"    # Nano segmentation model for speed
YOLO_CONFIDENCE = 0.5            # Minimum confidence to accept a detection (0-1)

# ============================================
# Depth Estimation (MDE) Settings
# ============================================
MDE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # Official HF ID
HAZARD_PROCESS_N_FRAMES = 5      # Run depth analysis every 5th frame (heavy!)
POTHOLE_THRESHOLD = 40           # Difference in depth to trigger pothole alert
STAIRS_THRESHOLD = 8             # Strict horizontal threshold for stairs

# ============================================
# Obstacle Detection Settings
# ============================================
# Objects we consider as obstacles (COCO dataset class names)
OBSTACLE_CLASSES = ["chair", "person", "dining table", "couch", "bed",
                    "dog", "cat", "backpack", "suitcase", "bottle",
                    "potted plant", "tv", "laptop", "cell phone", "book"]

# Screen region boundaries (percentage of frame width)
LEFT_REGION = (0.0, 0.33)     # 0% to 33% of frame width
CENTER_REGION = (0.33, 0.66)  # 33% to 66% of frame width
RIGHT_REGION = (0.66, 1.0)    # 66% to 100% of frame width

# Distance estimation thresholds (based on bounding box area ratio)
# Ratio = (bbox_area / frame_area)
CLOSE_THRESHOLD = 0.15        # Object is CLOSE if bbox covers >15% of frame
MEDIUM_THRESHOLD = 0.05       # Object is MEDIUM if bbox covers >5% of frame
# Below MEDIUM_THRESHOLD = FAR

# Safety: minimum bbox area ratio to trigger STOP command
SAFETY_STOP_THRESHOLD = 0.12  # STOP if center obstacle covers >12% of frame

# ============================================
# Voice Output Settings
# ============================================
# ElevenLabs API
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default "Rachel" voice

# pyttsx3 fallback settings
PYTTSX3_RATE = 175            # Words per minute
PYTTSX3_VOLUME = 1.0          # Volume (0.0 to 1.0)

# Speech cooldown: minimum seconds between repeated instructions
SPEECH_COOLDOWN = 2.0         # Avoid spamming the same instruction

# ============================================
# Voice Command Settings
# ============================================
VOICE_COMMANDS = {
    "start": ["start navigation", "start", "go", "begin"],
    "stop": ["stop navigation", "stop", "halt", "pause"],
}

# ============================================
# OpenAI / Gemini Settings
# ============================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
COGNITIVE_SENSITIVITY = 0.8  # Threshold to trigger LLM on high-danger hazards

# ============================================
# Performance Settings
# ============================================
PROCESS_EVERY_N_FRAMES = 2    # Run YOLO on every Nth frame to save CPU
MAX_RESPONSE_TIME = 1.0       # Maximum allowed response time in seconds
