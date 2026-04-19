"""
============================================
Step 8: Navigation Decision Engine Test
============================================
Full pipeline test: Camera → YOLO → Obstacles → Navigation.
Displays live video with navigation instructions overlaid
and printed to console.

How to run:
    C:\\ba_venv\\Scripts\\python.exe step8_test_navigator.py

Expected output:
    - Live feed with region lines + obstacle boxes
    - Navigation instruction displayed on screen
    - Instructions printed to console with timestamps
    - "STOP" when obstacle is close + center
    - "Move left/right" for side obstacles
    - "Path is clear" when nothing is blocking

Controls:
    - Press 'q' to quit

Debugging tips:
    - No instructions? Objects may be FAR (move closer)
    - Always "path clear"? Check OBSTACLE_CLASSES in config.py
    - Too many instructions? Increase SPEECH_COOLDOWN in config.py
============================================
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.camera import Camera
from modules.detector import ObjectDetector
from modules.obstacle import ObstacleDetector
from modules.navigator import Navigator
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONFIDENCE, PROCESS_EVERY_N_FRAMES,
    OBSTACLE_CLASSES, CLOSE_THRESHOLD, MEDIUM_THRESHOLD,
    SAFETY_STOP_THRESHOLD, SPEECH_COOLDOWN
)

# Colors (BGR)
INSTRUCTION_COLORS = {
    "STOP": (0, 0, 255),       # Red
    "CAUTION": (0, 165, 255),  # Orange
    "MOVE_RIGHT": (255, 200, 0),  # Cyan-ish
    "MOVE_LEFT": (255, 200, 0),
    "CLEAR": (0, 255, 0),      # Green
}

DANGER_COLORS = {
    "HIGH": (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW": (0, 255, 0),
}


def draw_full_hud(frame, fps, obstacles, instruction, left_b, right_b):
    """Draw the complete heads-up display."""
    h, w = frame.shape[:2]

    # ---- Region divider lines ----
    dash = 15
    for y in range(0, h, dash * 2):
        cv2.line(frame, (left_b, y), (left_b, min(y + dash, h)), (255, 255, 0), 1)
        cv2.line(frame, (right_b, y), (right_b, min(y + dash, h)), (255, 255, 0), 1)

    # ---- Obstacle bounding boxes ----
    for obs in obstacles:
        x1, y1, x2, y2 = obs["bbox"]
        color = DANGER_COLORS.get(obs["danger_level"], (255, 255, 255))
        thickness = 3 if obs["danger_level"] == "HIGH" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text = f"{obs['label']} | {obs['region']} | {obs['distance']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # ---- Top bar: FPS + obstacle count ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    fps_color = (0, 255, 0) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    cv2.putText(frame, f"Obstacles: {len(obstacles)}", (200, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ---- Bottom bar: Navigation instruction ----
    if instruction:
        inst_type = instruction["type"]
        message = instruction["message"]
        color = INSTRUCTION_COLORS.get(inst_type, (255, 255, 255))

        # Background bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 55), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        # Instruction type badge
        cv2.putText(frame, f"[{inst_type}]", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Instruction message
        cv2.putText(frame, message, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def main():
    print("=" * 50)
    print("  Step 8: Navigation Decision Engine Test")
    print("=" * 50)
    print()

    # ---- Initialize all modules ----
    cam = Camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cam.start():
        return

    detector = ObjectDetector(YOLO_MODEL, YOLO_CONFIDENCE)

    obs_detector = ObstacleDetector(
        obstacle_classes=OBSTACLE_CLASSES,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        close_threshold=CLOSE_THRESHOLD,
        medium_threshold=MEDIUM_THRESHOLD,
        safety_stop_threshold=SAFETY_STOP_THRESHOLD
    )

    navigator = Navigator(cooldown=SPEECH_COOLDOWN)

    left_b, right_b = obs_detector.get_region_boundaries()

    # State
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()
    last_detections = []
    last_obstacles = []
    last_instruction = None
    counter = 0

    print(f"[Config] Cooldown: {SPEECH_COOLDOWN}s between same instructions")
    print("\nPress 'q' to quit\n")
    print("-" * 60)
    print(f"{'Time':<10} {'Type':<12} {'Message'}")
    print("-" * 60)

    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

        counter += 1

        # Run detection pipeline on every Nth frame
        if counter >= PROCESS_EVERY_N_FRAMES:
            counter = 0

            # Stage 1: YOLO detection
            last_detections = detector.detect(frame)

            # Stage 2: Obstacle filtering + region + distance
            last_obstacles = obs_detector.process(last_detections)

            # Stage 3: Navigation decision
            instruction = navigator.decide(last_obstacles)

            if instruction:
                last_instruction = instruction
                # Print to console
                t = time.strftime("%H:%M:%S")
                itype = instruction["type"]
                msg = instruction["message"]
                print(f"{t:<10} {itype:<12} {msg}")

        # Draw everything
        display = draw_full_hud(
            frame.copy(), fps, last_obstacles,
            last_instruction, left_b, right_b
        )

        # FPS
        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 0.5:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now

        cv2.imshow("Blind Assistant - Navigator Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("\n" + "-" * 60)
    print("Navigation test complete!")


if __name__ == "__main__":
    main()
