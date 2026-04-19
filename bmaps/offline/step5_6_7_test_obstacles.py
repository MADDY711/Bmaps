"""
============================================
Steps 5, 6, 7: Obstacle Detection Test
============================================
Tests the obstacle module with live camera feed:
  - Step 5: Screen region lines (LEFT | CENTER | RIGHT)
  - Step 6: Only obstacle classes are highlighted
  - Step 7: Distance estimation shown per obstacle

How to run:
    C:\\ba_venv\\Scripts\\python.exe step5_6_7_test_obstacles.py

Expected output:
    - Live feed with 2 vertical lines dividing regions
    - Only obstacle objects get colored bounding boxes
    - Each box shows: label, region, distance, danger level
    - DANGER panel at bottom when HIGH danger detected

Controls:
    - Press 'q' to quit

Debugging tips:
    - No obstacles shown? Place a chair or person in view
    - All objects filtered? Check OBSTACLE_CLASSES in config.py
    - Distance always FAR? Move objects closer to camera
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
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONFIDENCE, PROCESS_EVERY_N_FRAMES,
    OBSTACLE_CLASSES, CLOSE_THRESHOLD, MEDIUM_THRESHOLD,
    SAFETY_STOP_THRESHOLD
)

# Colors for danger levels (BGR)
DANGER_COLORS = {
    "HIGH": (0, 0, 255),      # Red
    "MEDIUM": (0, 165, 255),   # Orange
    "LOW": (0, 255, 0),        # Green
}


def draw_regions(frame, left_boundary, right_boundary):
    """Draw vertical region divider lines on the frame."""
    h = frame.shape[0]

    # Draw dashed vertical lines for region boundaries
    dash_length = 15
    for y in range(0, h, dash_length * 2):
        cv2.line(frame, (left_boundary, y),
                 (left_boundary, min(y + dash_length, h)),
                 (255, 255, 0), 1)  # Cyan
        cv2.line(frame, (right_boundary, y),
                 (right_boundary, min(y + dash_length, h)),
                 (255, 255, 0), 1)

    # Region labels at the bottom
    cv2.putText(frame, "LEFT", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "CENTER",
                ((left_boundary + right_boundary) // 2 - 30, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "RIGHT", (right_boundary + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return frame


def draw_obstacles(frame, obstacles):
    """Draw obstacle bounding boxes with region/distance info."""
    for obs in obstacles:
        x1, y1, x2, y2 = obs["bbox"]
        color = DANGER_COLORS.get(obs["danger_level"], (255, 255, 255))

        # Thick border for HIGH danger
        thickness = 3 if obs["danger_level"] == "HIGH" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Info text: "chair | CENTER | CLOSE"
        info = f"{obs['label']} | {obs['region']} | {obs['distance']}"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        # Background for text
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, info, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Danger badge (bottom-right of bbox)
        danger_text = obs["danger_level"]
        cv2.putText(frame, danger_text, (x2 - 50, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Area ratio percentage
        pct = f"{obs['area_ratio']:.1%}"
        cv2.putText(frame, pct, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def draw_danger_panel(frame, obstacles):
    """Draw a danger alert panel if HIGH danger is detected."""
    high_danger = [o for o in obstacles if o["danger_level"] == "HIGH"]

    if high_danger:
        h, w = frame.shape[:2]
        # Red warning bar at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Warning text
        obj_names = ", ".join(set(o["label"] for o in high_danger))
        cv2.putText(frame, f"DANGER: {obj_names} ahead! STOP!",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    return frame


def draw_hud(frame, fps, total_detections, obstacle_count):
    """Draw info bar at top."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    fps_color = (0, 255, 0) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    cv2.putText(frame, f"YOLO: {total_detections}", (160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Obstacles: {obstacle_count}", (310, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def main():
    print("=" * 50)
    print("  Steps 5/6/7: Obstacle Detection Test")
    print("=" * 50)
    print()

    # Initialize modules
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

    left_b, right_b = obs_detector.get_region_boundaries()

    # State
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()
    last_detections = []
    last_obstacles = []
    counter = 0

    print(f"[Config] Obstacle classes: {OBSTACLE_CLASSES}")
    print(f"[Config] Regions: LEFT(0-{left_b}px) CENTER({left_b}-{right_b}px) RIGHT({right_b}-{FRAME_WIDTH}px)")
    print(f"[Config] Close threshold: {CLOSE_THRESHOLD:.0%} of frame area")
    print(f"[Config] Safety stop threshold: {SAFETY_STOP_THRESHOLD:.0%}")
    print("\nPress 'q' to quit\n")

    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

        counter += 1

        # Run detection on every Nth frame
        if counter >= PROCESS_EVERY_N_FRAMES:
            counter = 0
            last_detections = detector.detect(frame)
            last_obstacles = obs_detector.process(last_detections)

            # Print obstacles to console (limited to avoid spam)
            if last_obstacles:
                high = [o for o in last_obstacles if o["danger_level"] == "HIGH"]
                if high:
                    for o in high:
                        print(f"  ⚠️  {o['label']} | {o['region']} | "
                              f"{o['distance']} | area={o['area_ratio']:.1%} | STOP={o['should_stop']}")

        # Draw everything
        display = frame.copy()
        display = draw_regions(display, left_b, right_b)
        display = draw_obstacles(display, last_obstacles)
        display = draw_danger_panel(display, last_obstacles)

        # FPS
        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 0.5:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now

        display = draw_hud(display, fps, len(last_detections), len(last_obstacles))

        cv2.imshow("Blind Assistant - Obstacle Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("\nObstacle detection test complete!")


if __name__ == "__main__":
    main()
