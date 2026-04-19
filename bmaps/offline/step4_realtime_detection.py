"""
============================================
Step 4: Real-Time Detection with Bounding Boxes
============================================
Combines the Camera and Detector modules into
a live video loop with real-time object detection
and bounding box visualization.

How to run:
    C:\\ba_venv\\Scripts\\python.exe step4_realtime_detection.py

Expected output:
    - Live camera feed with bounding boxes drawn
      around detected objects
    - Labels and confidence scores shown above boxes
    - FPS counter in top-left corner
    - Detection count in top-right area

Controls:
    - Press 'q' to quit
    - Press 's' to save a screenshot
    - Press '+' to increase confidence threshold
    - Press '-' to decrease confidence threshold

Debugging tips:
    - Low FPS (<10): Increase PROCESS_EVERY_N_FRAMES in config.py
    - No detections: Lower YOLO_CONFIDENCE in config.py
    - Lag/stuttering: Reduce FRAME_WIDTH/HEIGHT in config.py
============================================
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.camera import Camera
from modules.detector import ObjectDetector
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONFIDENCE, PROCESS_EVERY_N_FRAMES
)


# ============================================
# Color palette for different object classes
# ============================================
# BGR format for OpenCV
COLORS = {
    "person": (0, 255, 0),        # Green
    "chair": (0, 0, 255),         # Red
    "dining table": (255, 0, 0),  # Blue
    "couch": (0, 165, 255),       # Orange
    "bed": (255, 0, 255),         # Magenta
    "bottle": (255, 255, 0),      # Cyan
    "tv": (128, 0, 128),          # Purple
    "laptop": (0, 128, 255),      # Light orange
    "cell phone": (255, 128, 0),  # Light blue
    "dog": (0, 200, 200),         # Yellow-green
    "cat": (200, 200, 0),         # Teal
}
DEFAULT_COLOR = (0, 255, 0)       # Green fallback


def get_color(label):
    """Get a color for a given object label."""
    return COLORS.get(label, DEFAULT_COLOR)


def draw_detections(frame, detections):
    """
    Draw bounding boxes, labels, and confidence on the frame.

    Args:
        frame: BGR image (numpy array)
        detections: List of detection dicts
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        color = get_color(label)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        text = f"{label} {conf:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw filled background for label
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color, -1
        )

        # Draw label text (black on colored background)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    return frame


def draw_hud(frame, fps, detection_count, confidence_threshold):
    """
    Draw heads-up display with FPS and detection info.

    Args:
        frame: BGR image
        fps: Current frames per second
        detection_count: Number of objects detected
        confidence_threshold: Current confidence threshold
    """
    h, w = frame.shape[:2]

    # Semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS (top-left)
    fps_color = (0, 255, 0) if fps >= 15 else (0, 255, 255) if fps >= 8 else (0, 0, 255)
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2
    )

    # Detection count (center)
    cv2.putText(
        frame, f"Objects: {detection_count}", (w // 2 - 60, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # Confidence threshold (top-right)
    cv2.putText(
        frame, f"Conf: {confidence_threshold:.0%}", (w - 150, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    return frame


def main():
    """Run real-time object detection loop."""
    print("=" * 50)
    print("  Step 4: Real-Time Detection")
    print("=" * 50)
    print()

    # ---- 1. Initialize modules ----
    cam = Camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cam.start():
        return

    detector = ObjectDetector(YOLO_MODEL, YOLO_CONFIDENCE)

    # ---- 2. State variables ----
    fps = 0.0
    frame_count = 0
    last_fps_time = time.time()
    last_detections = []          # Cache detections for skipped frames
    process_frame_counter = 0     # Track which frames to process
    confidence = YOLO_CONFIDENCE

    print(f"\n[Config] Processing every {PROCESS_EVERY_N_FRAMES} frame(s)")
    print(f"[Config] Confidence threshold: {confidence:.0%}")
    print("\n[Controls]")
    print("  'q' = quit")
    print("  's' = screenshot")
    print("  '+' = increase confidence")
    print("  '-' = decrease confidence")
    print()

    # ---- 3. Main loop ----
    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

        process_frame_counter += 1

        # Run YOLO only on every Nth frame (performance optimization)
        if process_frame_counter >= PROCESS_EVERY_N_FRAMES:
            process_frame_counter = 0
            detector.confidence = confidence
            last_detections = detector.detect(frame)

        # Draw detections (use cached results on skipped frames)
        display_frame = draw_detections(frame.copy(), last_detections)

        # Calculate FPS
        frame_count += 1
        now = time.time()
        elapsed = now - last_fps_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            last_fps_time = now

        # Draw HUD overlay
        display_frame = draw_hud(
            display_frame, fps, len(last_detections), confidence
        )

        # Show frame
        cv2.imshow("Blind Assistant - Real-Time Detection", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, display_frame)
            print(f"[Screenshot] {fname}")
        elif key == ord('+') or key == ord('='):
            confidence = min(0.95, confidence + 0.05)
            print(f"[Confidence] Increased to {confidence:.0%}")
        elif key == ord('-'):
            confidence = max(0.1, confidence - 0.05)
            print(f"[Confidence] Decreased to {confidence:.0%}")

    # ---- 4. Cleanup ----
    cam.stop()
    cv2.destroyAllWindows()
    print("\nReal-time detection stopped.")


if __name__ == "__main__":
    main()
