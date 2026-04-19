"""
============================================
Step 3: Test YOLOv8 Object Detection
============================================
Captures a single frame from the camera, runs
YOLOv8 detection, and prints all detected objects.

How to run:
    C:\\ba_venv\\Scripts\\python.exe step3_test_detector.py

Expected output:
    - List of detected objects with labels,
      confidence scores, and bounding boxes
    - A saved image "step3_detection_result.jpg"
      with bounding boxes drawn on it

Debugging tips:
    - If no objects detected: point camera at a person,
      chair, bottle, or other common objects
    - If slow: model is loading for the first time
      (subsequent runs will be faster)
    - If out of memory: you're using a GPU model on CPU,
      but yolov8n should be fine on CPU
============================================
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.camera import Camera
from modules.detector import ObjectDetector
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, YOLO_MODEL, YOLO_CONFIDENCE


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame: BGR image (numpy array)
        detections: List of detection dicts from ObjectDetector
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        # Draw bounding box (green rectangle)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence above the box
        text = f"{label} {conf:.2f}"
        # Background rectangle for text readability
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 10),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1  # Filled rectangle
        )
        cv2.putText(
            frame, text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

    return frame


def main():
    """Run single-frame detection test."""
    print("=" * 50)
    print("  Step 3: YOLOv8 Object Detection Test")
    print("=" * 50)
    print()

    # ---- 1. Initialize camera ----
    cam = Camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cam.start():
        return

    # ---- 2. Initialize detector ----
    detector = ObjectDetector(
        model_path=YOLO_MODEL,
        confidence=YOLO_CONFIDENCE
    )

    # ---- 3. Capture a frame ----
    print("\n[Test] Capturing frame...")
    # Read a few frames to let camera warm up (auto-exposure)
    for _ in range(10):
        frame = cam.read_frame()

    if frame is None:
        print("Failed to capture frame.")
        cam.stop()
        return

    # ---- 4. Run detection ----
    print("[Test] Running YOLOv8 detection...")
    start_time = time.time()
    detections = detector.detect(frame)
    elapsed = time.time() - start_time

    # ---- 5. Print results ----
    print(f"\n[Results] Detection took {elapsed:.3f} seconds")
    print(f"[Results] Found {len(detections)} object(s):\n")

    if len(detections) == 0:
        print("  No objects detected.")
        print("  Tip: Point camera at a person, chair, bottle, etc.")
    else:
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['label']}")
            print(f"     Confidence: {det['confidence']:.2%}")
            print(f"     Bounding box: {det['bbox']}")
            print(f"     Class ID: {det['class_id']}")
            print()

    # ---- 6. Save annotated image ----
    annotated = draw_detections(frame.copy(), detections)
    output_file = "step3_detection_result.jpg"
    cv2.imwrite(output_file, annotated)
    print(f"[Saved] Annotated image: {output_file}")

    # ---- 7. Display result (press any key to close) ----
    cv2.imshow("Step 3 - Detection Result", annotated)
    print("\nPress any key on the image window to close...")
    cv2.waitKey(0)

    # ---- 8. Cleanup ----
    cam.stop()
    cv2.destroyAllWindows()
    print("\nDetection test complete!")


if __name__ == "__main__":
    main()
