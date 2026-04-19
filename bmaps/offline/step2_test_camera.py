"""
============================================
Step 2: Test Camera Module
============================================
Opens your webcam, displays a live feed with
an FPS counter, and exits when you press 'q'.

How to run:
    C:\\ba_venv\\Scripts\\python.exe step2_test_camera.py

Expected output:
    - A window titled "Blind Assistant - Camera Test"
    - Live webcam feed with green FPS counter (top-left)
    - Press 'q' to quit

Debugging tips:
    - If no window appears: check if another app uses the camera
    - If FPS is low (<15): reduce resolution in config.py
    - If frame is black: try camera_index=1 (external webcam)
============================================
"""

import cv2
import time
import sys
import os

# Add project root to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.camera import Camera
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


def main():
    """Run camera test with live FPS display."""
    print("=" * 50)
    print("  Step 2: Camera Module Test")
    print("=" * 50)
    print()

    # ---- 1. Create and start camera ----
    cam = Camera(
        camera_index=CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT
    )

    if not cam.start():
        print("Failed to start camera. Exiting.")
        return

    # ---- 2. FPS tracking variables ----
    fps = 0.0
    frame_count = 0
    fps_update_interval = 0.5   # Update FPS display every 0.5 seconds
    last_fps_time = time.time()

    print("\n[Controls]")
    print("  Press 'q' to quit")
    print("  Press 's' to save a screenshot")
    print()

    # ---- 3. Main loop: read and display frames ----
    while True:
        frame = cam.read_frame()

        if frame is None:
            print("No frame received. Retrying...")
            time.sleep(0.1)
            continue

        # Count frames for FPS calculation
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - last_fps_time

        # Update FPS every 0.5 seconds (smoother display)
        if elapsed >= fps_update_interval:
            fps = frame_count / elapsed
            frame_count = 0
            last_fps_time = current_time

        # ---- 4. Draw FPS on frame ----
        # cv2.putText(image, text, position, font, scale, color, thickness)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,               # Image to draw on
            fps_text,            # Text string
            (10, 30),            # Position (x, y) from top-left
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1.0,                 # Font scale
            (0, 255, 0),         # Color (BGR) - Green
            2                    # Thickness
        )

        # Draw frame dimensions info
        h, w = frame.shape[:2]
        info_text = f"Resolution: {w}x{h}"
        cv2.putText(
            frame, info_text, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
        )

        # ---- 5. Show the frame ----
        cv2.imshow("Blind Assistant - Camera Test", frame)

        # ---- 6. Handle key presses ----
        # waitKey(1) waits 1ms and returns the key pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[Quit] 'q' pressed. Exiting...")
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[Screenshot] Saved: {filename}")

    # ---- 7. Cleanup ----
    cam.stop()
    cv2.destroyAllWindows()
    print("\nCamera test complete!")


if __name__ == "__main__":
    main()
