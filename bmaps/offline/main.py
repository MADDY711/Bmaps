"""
============================================
AI-Based Blind Assistant
Main Application (Steps 12 & 13)
============================================
Complete real-time system that:
  1. Captures camera frames
  2. Detects objects with YOLOv8
  3. Filters obstacles and estimates distance
  4. Makes navigation decisions
  5. Speaks instructions via TTS
  6. Accepts voice commands (start/stop)

How to run:
    C:\\ba_venv\\Scripts\\python.exe main.py

Controls:
    Voice: "start navigation" / "stop navigation"
    Keyboard:
        'q'     = quit
        's'     = start/resume navigation
        'p'     = pause navigation
        SPACE   = toggle navigation on/off
        '+'     = increase confidence
        '-'     = decrease confidence

Performance Optimizations (Step 12):
    - YOLO runs every Nth frame (PROCESS_EVERY_N_FRAMES)
    - TTS runs in background thread (non-blocking)
    - Speech queue flushes old messages (only latest)
    - Cooldown prevents instruction spam
    - Frame resolution tuned for speed (640x480)
============================================
"""

import cv2
import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.camera import Camera
from modules.detector import ObjectDetector
from modules.obstacle import ObstacleDetector
from modules.depth_estimator import DepthEstimator
from modules.brain import GeminiBrain
from modules.navigator import Navigator
from modules.voice_output import VoiceOutput
from modules.voice_command import VoiceCommandListener
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONFIDENCE, PROCESS_EVERY_N_FRAMES,
    OBSTACLE_CLASSES, CLOSE_THRESHOLD, MEDIUM_THRESHOLD,
    SAFETY_STOP_THRESHOLD, SPEECH_COOLDOWN,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    PYTTSX3_RATE, PYTTSX3_VOLUME,
    MDE_MODEL, HAZARD_PROCESS_N_FRAMES,
    GEMINI_API_KEY
)


# ============================================
# Visual Constants
# ============================================
DANGER_COLORS = {
    "HIGH": (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW": (0, 255, 0),
}

INSTRUCTION_COLORS = {
    "STOP": (0, 0, 255),
    "CAUTION": (0, 165, 255),
    "MOVE_RIGHT": (255, 200, 0),
    "MOVE_LEFT": (255, 200, 0),
    "CLEAR": (0, 255, 0),
}


class BlindAssistant:
    """
    Main application class that integrates all modules.

    Pipeline per frame:
        Camera → YOLO → Obstacle Filter → Navigator → Voice
    """

    def __init__(self):
        """Initialize all modules."""
        print("=" * 55)
        print("  AI-Based Blind Assistant")
        print("  Real-Time Obstacle Detection & Indoor Navigation")
        print("=" * 55)
        print()

        # ---- State ----
        self.is_navigating = False   # Navigation active?
        self.is_running = True       # App running?
        self.confidence = YOLO_CONFIDENCE

        # ---- Frame processing state (shared with detector thread) ----
        self.last_detections = []
        self.last_obstacles = []
        self.last_hazards = []
        self.last_depth_map = None
        self.last_instruction = None
        self.last_brain_response = None
        self.last_brain_trigger_time = 0
        self._detect_lock = threading.Lock()  # Protects shared detection results
        self._latest_frame = None             # Latest frame for detector thread
        self._frame_lock = threading.Lock()   # Protects latest_frame
        self._detector_thread = None

        # ---- FPS tracking ----
        self.fps = 0.0
        self.fps_frame_count = 0
        self.fps_last_time = time.time()

        # ---- Initialize modules ----
        print("[Init] Loading modules...\n")

        # Camera
        self.camera = Camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

        # Object Detector
        self.detector = ObjectDetector(YOLO_MODEL, YOLO_CONFIDENCE)

        # Depth Estimator
        self.depth_estimator = DepthEstimator(MDE_MODEL)

        # Gemini Brain (Cognitive Core)
        self.brain = GeminiBrain(GEMINI_API_KEY)

        # Obstacle Detector
        self.obstacle_detector = ObstacleDetector(
            obstacle_classes=OBSTACLE_CLASSES,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            close_threshold=CLOSE_THRESHOLD,
            medium_threshold=MEDIUM_THRESHOLD,
            safety_stop_threshold=SAFETY_STOP_THRESHOLD
        )
        self.left_boundary, self.right_boundary = \
            self.obstacle_detector.get_region_boundaries()

        # Navigator
        self.navigator = Navigator(cooldown=SPEECH_COOLDOWN)

        # Voice Output
        self.voice = VoiceOutput(
            elevenlabs_api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            pyttsx3_rate=PYTTSX3_RATE,
            pyttsx3_volume=PYTTSX3_VOLUME,
            use_elevenlabs=bool(ELEVENLABS_API_KEY)
        )

        # Voice Command Listener
        self.voice_listener = VoiceCommandListener(
            callback=self._on_voice_command
        )

        print(f"\n[Init] TTS Engine: {self.voice.get_status()}")
        print("[Init] All modules loaded!\n")

    def _on_voice_command(self, command):
        """
        Callback fired when a voice command is recognized.
        """
        if command == "start" and not self.is_navigating:
            print("\n>>> VOICE COMMAND: START NAVIGATION <<<")
            self.start_navigation()
        elif command == "stop" and self.is_navigating:
            print("\n>>> VOICE COMMAND: STOP NAVIGATION <<<")
            self.stop_navigation()
        else:
            # General Question to the Brain (e.g. "What's in front of me?")
            print(f"\n>>> BRAIN QUERY: {command} <<<")
            frame = self.camera.read_frame()
            if frame is not None:
                # Provide a quick situation report as context
                with self._detect_lock:
                    report = {
                        "obstacles": [o["label"] for o in self.last_obstacles],
                        "hazards": [{"type": h["label"], "region": h["region"]} for h in self.last_hazards],
                    }
                self.brain.analyze_situation(frame, report, user_query=command)

    def start_navigation(self):
        """Activate navigation mode."""
        if self.is_navigating:
            return
        self.is_navigating = True
        self.navigator.reset()
        self.voice.speak("Navigation started. I will guide you.")
        print("[Nav] Navigation STARTED")

    def stop_navigation(self):
        """Deactivate navigation mode."""
        if not self.is_navigating:
            return
        self.is_navigating = False
        self.last_instruction = None
        self.voice.speak("Navigation stopped.")
        print("[Nav] Navigation STOPPED")

    def _detection_worker(self):
        """
        Background thread that runs YOLO and Depth estimation.
        """
        frame_count = 0
        while self.is_running:
            if not self.camera.has_new_frame():
                time.sleep(0.05)
                continue

            frame = self.camera.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # 1. Run YOLO (Objects/Obstacles)
            self.detector.confidence = self.confidence
            detections = self.detector.detect(frame)
            obstacles = self.obstacle_detector.process(detections)

            # 2. Run Depth Hazard Detection (Potholes/Stairs)
            # This is heavy, so we run it every N frames
            hazards = []
            depth_map = None
            if frame_count % HAZARD_PROCESS_N_FRAMES == 0:
                depth_map = self.depth_estimator.estimate(frame)
                hazards = self.depth_estimator.detect_hazards(depth_map)

            frame_count += 1

            # 3. Decision Logic (Hazards > Obstacles)
            instruction = None
            if self.is_navigating:
                # Check for critical hazards first
                if hazards:
                    for h in hazards:
                        region_text = h.get("region", "ahead").lower()
                        if h["danger_level"] == "HIGH":
                            instruction = {
                                "type": "STOP",
                                "message": f"Danger! {h['label']} detected {region_text}. Stop now."
                            }
                            break
                        elif not instruction: # Moderate hazard
                            instruction = {
                                "type": "CAUTION",
                                "message": f"Caution, {h['label']} {region_text}."
                            }

                # If no critical hazard, check regular obstacles
                if not instruction:
                    instruction = self.navigator.decide(obstacles)

            # 4. Situation Report (Context for Gemini Brain)
            report = {
                "obstacles": [o["label"] for o in obstacles],
                "hazards": [{"type": h["label"], "region": h["region"]} for h in hazards],
                "instruction": instruction["type"] if instruction else "NONE"
            }

            # 5. Cognitive Trigger (Brain)
            # Trigger Gemini if there's a HIGH danger hazard and we haven't asked recently
            if self.is_navigating and any(h["danger_level"] == "HIGH" for h in hazards):
                now = time.time()
                if now - self.last_brain_trigger_time > 12: # Cooldown for Gemini Brain
                    self.brain.analyze_situation(frame, report)
                    self.last_brain_trigger_time = now

            # 6. Atomic Update
            with self._detect_lock:
                self.last_detections = detections
                self.last_obstacles = obstacles
                if hazards: self.last_hazards = hazards
                if depth_map is not None: self.last_depth_map = depth_map
                if instruction: self.last_instruction = instruction

            # 7. Speak instruction (Limb Reflexes - Fast)
            if instruction:
                self.voice.speak(instruction["message"])
                t = time.strftime("%H:%M:%S")
                print(f"  [{t}] {instruction['type']}: {instruction['message']}")

    def run(self):
        """Main application loop."""
        # Start camera
        if not self.camera.start():
            print("FATAL: Cannot start camera. Exiting.")
            return

        # Start voice systems
        self.voice.start()
        self.voice_listener.start()

        # Start the YOLO detection background thread
        self._detector_thread = threading.Thread(
            target=self._detection_worker,
            daemon=True,
            name="YOLODetectorWorker"
        )
        self._detector_thread.start()
        print("[Perf] YOLO detection running in background thread")

        # Startup announcement
        self.voice.speak("Blind assistant ready. Say start to begin navigation.")

        print("\n" + "=" * 55)
        print("  SYSTEM RUNNING")
        print("  Voice: say 'start' or 'stop'")
        print("  Keys:  's'=start  'p'=pause  'q'=quit  SPACE=toggle")
        print("=" * 55 + "\n")

        try:
            while self.is_running:
                # read_frame() returns instantly (threaded camera)
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Read the latest detection results (non-blocking)
                with self._detect_lock:
                    obstacles = self.last_obstacles
                    hazards = self.last_hazards
                    depth_map = self.last_depth_map
                    instruction = self.last_instruction

                # ---- Draw display (on the main thread, never blocks) ----
                display = self._draw_display(frame.copy(), obstacles, instruction,
                                            hazards=hazards, depth_map=depth_map)

                # ---- Update FPS ----
                self._update_fps()

                # ---- Brain Response (Cognitive Output) ----
                brain_res = self.brain.get_latest_response()
                if brain_res:
                    self.voice.speak(brain_res)

                # ---- Show window ----
                cv2.imshow("AI Blind Assistant", display)

                # ---- Handle keyboard (waitKey 33ms = ~30 FPS cap) ----
                if not self._handle_keys():
                    break

        except KeyboardInterrupt:
            print("\n[Interrupt] Ctrl+C detected")

        finally:
            self._shutdown()

    def _draw_display(self, frame, obstacles, instruction, hazards=[], depth_map=None):
        """Draw the complete UI overlay on the frame."""
        display = frame
        h, w = display.shape[:2]

        # ---- Region divider lines ----
        dash = 15
        for y in range(45, h - 60, dash * 2):
            cv2.line(display, (self.left_boundary, y),
                     (self.left_boundary, min(y + dash, h)),
                     (255, 255, 0), 1)
            cv2.line(display, (self.right_boundary, y),
                     (self.right_boundary, min(y + dash, h)),
                     (255, 255, 0), 1)

        # ---- Obstacle bounding boxes ----
        for obs in obstacles:
            x1, y1, x2, y2 = obs["bbox"]
            color = DANGER_COLORS.get(obs["danger_level"], (200, 200, 200))
            thick = 3 if obs["danger_level"] == "HIGH" else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thick)

            # Label
            text = f"{obs['label']} | {obs['region']} | {obs['distance']}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # ---- Hazard bounding boxes (Potholes/Stairs) ----
        for haz in hazards:
            x1, y1, x2, y2 = haz["bbox"]
            color = (255, 0, 255) if haz["label"] == "pothole" else (0, 255, 255)
            thick = 3 if haz["danger_level"] == "HIGH" else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thick)
            cv2.putText(display, f"HAZARD: {haz['label']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- Top bar (solid black for speed, no alpha blend) ----
        cv2.rectangle(display, (0, 0), (w, 42), (0, 0, 0), -1)

        # ---- Depth Map Thumbnail (Top-Right, Above top bar overlay) ----
        if depth_map is not None:
            # Colorize for better visibility
            depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            thumb_w = 120
            thumb_h = int(h * (thumb_w / w))
            depth_thumb = cv2.resize(depth_color, (thumb_w, thumb_h))
            # Place it slightly below the top bar for a cleaner look
            y_off = 45
            display[y_off:y_off+thumb_h, w-thumb_w-10:w-10] = depth_thumb
            cv2.rectangle(display, (w-thumb_w-10, y_off), (w-10, y_off+thumb_h), (255, 255, 255), 1)
            cv2.putText(display, "DEPTH", (w-thumb_w-10, y_off - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display FPS + Camera hardware FPS
        cam_fps = self.camera.get_camera_fps()
        fps_c = (0, 255, 0) if self.fps >= 15 else (0, 165, 255) if self.fps >= 8 else (0, 0, 255)
        cv2.putText(display, f"FPS: {self.fps:.0f} (cam:{cam_fps:.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_c, 2)

        # Navigation status
        if self.is_navigating:
            cv2.putText(display, "NAV: ON", (180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(display, "NAV: OFF", (180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # Obstacle count
        cv2.putText(display, f"Obs: {len(obstacles)}", (360, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # TTS engine
        cv2.putText(display, self.voice.get_status(), (480, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ---- Bottom bar: Navigation instruction ----
        if instruction and self.is_navigating:
            inst = instruction
            color = INSTRUCTION_COLORS.get(inst["type"], (255, 255, 255))

            cv2.rectangle(display, (0, h - 55), (w, h), (0, 0, 0), -1)

            cv2.putText(display, f"[{inst['type']}]", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, inst["message"], (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        elif not self.is_navigating:
            cv2.rectangle(display, (0, h - 35), (w, h), (0, 0, 0), -1)
            cv2.putText(display, "Press 's' or say 'start' to begin navigation",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150, 150, 150), 1)

        return display

    def _update_fps(self):
        """Calculate smoothed FPS."""
        self.fps_frame_count += 1
        now = time.time()
        elapsed = now - self.fps_last_time
        if elapsed >= 0.5:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_last_time = now

    def _handle_keys(self):
        """
        Handle keyboard input.

        Returns:
            bool: False if should quit, True otherwise
        """
        key = cv2.waitKey(33) & 0xFF  # ~30 FPS display cap

        if key == ord('q'):
            return False
        elif key == ord('s'):
            self.start_navigation()
        elif key == ord('p'):
            self.stop_navigation()
        elif key == ord(' '):  # Space = toggle
            if self.is_navigating:
                self.stop_navigation()
            else:
                self.start_navigation()
        elif key == ord('+') or key == ord('='):
            self.confidence = min(0.95, self.confidence + 0.05)
            print(f"[Config] Confidence: {self.confidence:.0%}")
        elif key == ord('-'):
            self.confidence = max(0.1, self.confidence - 0.05)
            print(f"[Config] Confidence: {self.confidence:.0%}")

        return True

    def _shutdown(self):
        """Clean shutdown of all modules."""
        print("\n[Shutdown] Stopping all modules...")
        self.is_running = False
        self.is_navigating = False

        self.voice.speak("System shutting down. Goodbye.")
        time.sleep(1)

        self.voice_listener.stop()
        self.voice.stop()
        self.camera.stop()
        cv2.destroyAllWindows()
        print("[Shutdown] Complete. Goodbye!")


def main():
    """Entry point."""
    app = BlindAssistant()
    app.run()


if __name__ == "__main__":
    main()
