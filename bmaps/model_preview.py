# model_preview.py
# A high-fidelity diagnostic tool that mirrors the server.py integration.
# Shows exactly what metadata (JSON) is being sent to the Gemini LLM.

import cv2
import sys
import os
import time
import json
import numpy as np
from dotenv import load_dotenv

# Add offline modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "offline"))

from offline.modules.detector import ObjectDetector
from offline.modules.obstacle import ObstacleDetector
from offline.modules.depth_estimator import DepthEstimator
from offline.modules.navigator import Navigator
from offline.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL, YOLO_CONFIDENCE, MDE_MODEL,
    OBSTACLE_CLASSES, CLOSE_THRESHOLD, MEDIUM_THRESHOLD,
    SAFETY_STOP_THRESHOLD, HAZARD_PROCESS_N_FRAMES,
    SPEECH_COOLDOWN
)

# Load environment variables
load_dotenv()

# Colors
COLOR_OBSTACLE = (0, 255, 0)
COLOR_HAZARD = (0, 0, 255)
COLOR_JSON = (200, 200, 200)
COLOR_INSTRUCTION = (255, 255, 0)

class HighFidelityPreviewer:
    def __init__(self):
        print("[Init] Loading models to match server.py...")
        
        # 1. Initialize identical model chain
        self.detector = ObjectDetector(YOLO_MODEL, YOLO_CONFIDENCE)
        self.depth_estimator = DepthEstimator(MDE_MODEL)
        self.obstacle_detector = ObstacleDetector(
            obstacle_classes=OBSTACLE_CLASSES,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            close_threshold=CLOSE_THRESHOLD,
            medium_threshold=MEDIUM_THRESHOLD,
            safety_stop_threshold=SAFETY_STOP_THRESHOLD
        )
        self.navigator = Navigator(cooldown=SPEECH_COOLDOWN)
        
        # 2. Camera setup
        source = os.getenv("CAMERA_SOURCE", CAMERA_INDEX)
        try:
            if str(source).isdigit(): source = int(source)
        except: pass
        
        self.cap = cv2.VideoCapture(source)
        print(f"[Init] Source: {source}")

    def process_frame(self, frame):
        """EXACT logic from server.py:process_reflexes"""
        detections = self.detector.detect(frame)
        obstacles = self.obstacle_detector.process(detections)
        depth_map = self.depth_estimator.estimate(frame)
        hazards = self.depth_estimator.detect_hazards(depth_map)
        
        instruction = None
        if hazards:
            for h in hazards:
                if h["danger_level"] == "HIGH":
                    instruction = {"type": "STOP", "message": f"Stop! {h['label']} detected {h['region'].lower()}."}
                    break
        
        if not instruction:
            instruction = self.navigator.decide(obstacles)

        report = {
            "obstacles": [o["label"] for o in obstacles],
            "hazards": [{"type": h["label"], "region": h["region"]} for h in hazards],
            "instruction": instruction["type"] if instruction else "NONE"
        }
        return obstacles, hazards, depth_map, report, instruction

    def run(self):
        cv2.namedWindow("Brain-Limb High Fidelity Preview", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # Run the server pipeline
            obstacles, hazards, depth_map, report, instruction = self.process_frame(frame)

            # Create a large canvas to show both Video and JSON
            # 640px (video) + 400px (json panel)
            canvas = np.zeros((600, 1040, 3), dtype=np.uint8)
            
            # 1. Main Video Feed (Top Left)
            display = cv2.resize(frame, (640, 480))
            
            # Draw Bboxes on display
            for obs in obstacles:
                x1, y1, x2, y2 = obs["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_OBSTACLE, 2)
                cv2.putText(display, f"{obs['label']} ({obs['distance']})", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OBSTACLE, 1)

            for haz in hazards:
                x1, y1, x2, y2 = haz["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_HAZARD, 3)
                cv2.putText(display, f"HAZARD: {haz['label'].upper()}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAZARD, 2)

            canvas[60:540, 20:660] = display

            # 2. JSON Report Panel (Right Side)
            panel_x = 680
            cv2.putText(canvas, "SITUATION_REPORT (To LLM)", (panel_x, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INSTRUCTION, 2)
            
            report_str = json.dumps(report, indent=2)
            y0, dy = 80, 25
            for i, line in enumerate(report_str.split('\n')):
                y = y0 + i*dy
                cv2.putText(canvas, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_JSON, 1)

            # 3. Decision Instruction (Bottom)
            inst_text = f"DECISION: {instruction['type'] if instruction else 'CLEAR'}"
            cv2.putText(canvas, inst_text, (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INSTRUCTION, 2)

            # 4. Depth Heatmap (Small, Top Right of Video)
            if depth_map is not None:
                depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                depth_thumb = cv2.resize(depth_color, (120, 90))
                canvas[60:150, 540:660] = depth_thumb
                cv2.rectangle(canvas, (540, 60), (660, 150), (255,255,255), 1)

            cv2.imshow("Brain-Limb High Fidelity Preview", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    HighFidelityPreviewer().run()
