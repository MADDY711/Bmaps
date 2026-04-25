"""
============================================
Module: Depth Estimator (Depth-Anything-V2)
============================================
Uses the official 2024 SOTA Depth-Anything-V2-Small
from Hugging Face (ViT-Small backbone).
============================================
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    """
    Monocular Depth Estimator using Depth-Anything-V2-Small.
    """

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Load the Depth-Anything-V2-Small model. Prioritizes local cache.
        """
        import os
        self.device = torch.device("cpu")
        hf_token = os.getenv("HF_TOKEN")
        
        # Dedicated cache folder in the project directory
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "depth_anything")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # 1. Try loading from local cache ONLY (Instant, no internet)
            print(f"[Depth] Attempting to load from local cache: {cache_dir}")
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, cache_dir=cache_dir, local_files_only=True
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id, cache_dir=cache_dir, local_files_only=True
            )
            print("[Depth] Success: Model loaded from local disk.")
        except Exception:
            # 2. Fallback: Download from HF (Only happens if local files missing)
            print(f"[Depth] Local files not found. Downloading from Hugging Face: {model_id}")
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    model_id, token=hf_token, cache_dir=cache_dir, local_files_only=False
                )
                self.model = AutoModelForDepthEstimation.from_pretrained(
                    model_id, token=hf_token, cache_dir=cache_dir, local_files_only=False
                )
                print("[Depth] Success: Model downloaded and cached locally.")
            except Exception as e:
                print(f"[Depth] CRITICAL ERROR: Could not download model: {e}")
                self.model = None
                return

        if self.model:
            self.model.to(self.device)
            self.model.eval()
            print("[Depth] Depth-Anything-V2 ready!")

    def estimate(self, frame):
        """
        Run depth estimation on a single frame.
        """
        if self.model is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pre-process
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Interpolate to original frame size
            prediction = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255 (invert: closer is brighter)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
        
        return depth_map.astype(np.uint8)

    def detect_hazards(self, depth_map):
        """
        Analyze the depth map for potholes and stairs using 
        Differential Surface Analysis (DSA).
        """
        h, w = depth_map.shape
        
        # Focus on the 'walking path' (Bottom 40% of frame)
        roi_h = int(h * 0.4)
        roi_w = int(w * 0.6)
        start_y = h - roi_h - 15 
        start_x = (w - roi_w) // 2
        
        roi = depth_map[start_y:start_y + roi_h, start_x:start_x + roi_w]
        hazards = []

        # 1. DSA POTHOLE DETECTION
        # Analyze in horizontal strips
        strip_height = 12
        for y in range(0, roi_h - strip_height, strip_height):
            strip = roi[y:y+strip_height, :]
            
            row_median = np.median(strip)
            row_min = np.min(strip)
            
            # A pothole is a localized "dip" 
            # (V2 provides a wider range, so we use a higher threshold)
            if (row_median - row_min) > 40: 
                min_idx = np.argmin(np.mean(strip, axis=0))
                
                # Neighborhood validation
                win = 25
                left_side = np.median(strip[:, max(0, min_idx-win):min_idx]) if min_idx > 0 else 0
                right_side = np.median(strip[:, min_idx:min_idx+win]) if min_idx < roi_w else 0
                
                # Validation: Pothole must be surrounded by closer ground
                if left_side > row_min + 15 and right_side > row_min + 15:
                    center_x = start_x + min_idx
                    if center_x < w * 0.33:
                        region = "LEFT"
                    elif center_x > w * 0.66:
                        region = "RIGHT"
                    else:
                        region = "CENTER"

                    hazards.append({
                        "label": "pothole",
                        "region": region,
                        "confidence": 0.95,
                        "bbox": (start_x + min_idx - 40, start_y + y, 
                                 start_x + min_idx + 40, start_y + y + strip_height),
                        "danger_level": "HIGH" if (row_median - row_min) > 60 else "MEDIUM"
                    })
                    break

        # 2. STAIRS/STEP-DOWN DETECTION (Robust Edge Density Analysis)
        if not hazards:
            # Use a slightly wider ROI for stairs
            stair_roi_w = int(w * 0.8)
            stair_start_x = (w - stair_roi_w) // 2
            stair_roi = depth_map[start_y:start_y + roi_h, stair_start_x:stair_start_x + stair_roi_w]
            
            # Canny with lower thresholds to catch faint step edges
            edges = cv2.Canny(stair_roi, 20, 80)
            
            # Use Probabilistic Hough Lines with more lenient angle and gap settings
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                    minLineLength=stair_roi_w // 3, maxLineGap=20)
            
            if lines is not None:
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    # Allow up to 15 degrees of tilt (common while walking)
                    if angle < 15 or angle > 165:
                        horizontal_lines += 1
                
                # If we see 3 or more parallel-ish lines, it's likely stairs
                if horizontal_lines >= 3:
                    hazards.append({
                        "label": "stairs",
                        "region": "CENTER",
                        "confidence": 0.85,
                        "bbox": (stair_start_x, start_y, stair_start_x + stair_roi_w, start_y + roi_h),
                        "danger_level": "MEDIUM"
                    })

        return hazards
