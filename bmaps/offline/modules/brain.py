"""
============================================
Module: Gemini Brain (Cognitive Core)
============================================
The "Brain" of the assistant. It receives:
  1. The latest camera frame (Visual)
  2. The Situation Report (Metadata from Limbs)
  
Uses the NEW Google GenAI SDK (matching server.py).
============================================
"""

from google import genai
from google.genai import types
import cv2
import json
import threading
import time
import base64

class GeminiBrain:
    """
    The LLM-based reasoning engine for the assistant.
    """

    def __init__(self, api_key, model_name="gemini-2.5-flash-native-audio-latest"):
        """
        Initialize the Gemini API using the NEW SDK.
        """
        print(f"[Brain] Initializing Gemini Brain ({model_name})...")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.is_processing = False
        self._lock = threading.Lock()

    def analyze_situation(self, frame, situation_report, user_query=None):
        """
        Sends the frame and metadata to Gemini for analysis.
        """
        if self.is_processing:
            return None
            
        thread = threading.Thread(
            target=self._run_inference,
            args=(frame.copy(), situation_report, user_query),
            daemon=True
        )
        thread.start()
        return True

    def _run_inference(self, frame, report, query):
        """Internal thread for Gemini API call."""
        with self._lock:
            self.is_processing = True
            
        try:
            # 1. Prepare Image (Base64 Encode for New SDK)
            small_frame = cv2.resize(frame, (640, 480))
            success, encoded_image = cv2.imencode('.jpg', small_frame)
            if not success:
                raise ValueError("Could not encode image")
            
            image_bytes = encoded_image.tobytes()

            # 2. Build Human-Centric Prompt
            report_str = json.dumps(report, indent=2)
            
            prompt = f"""
            SYSTEM: You are the warm, supportive 'Brain' of a visual assistant. 
            You are talking to a visually impaired person.
            
            OFFLINE SENSOR DATA: {report_str}
            
            TASK: 
            Provide a warm, humanized description of the scene. 
            Don't just repeat the sensor data. Talk like a friend.
            
            EXAMPLE:
            Instead of "Person detected left", say "There's someone walking on your left, probably just passing by."
            Instead of "Path is clear", say "Everything looks good ahead, you've got a clear path."
            
            Keep it to 1-2 natural sentences. Speak directly to 'you'.
            """
            
            if query:
                prompt += f"\nUSER ASKED: {query}"
            
            # 3. Generate content using the new SDK format
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ]
            )
            
            text = response.text.strip()
            
            self.last_response = text
            print(f"[Brain] Humanized Output: {text}")

        except Exception as e:
            print(f"[Brain] Error during inference: {e}")
            self.last_response = None
        
        finally:
            with self._lock:
                self.is_processing = False

    def get_latest_response(self):
        """Retrieve and clear the latest response."""
        with self._lock:
            res = getattr(self, 'last_response', None)
            self.last_response = None
            return res
