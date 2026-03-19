# client.py
import cv2, pyaudio, asyncio, websockets, base64, json
import numpy as np
import threading
import queue

BACKEND_URL = "ws://localhost:8000/ws/stream"
FRAME_INTERVAL = 2.0  # 1 FPS

# --- UPDATED DIAGNOSTIC FUNCTION ---
def is_speech(audio_chunk: bytes, threshold=50) -> bool:
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    # Handle empty chunks to avoid errors
    if len(audio_array) == 0:
        return False
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    
    # ✅ Print actual mic level to terminal for calibration
    status = "🎤 SPEECH DETECTED" if rms > threshold else "..."
    print(f"🎙️ RMS level: {rms:5.1f} | Threshold: {threshold} | {status}", end='\r')
    
    return rms > threshold

class WalkingAssistantClient:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1,
            rate=16000, input=True, frames_per_buffer=1024
        )
        self.lat = 18.5204
        self.lng = 73.8567

        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

        self.greeting_done = asyncio.Event()

    def _playback_worker(self):
        out_stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                break
            out_stream.write(chunk)
        out_stream.close()

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buffer).decode('utf-8')

    def capture_audio_chunk(self):
        # Using exception_on_overflow=False to prevent crashes during processing lags
        return self.stream.read(1024, exception_on_overflow=False)

    async def run(self):
        async with websockets.connect(BACKEND_URL) as ws:
            print(f"Connected to {BACKEND_URL}. Waiting for greeting...")
            last_frame_time = 0

            async def receive_loop():
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)

                        if data.get("audio"):
                            self.audio_queue.put(base64.b64decode(data["audio"]))

                        if data.get("turn_complete"):
                            if not self.greeting_done.is_set():
                                self.greeting_done.set()
                                print("\n✅ Greeting done, now listening...")
                            else:
                                print("\n✅ Gemini finished responding, listening again...")

                    except Exception as e:
                        print(f"\n❌ Receive error: {e}")
                        break

            async def send_loop():
                nonlocal last_frame_time
                # Adjust SILENCE_THRESHOLD: higher number = longer wait after speaking
                SILENCE_THRESHOLD = 12 
                speech_silence_counter = 0
                was_speaking = False 

                await self.greeting_done.wait()
                print("\n🎙️ Ready — speak now!")

                while True:
                    try:
                        now = asyncio.get_event_loop().time()
                        payload = {"lat": self.lat, "lng": self.lng}

                        raw_audio = self.capture_audio_chunk()

                        # --- UPDATED SEND LOGIC ---
                        if is_speech(raw_audio, threshold=50):
                            payload["audio"] = base64.b64encode(raw_audio).decode('utf-8')
                            speech_silence_counter = 0
                            was_speaking = True 
                        else:
                            if was_speaking:
                                speech_silence_counter += 1

                        # Trigger end-of-turn
                        if was_speaking and speech_silence_counter >= SILENCE_THRESHOLD:
                            print("\n🔚 Silence detected — sending turn_complete...")
                            payload["turn_complete"] = True
                            speech_silence_counter = 0
                            was_speaking = False 

                        # Periodic frame capture
                        if now - last_frame_time >= FRAME_INTERVAL:
                            frame = self.capture_frame()
                            if frame:
                                payload["frame"] = frame
                            last_frame_time = now

                        await ws.send(json.dumps(payload))
                        await asyncio.sleep(0.1) # Slightly faster polling for audio

                    except Exception as e:
                        print(f"\n❌ Send error: {e}")
                        break

            await asyncio.gather(receive_loop(), send_loop())

if __name__ == "__main__":
    try:
        asyncio.run(WalkingAssistantClient().run())
    except KeyboardInterrupt:
        print("\nStopping client...")