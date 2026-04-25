# client.py
import cv2, pyaudio, asyncio, websockets, base64, json
import numpy as np
import threading
import queue
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

BACKEND_URL = "ws://localhost:8000/ws/stream"
FRAME_INTERVAL = 2.0  # 1 FPS

# ✅ Toggle this to switch between Push-to-Talk and VAD
USE_PUSH_TO_TALK = False  # False = use VAD (automatic speech detection)

def is_speech(audio_chunk: bytes, threshold=50) -> bool:
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    return rms > threshold


def find_working_mic(audio):
    preferred = [2, 1, 0]
    for idx in preferred:
        for rate in [16000, 44100, 48000]:
            try:
                stream = audio.open(
                    format=pyaudio.paInt16, channels=1, rate=rate,
                    input=True, frames_per_buffer=1024, input_device_index=idx
                )
                info = audio.get_device_info_by_index(idx)
                print(f"✅ Mic opened: [{idx}] {info['name']} @ {rate}Hz")
                return stream, rate
            except Exception:
                continue
    raise RuntimeError("❌ No working microphone found.")


class WalkingAssistantClient:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream, self.mic_rate = find_working_mic(self.audio)

        self.lat = 18.5204
        self.lng = 73.8567

        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

        self.greeting_done = asyncio.Event()

        self.conversation = []
        self.status = "Waiting for greeting..."
        self.is_speaking = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # ✅ Push-to-talk state — set by UI thread, read by send_loop
        self.ptt_active = False        # True while SPACE is held
        self.ptt_just_released = False # True for one cycle after release

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

    def add_message(self, role: str, text: str):
        time_str = datetime.now().strftime("%H:%M:%S")
        self.conversation.append({"role": role, "text": text, "time": time_str})
        if len(self.conversation) > 10:
            self.conversation.pop(0)

    def capture_audio_chunk(self):
        return self.stream.read(1024, exception_on_overflow=False)

    def ui_thread(self):
        # cap = cv2.VideoCapture(0)
        camera_source = os.getenv("CAMERA_SOURCE", "1")
        # convert to int if it's a digit (device index), keep as string if it's a URL
        camera_source = int(camera_source) if camera_source.isdigit() else camera_source
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if USE_PUSH_TO_TALK:
            print("🖥️  Monitor window opened. Hold SPACE to speak, release to send. Press Q to quit.")
        else:
            print("🖥️  Monitor window opened. Press Q to quit.")

        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()

            self._draw_ui()
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                self.running = False
                break

            # ✅ SPACE bar held = recording
            if USE_PUSH_TO_TALK:
                if key == 32:  # spacebar
                    if not self.ptt_active:
                        self.ptt_active = True
                        print("🔴 Recording...")
                else:
                    if self.ptt_active:
                        self.ptt_active = False
                        self.ptt_just_released = True
                        print("⏹️  Stopped recording, sending...")

        cap.release()
        cv2.destroyAllWindows()
        self.audio_queue.put(None)

    def _draw_ui(self):
        canvas = np.zeros((600, 1100, 3), dtype=np.uint8)

        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is not None:
            cam = cv2.resize(frame, (640, 480))
            canvas[60:540, 20:660] = cam
        else:
            cv2.rectangle(canvas, (20, 60), (660, 540), (40, 40, 40), -1)
            cv2.putText(canvas, "No Camera", (280, 310),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        cv2.putText(canvas, "LIVE CAMERA", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        if self.ptt_active:
            status_color = (0, 0, 255)
            status_text = "● RECORDING"
        elif self.is_speaking:
            status_color = (0, 255, 0)
            status_text = "● SPEAKING"
        elif not self.greeting_done.is_set():
            status_color = (0, 200, 255)
            status_text = "● GREETING"
        else:
            status_color = (100, 100, 255)
            status_text = "● LISTENING"

        cv2.putText(canvas, status_text, (420, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        if USE_PUSH_TO_TALK and self.greeting_done.is_set():
            hint = "[ HOLD SPACE TO SPEAK ]" if not self.ptt_active else "[ RELEASE SPACE TO SEND ]"
            hint_color = (0, 255, 255) if not self.ptt_active else (0, 0, 255)
            cv2.putText(canvas, hint, (100, 555),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, hint_color, 2)

        panel_x = 680
        cv2.putText(canvas, "CONVERSATION", (panel_x, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.line(canvas, (panel_x, 55), (1090, 55), (50, 50, 50), 1)

        y = 90
        for msg in self.conversation[-8:]:
            if msg["role"] == "user":
                color = (100, 255, 100)
                label = f"[{msg['time']}] YOU:"
            else:
                color = (255, 200, 50)
                label = f"[{msg['time']}] GEMINI:"

            cv2.putText(canvas, label, (panel_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 18

            words = msg["text"].split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > 42:
                    cv2.putText(canvas, line, (panel_x + 10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                    y += 16
                    line = word
                else:
                    line = line + " " + word if line else word
            if line:
                cv2.putText(canvas, line, (panel_x + 10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                y += 22

            if y > 560:
                break

        cv2.rectangle(canvas, (0, 570), (1100, 600), (30, 30, 30), -1)
        cv2.putText(canvas, f"Status: {self.status}   |   GPS: {self.lat}, {self.lng}",
                   (10, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Walking Assistant Monitor", canvas)

    async def connect_and_run(self):
        async with websockets.connect(
            BACKEND_URL,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=10
        ) as ws:
            self.status = "Connected — waiting for greeting..."
            self.greeting_done.clear()

            async def receive_loop():
                while self.running:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(response)

                        if data.get("audio"):
                            self.audio_queue.put(base64.b64decode(data["audio"]))
                            self.status = "Gemini speaking..."

                        if data.get("transcript"):
                            self.add_message("gemini", data["transcript"])

                        if data.get("turn_complete"):
                            if not self.greeting_done.is_set():
                                self.greeting_done.set()
                                self.status = "Ready — hold SPACE to speak!" if USE_PUSH_TO_TALK else "Ready — speak now!"
                                self.add_message("gemini", "Hi! I am your walking assistant.")
                            else:
                                self.status = "Ready — hold SPACE to speak!" if USE_PUSH_TO_TALK else "Ready — speak now!"

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"❌ Receive error: {e}")
                        self.status = "Connection lost. Reconnecting..."
                        break

            async def send_loop():
                last_frame_time = 0

                # VAD-only variables
                SILENCE_THRESHOLD = 10
                speech_silence_counter = 0
                was_speaking = False

                await self.greeting_done.wait()

                while self.running:
                    try:
                        now = asyncio.get_event_loop().time()
                        payload = {"lat": self.lat, "lng": self.lng}
                        raw_audio = self.capture_audio_chunk()

                        # ============================================================
                        # INPUT MODE: PUSH-TO-TALK
                        # ============================================================
                        if USE_PUSH_TO_TALK:
                            if self.ptt_active:
                                self.is_speaking = True
                                self.status = "Recording... (release SPACE to send)"
                                payload["audio"] = base64.b64encode(raw_audio).decode('utf-8')
                                self._audio_sent_this_ptt = True

                            elif self.ptt_just_released:
                                self.is_speaking = False
                                self.ptt_just_released = False
                                if getattr(self, "_audio_sent_this_ptt", False):
                                    self.status = "Processing your speech..."
                                    payload["speech_end"] = True   # ✅ FIX: triggers ActivityEnd on server
                                    self.add_message("user", "[voice input]")
                                    self._audio_sent_this_ptt = False
                                else:
                                    print("⚠️ Skipping — no audio was recorded")

                            else:
                                self.is_speaking = False

                        # ============================================================
                        # INPUT MODE: VAD (automatic)
                        # ============================================================
                        else:
                            if is_speech(raw_audio):
                                self.is_speaking = True
                                self.status = "You are speaking..."
                                payload["audio"] = base64.b64encode(raw_audio).decode('utf-8')
                                speech_silence_counter = 0

                                # ✅ FIX: signal start of a new utterance on the first speech chunk
                                if not was_speaking:
                                    payload["speech_start"] = True
                                    print("🎙️ Speech started → sending speech_start")

                                was_speaking = True
                            else:
                                self.is_speaking = False
                                if was_speaking:
                                    speech_silence_counter += 1

                            # ✅ FIX: send speech_end so server can call ActivityEnd on Gemini
                            if was_speaking and speech_silence_counter >= SILENCE_THRESHOLD:
                                self.status = "Waiting for response..."
                                payload["speech_end"] = True
                                print("🔇 Silence detected → sending speech_end")
                                self.add_message("user", "[voice input]")
                                speech_silence_counter = 0
                                was_speaking = False

                        # Frame sampling
                        if now - last_frame_time >= FRAME_INTERVAL:
                            with self.frame_lock:
                                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                            if frame is not None:
                                resized = cv2.resize(frame, (640, 480))
                                _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                payload["frame"] = base64.b64encode(buffer).decode('utf-8')
                            last_frame_time = now

                        await ws.send(json.dumps(payload))
                        await asyncio.sleep(0.2)

                    except Exception as e:
                        print(f"❌ Send error: {e}")
                        self.status = "Connection lost. Reconnecting..."
                        break

            await asyncio.gather(receive_loop(), send_loop())

    async def run(self):
        ui = threading.Thread(target=self.ui_thread, daemon=True)
        ui.start()

        while self.running:
            try:
                print("🔌 Connecting to server...")
                await self.connect_and_run()
            except Exception as e:
                print(f"❌ Connection failed: {e}")

            if self.running:
                self.status = "Reconnecting in 3 seconds..."
                print("🔄 Reconnecting in 3 seconds...")
                await asyncio.sleep(3)

        ui.join(timeout=2)


asyncio.run(WalkingAssistantClient().run())