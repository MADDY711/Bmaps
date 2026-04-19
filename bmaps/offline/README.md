# AI-Based Blind Assistant with Real-Time Obstacle Detection and Indoor Navigation

A real-time system that uses a camera + YOLOv8 to detect obstacles and give voice instructions, enabling a blindfolded user to safely navigate indoors.

## Project Structure

```
project/
├── main.py                        # Complete application entry point
├── config.py                      # Central configuration (all settings)
├── requirements.txt               # Python dependencies
├── .env.example                   # API keys template
├── .env                           # Your API keys (create from .env.example)
├── yolov8n.pt                     # YOLOv8 nano model (auto-downloaded)
├── modules/
│   ├── __init__.py
│   ├── camera.py                  # Camera access module
│   ├── detector.py                # YOLOv8 object detection
│   ├── obstacle.py                # Obstacle filtering & distance estimation
│   ├── navigator.py               # Navigation decision engine
│   ├── voice_output.py            # TTS (ElevenLabs + pyttsx3 fallback)
│   └── voice_command.py           # Voice command recognition
├── step1_verify_setup.py          # Setup verification
├── step2_test_camera.py           # Camera test
├── step3_test_detector.py         # YOLOv8 test
├── step4_realtime_detection.py    # Real-time detection test
├── step5_6_7_test_obstacles.py    # Obstacle detection test
├── step8_test_navigator.py        # Navigation engine test
└── step9_10_test_voice.py         # Voice output test
```

## Quick Start

### 1. Virtual environment (created at short path to avoid Windows long-path issues)

```bash
python -m venv C:\ba_venv
```

### 2. Install dependencies

```bash
C:\ba_venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu --timeout 120
C:\ba_venv\Scripts\pip.exe install --timeout 120 opencv-python ultralytics pyttsx3 elevenlabs SpeechRecognition PyAudio openai python-dotenv
```

### 3. Set up API keys (optional — system works offline with pyttsx3)

```bash
copy .env.example .env
# Edit .env and add your ElevenLabs API key
```

### 4. Verify setup

```bash
C:\ba_venv\Scripts\python.exe step1_verify_setup.py
```

### 5. Run the full system

```bash
C:\ba_venv\Scripts\python.exe main.py
```

## Controls

| Input                               | Action                      |
| ----------------------------------- | --------------------------- |
| Voice: "start" / "start navigation" | Begin navigation            |
| Voice: "stop" / "stop navigation"   | Stop navigation             |
| Key: `s`                            | Start navigation            |
| Key: `p`                            | Pause navigation            |
| Key: `SPACE`                        | Toggle navigation on/off    |
| Key: `+` / `-`                      | Adjust detection confidence |
| Key: `q`                            | Quit application            |

## System Pipeline

```
Camera Frame → YOLOv8 Detection → Obstacle Filtering → Region + Distance
    → Navigation Decision → Voice Instruction (non-blocking TTS)
```

## Tech Stack

| Component         | Technology                                            |
| ----------------- | ----------------------------------------------------- |
| Camera Input      | OpenCV (640x480, real-time)                           |
| Object Detection  | YOLOv8 nano (Ultralytics)                             |
| Obstacle Analysis | Custom region division + bbox distance estimation     |
| Navigation Logic  | Priority-based decision engine with cooldown          |
| Voice Output      | ElevenLabs API (primary) + pyttsx3 (offline fallback) |
| Voice Commands    | SpeechRecognition + PyAudio + Google API              |

## Performance

- **Detection speed**: ~0.275s per frame on CPU (YOLOv8 nano)
- **Frame skipping**: YOLO runs every 2nd frame, cached results reused
- **Speech**: Non-blocking background thread with queue flushing
- **Cooldown**: 2s between repeated instructions (1s for STOP)

## Why No GPS?

GPS is **not accurate indoors** (error of 5-15 meters). Satellite signals are blocked/reflected by walls and ceilings. This system uses **camera-based computer vision** for precise, real-time indoor navigation with no GPS dependency.
