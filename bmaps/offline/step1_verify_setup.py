"""
============================================
Step 1: Environment Setup Verification
============================================
Run this script after installing dependencies to verify
that everything is correctly set up.

How to run:
    python step1_verify_setup.py

Expected output:
    All checks should show [OK]. Any [FAIL] means
    that dependency needs to be reinstalled.
"""

import sys

def check_python_version():
    """Verify Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [FAIL] Python {version.major}.{version.minor}.{version.micro} "
              f"(need 3.8+)")
        return False


def check_import(module_name, display_name=None):
    """Try importing a module and report success/failure."""
    name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "version unknown")
        print(f"  [OK] {name} ({version})")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name} - {e}")
        return False


def check_camera():
    """Verify that a camera is accessible."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [OK] Camera accessible (resolution: {w}x{h})")
                return True
            else:
                print("  [FAIL] Camera opened but could not read a frame")
                return False
        else:
            print("  [FAIL] Cannot open camera (index 0)")
            print("         Tip: Make sure no other app is using the camera")
            return False
    except Exception as e:
        print(f"  [FAIL] Camera check error: {e}")
        return False


def check_yolo_model():
    """Verify that YOLOv8 nano model can be loaded."""
    try:
        from ultralytics import YOLO
        # This will auto-download yolov8n.pt if not present (~6MB)
        model = YOLO("yolov8n.pt")
        num_classes = len(model.names)
        print(f"  [OK] YOLOv8 nano model loaded ({num_classes} classes)")
        return True
    except Exception as e:
        print(f"  [FAIL] YOLOv8 model load error: {e}")
        return False


def check_tts():
    """Verify that pyttsx3 (offline TTS) works."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        print(f"  [OK] pyttsx3 initialized ({len(voices)} voice(s) available)")
        engine.stop()
        return True
    except Exception as e:
        print(f"  [FAIL] pyttsx3 error: {e}")
        return False


def check_dotenv():
    """Verify python-dotenv is installed."""
    try:
        import dotenv
        version = getattr(dotenv, "__version__", "installed")
        print(f"  [OK] python-dotenv ({version})")
        return True
    except ImportError:
        print("  [FAIL] python-dotenv not installed")
        return False


def main():
    """Run all setup verification checks."""
    print("=" * 50)
    print("  AI Blind Assistant - Setup Verification")
    print("=" * 50)

    results = []

    # 1. Python version
    print("\n[1/7] Python Version:")
    results.append(check_python_version())

    # 2. Core libraries
    print("\n[2/7] Core Libraries:")
    results.append(check_import("cv2", "OpenCV"))
    results.append(check_import("numpy", "NumPy"))

    # 3. YOLOv8
    print("\n[3/7] YOLOv8 (Object Detection):")
    results.append(check_import("ultralytics", "Ultralytics"))
    results.append(check_yolo_model())

    # 4. Text-to-Speech
    print("\n[4/7] Text-to-Speech:")
    results.append(check_tts())
    results.append(check_import("elevenlabs", "ElevenLabs SDK"))

    # 5. Voice Recognition
    print("\n[5/7] Voice Recognition:")
    results.append(check_import("speech_recognition", "SpeechRecognition"))
    # PyAudio check
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        pa.terminate()
        print(f"  [OK] PyAudio ({device_count} audio device(s))")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] PyAudio - {e}")
        print("         Tip (Windows): pip install pipwin && pipwin install pyaudio")
        print("         Tip (Linux): sudo apt install portaudio19-dev && pip install pyaudio")
        results.append(False)

    # 6. Utilities
    print("\n[6/7] Utilities:")
    results.append(check_dotenv())
    results.append(check_import("openai", "OpenAI SDK"))

    # 7. Camera
    print("\n[7/7] Camera Access:")
    results.append(check_camera())

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 50)
    print(f"  Results: {passed}/{total} checks passed")
    print("=" * 50)

    if passed == total:
        print("\n  ✅ All checks passed! You are ready for Step 2.")
    else:
        failed = total - passed
        print(f"\n  ⚠️  {failed} check(s) failed. Fix them before proceeding.")
        print("  Re-run this script after fixing: python step1_verify_setup.py")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
