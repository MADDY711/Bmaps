"""
============================================
Steps 9 & 10: Voice Output Test
============================================
Tests both ElevenLabs (if API key set) and pyttsx3 fallback.

How to run:
    C:\\ba_venv\\Scripts\\python.exe step9_10_test_voice.py

Expected output:
    - You should HEAR the system speak test messages
    - Console shows which TTS engine is being used
    - If no ElevenLabs key, falls back to pyttsx3 automatically

Debugging tips:
    - No sound? Check system volume and audio output device
    - pyttsx3 error? Try: pip install pyttsx3 --force-reinstall
    - ElevenLabs error? Check API key in .env file
============================================
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.voice_output import VoiceOutput
from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    PYTTSX3_RATE, PYTTSX3_VOLUME
)


def main():
    print("=" * 50)
    print("  Steps 9 & 10: Voice Output Test")
    print("=" * 50)
    print()

    # Initialize voice output
    voice = VoiceOutput(
        elevenlabs_api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        pyttsx3_rate=PYTTSX3_RATE,
        pyttsx3_volume=PYTTSX3_VOLUME,
        use_elevenlabs=bool(ELEVENLABS_API_KEY)
    )

    print(f"\n[Status] TTS Engine: {voice.get_status()}")

    # Start the background speech worker
    voice.start()
    time.sleep(0.5)

    # Test messages (simulating navigation instructions)
    test_messages = [
        "Blind assistant activated. System ready.",
        "Path is clear. Move forward.",
        "Caution. Chair ahead. Slow down.",
        "Stop! Person directly ahead.",
        "Person on your left. Move right.",
    ]

    print("\n[Test] Speaking test messages...\n")

    for i, msg in enumerate(test_messages, 1):
        print(f"  [{i}/{len(test_messages)}] {msg}")
        voice.speak(msg)
        time.sleep(3)  # Wait between messages

    print("\n[Done] Voice test complete!")

    # Cleanup
    voice.stop()


if __name__ == "__main__":
    main()
