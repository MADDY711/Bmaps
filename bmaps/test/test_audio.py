import pyaudio
p = pyaudio.PyAudio()
print("\n--- AUDIO INPUT DEVICES ---")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"  Index {i}: {info['name']}")
p.terminate()
# ```

# Run it and share the output — you'll see something like:
# ```
# Index 0: Microsoft Sound Mapper
# Index 1: Microphone (Realtek)
# Index 2: Headset Microphone