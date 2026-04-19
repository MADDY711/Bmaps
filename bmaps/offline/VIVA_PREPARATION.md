# Viva Preparation - AI-Based Blind Assistant

## Q1: Why is GPS not used for indoor navigation?

GPS (Global Positioning System) relies on signals from satellites orbiting Earth. Indoors, these signals are:
- **Blocked** by walls, ceilings, and roofs
- **Reflected** off surfaces (multipath interference)
- **Accuracy drops to 5-15 meters** indoors (useless for obstacle avoidance)

Our system needs **centimeter-level precision** to detect a chair 1 meter away. GPS cannot provide this. Instead, we use **camera-based computer vision** which works perfectly indoors with no satellite dependency.

---

## Q2: Difference between Object Detection and Obstacle Detection

| Aspect | Object Detection | Obstacle Detection |
|--------|-----------------|-------------------|
| **What it does** | Identifies ALL objects in an image | Filters only DANGEROUS objects |
| **Output** | Label + bounding box + confidence | Label + region + distance + danger level |
| **Example** | Detects: person, chair, book, clock | Only keeps: person, chair (ignores book, clock) |
| **Module** | `detector.py` (YOLOv8) | `obstacle.py` (custom filtering) |
| **Purpose** | "What's in the image?" | "What's blocking the user's path?" |

**Object detection** is general-purpose. **Obstacle detection** is task-specific — it adds spatial reasoning (left/center/right) and distance estimation (close/medium/far) to determine if an object is a threat.

---

## Q3: How does distance estimation work without a depth sensor?

We use **bounding box area ratio**:
```
area_ratio = (bbox_width × bbox_height) / (frame_width × frame_height)
```

- **Large bbox** (>15% of frame) → object is **CLOSE** (it appears big because it's near)
- **Medium bbox** (5-15%) → **MEDIUM** distance
- **Small bbox** (<5%) → **FAR** away

This works because of **perspective projection**: closer objects appear larger in the camera image. It's an approximation but sufficient for indoor navigation at walking speed.

---

## Q4: How does the navigation decision engine work?

Priority-based rules applied to each obstacle:

1. **STOP** (highest priority): Obstacle in CENTER region AND CLOSE distance
2. **CAUTION**: Obstacle in CENTER AND MEDIUM distance
3. **MOVE RIGHT**: Obstacle in LEFT region
4. **MOVE LEFT**: Obstacle in RIGHT region
5. **CLEAR** (lowest priority): No obstacles detected

**Anti-spam**: A cooldown timer (2 seconds) prevents repeating the same instruction. STOP has a shorter cooldown (1 second) because safety is critical.

---

## Q5: What are the screen regions?

The camera frame (640×480) is divided into 3 vertical columns:
```
|  LEFT (0-33%)  |  CENTER (33-66%)  |  RIGHT (66-100%)  |
|    0-211px      |    211-422px      |    422-640px       |
```

The obstacle's region is determined by the **center x-coordinate** of its bounding box.

---

## Q6: Why YOLOv8 nano and not a larger model?

| Model | Size | Speed (CPU) | Accuracy |
|-------|------|-------------|----------|
| YOLOv8n (nano) | 6.2 MB | ~0.275s/frame | Good enough |
| YOLOv8s (small) | 22 MB | ~0.5s/frame | Better |
| YOLOv8m (medium) | 52 MB | ~1.2s/frame | Best |

We chose **nano** because:
- Real-time requirement: must respond **< 1 second**
- Running on **CPU** (no GPU required)
- Accuracy is sufficient for detecting large indoor objects (chairs, people)
- Smaller model = less memory, faster startup

---

## Q7: Why is TTS non-blocking?

If text-to-speech ran on the main thread, the camera would **freeze** for 1-2 seconds every time an instruction is spoken. During that freeze:
- New obstacles would go undetected
- The system becomes blind and dangerous

Solution: TTS runs in a **background thread** with a message queue. The camera loop never pauses.

---

## Q8: Limitations of the system

1. **No true depth sensing** — distance estimation from bbox size is approximate
2. **Cannot detect transparent objects** (glass doors, windows)
3. **Cannot detect stairs reliably** — COCO dataset doesn't include "stairs" class
4. **Requires good lighting** — camera-based, fails in darkness
5. **Single camera = no stereo depth** — cannot measure exact distances
6. **Internet needed for ElevenLabs** — falls back to robotic pyttsx3 offline
7. **Cannot detect floor hazards** (wet floor, cables, small objects)
8. **Limited to COCO's 80 classes** — cannot detect unknown objects

---

## Q9: Future improvements

1. **Depth camera** (Intel RealSense / iPhone LiDAR) for true distance measurement
2. **Custom YOLO training** to detect stairs, doors, walls, floor obstacles
3. **SLAM** (Simultaneous Localization and Mapping) for building indoor maps
4. **Edge AI** deployment on Raspberry Pi / Jetson Nano for portability
5. **Haptic feedback** (vibration wristband) in addition to voice
6. **Multi-language TTS** support
7. **Floor segmentation** using semantic segmentation (DeepLab, SAM)
8. **Path planning** algorithms (A*, RRT) for longer navigation routes
9. **AR glasses integration** for hands-free operation
10. **Transfer to mobile** (Android/iOS app using TFLite/CoreML)

---

## Q10: What happens during the demo test?

**Setup:**
1. Place 2-3 obstacles (chairs, table) in a room
2. Blindfold the user
3. Hold laptop/phone camera facing forward
4. Run `main.py` and say "start"

**Expected behavior:**
- "Path is clear. Move forward." → user walks
- "Caution. Chair ahead. Slow down." → user slows
- "Stop! Chair directly ahead." → user stops
- "Chair on your left. Move right." → user turns right
- User safely walks 1+ meter avoiding all obstacles

---

## Q11: Key technical terms for viva

- **YOLO**: You Only Look Once — single-pass object detection
- **Bounding Box**: Rectangle around detected object (x1, y1, x2, y2)
- **Confidence Score**: How sure the model is (0-100%)
- **COCO Dataset**: 80-class dataset YOLO is trained on
- **TTS**: Text-to-Speech (converting text to audio)
- **Non-blocking**: Operation that doesn't freeze the main program
- **Cooldown**: Minimum time gap between repeated instructions
- **Inference**: Running a trained model on new data
- **FPS**: Frames Per Second (higher = smoother)
- **BGR**: Blue-Green-Red color format used by OpenCV
