# main.py — REST API for Android client
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from google import genai as genai_live
from google.genai import types
from PIL import Image
import io, os, base64, json, asyncio
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# ✅ Both APIs use the same key
GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Walking Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
vision_model = genai.GenerativeModel('gemini-2.0-flash')
live_client = genai_live.Client(api_key=GEMINI_API_KEY)

COMPANION_PROMPT = """You are a helpful, friendly AI companion for a visually impaired person.
- Describe surroundings clearly and concisely
- Alert about obstacles and dangers immediately
- Use directional terms: on your left, straight ahead, behind you
- Mention distances: about 2 feet away, 5 steps ahead
- Keep responses brief (2-3 sentences max unless asked for more)
- Prioritize safety information first
- Stay calm and reassuring even in urgent situations
"""

# ──────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "online", "message": "Walking Assistant API is running", "version": "2.0.0"}


# ──────────────────────────────────────────
# SCENE ANALYSIS — Android sends a JPEG frame
# ──────────────────────────────────────────
@app.post("/api/analyze-scene")
async def analyze_scene(
    image: UploadFile = File(...),
    mode: str = "general",
    lat: float = 0.0,
    lng: float = 0.0
):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        if mode == "obstacles":
            prompt = f"{COMPANION_PROMPT}\nIdentify obstacles, hazards, and the clearest path forward. Be specific about positions and distances."
        elif mode == "detailed":
            prompt = f"{COMPANION_PROMPT}\nDescribe the environment type, major objects, lighting, visible text, and people."
        else:
            prompt = f"{COMPANION_PROMPT}\nBriefly describe what you see. Focus on what's most important for navigation."

        response = vision_model.generate_content([prompt, img])
        return {
            "success": True,
            "description": response.text,
            "mode": mode,
            "location": {"lat": lat, "lng": lng}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────
# OBSTACLE DETECTION — returns structured JSON
# ──────────────────────────────────────────
@app.post("/api/detect-obstacles")
async def detect_obstacles(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        prompt = """You are a safety assistant for a visually impaired person who is WALKING RIGHT NOW.
Analyze this image and respond in this EXACT JSON format (no markdown, no backticks):
{
    "has_danger": true,
    "urgency": "high",
    "obstacles": [
        {
            "object": "name",
            "position": "left/center/right",
            "distance": "very close/close/moderate/far",
            "warning": "brief warning"
        }
    ],
    "safe_path": "brief navigation instruction"
}"""

        response = vision_model.generate_content([prompt, img])

        try:
            # Strip markdown fences if present
            text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
        except:
            result = {"has_danger": False, "urgency": "low", "obstacles": [], "safe_path": response.text}

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────
# TEXT READING — reads signs, labels, documents
# ──────────────────────────────────────────
@app.post("/api/read-text")
async def read_text(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

        prompt = """Extract and read ALL visible text in this image.
1. State if there IS text or NO text visible
2. Read it clearly and in order
3. Mention the type (sign, label, document, etc.)
4. Flag important info (warning signs, instructions)"""

        response = vision_model.generate_content([prompt, img])
        return {
            "success": True,
            "text_found": "no text visible" not in response.text.lower(),
            "content": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────
# WEBSOCKET — Live streaming (same as server.py)
# Android connects here for real-time mode
# ──────────────────────────────────────────
from maps import handle_maps_call

SYSTEM_PROMPT = """You are a real-time navigation assistant for a visually impaired user.
Describe what you see concisely. Warn about obstacles, curbs, traffic lights, and people.
When asked about nearby places, use your get_nearby_landmarks tool.
Keep responses short and spoken — no lists, no markdown."""

maps_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_nearby_landmarks",
        description="Fetches nearby points of interest from Google Maps given a lat/lng",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "latitude":      types.Schema(type=types.Type.NUMBER),
                "longitude":     types.Schema(type=types.Type.NUMBER),
                "radius_meters": types.Schema(type=types.Type.INTEGER),
                "place_type":    types.Schema(type=types.Type.STRING),
            },
            required=["latitude", "longitude"]
        )
    )
])

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Android WebSocket connected")

    try:
        async with live_client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-latest",
            config=types.LiveConnectConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[maps_tool],
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                    )
                )
            )
        ) as session:
            print("✅ Gemini Live session opened")

            await session.send_client_content(
                turns=[types.Content(
                    parts=[types.Part(text="Greet the user. Say: Hi! I am your walking assistant. I can see through your camera and help you navigate. Just speak naturally.")],
                    role="user"
                )],
                turn_complete=True
            )

            async def send_loop():
                audio_sent_this_turn = False
                while True:
                    try:
                        data = json.loads(await websocket.receive_text())

                        if "audio" in data:
                            await session.send_realtime_input(
                                audio=types.Blob(data=base64.b64decode(data["audio"]), mime_type="audio/pcm;rate=16000")
                            )
                            audio_sent_this_turn = True

                        if "frame" in data:
                            await session.send_realtime_input(
                                video=types.Blob(data=base64.b64decode(data["frame"]), mime_type="image/jpeg")
                            )

                        if data.get("turn_complete") and audio_sent_this_turn:
                            await session.send_client_content(turns=[], turn_complete=True)
                            audio_sent_this_turn = False

                    except Exception as e:
                        print(f"❌ Send error: {e}")
                        break

            async def receive_loop():
                while True:
                    try:
                        async for response in session.receive():
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    result = await handle_maps_call(fc)
                                    await session.send_tool_response(
                                        function_responses=[types.FunctionResponse(name=fc.name, id=fc.id, response=result)]
                                    )
                            if response.data:
                                await websocket.send_text(json.dumps({"audio": base64.b64encode(response.data).decode()}))
                            if response.server_content and response.server_content.turn_complete:
                                await websocket.send_text(json.dumps({"turn_complete": True}))
                    except Exception as e:
                        print(f"❌ Receive error: {e}")
                        break

            await asyncio.gather(send_loop(), receive_loop())

    except Exception as e:
        print(f"❌ Gemini session error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)