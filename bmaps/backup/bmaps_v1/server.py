# server.py
from fastapi import FastAPI, WebSocket
from google import genai
from google.genai import types
import asyncio, base64, json, os
from dotenv import load_dotenv

from maps import handle_maps_call

load_dotenv()
app = FastAPI()
client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))

SYSTEM_PROMPT = """You are a real-time navigation assistant for a visually impaired user.
Describe what you see concisely. Warn about obstacles, curbs, traffic lights, and people.
When asked about nearby places, use your get_nearby_landmarks tool.
Keep responses short and spoken — no lists, no markdown.
IMPORTANT: Always respond with spoken audio only. Never respond with text only."""

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
    print("✅ WebSocket accepted")

    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-latest",
            config=types.LiveConnectConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[maps_tool],
                response_modalities=["AUDIO"],
                # ✅ VAD disabled — we control turn-taking manually via ActivityStart/ActivityEnd
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
                ),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
                    )
                )
            )
        ) as session:
            print("✅ Gemini Live session opened")

            await session.send_client_content(
                turns=[
                    types.Content(
                        parts=[types.Part(text="Greet the user warmly. Say: Hi! I am your walking assistant. I can see through your camera and help you navigate. Just speak naturally.")],
                        role="user"
                    )
                ],
                turn_complete=True
            )

            async def send_loop():
                audio_sent_this_turn = False
                while True:
                    try:
                        data = json.loads(await websocket.receive_text())

                        # ✅ FIX: send ActivityStart when user begins speaking
                        if data.get("speech_start"):
                            print("🎙️ ActivityStart → Gemini")
                            await session.send_realtime_input(
                                activity_start=types.ActivityStart()
                            )

                        if "audio" in data:
                            print(f"📤 Sending audio chunk to Gemini")
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=base64.b64decode(data["audio"]),
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                            audio_sent_this_turn = True

                        if "frame" in data:
                            await session.send_realtime_input(
                                video=types.Blob(
                                    data=base64.b64decode(data["frame"]),
                                    mime_type="image/jpeg"
                                )
                            )

                        # ✅ FIX: send ActivityEnd on speech_end (VAD) or turn_complete (PTT)
                        # This tells Gemini the user has finished speaking — triggers response generation
                        if (data.get("speech_end") or data.get("turn_complete")) and audio_sent_this_turn:
                            print("🔚 ActivityEnd → Gemini")
                            await session.send_realtime_input(
                                activity_end=types.ActivityEnd()
                            )
                            audio_sent_this_turn = False

                    except Exception as e:
                        print(f"❌ Send error: {e}")
                        break

            async def receive_loop():
                while True:
                    try:
                        async for response in session.receive():

                            has_data = response.data is not None
                            has_text = False
                            text_content = ""

                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if hasattr(part, 'text') and part.text:
                                        has_text = True
                                        text_content += part.text

                            print(f"📩 Gemini — audio:{has_data} text:{has_text} turn_complete:{response.server_content.turn_complete if response.server_content else False}")

                            if has_text:
                                print(f"💬 Gemini text: {text_content[:100]}")
                                await websocket.send_text(json.dumps({
                                    "transcript": text_content
                                }))

                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    print(f"🗺️  Maps call: {fc.name}")
                                    result = await handle_maps_call(fc)
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                name=fc.name,
                                                id=fc.id,
                                                response=result
                                            )
                                        ]
                                    )

                            if response.data:
                                await websocket.send_text(json.dumps({
                                    "audio": base64.b64encode(response.data).decode()
                                }))

                            if response.server_content and response.server_content.turn_complete:
                                print("✅ Gemini finished turn")
                                await websocket.send_text(json.dumps({
                                    "turn_complete": True
                                }))

                    except Exception as e:
                        print(f"❌ Receive error: {e}")
                        break

            await asyncio.gather(send_loop(), receive_loop())

    except Exception as e:
        print(f"❌ Failed to open Gemini session: {e}")
        await websocket.close()