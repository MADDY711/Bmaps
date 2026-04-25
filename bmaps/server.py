# server.py
import sys
import os
import asyncio
import base64
import json
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Add channel_ii modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "channel_ii"))

from modules.detector import ObjectDetector
from modules.obstacle import ObstacleDetector
from modules.navigator import Navigator
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE,
    OBSTACLE_CLASSES, CLOSE_THRESHOLD, MEDIUM_THRESHOLD,
    SAFETY_STOP_THRESHOLD, SPEECH_COOLDOWN,
    FRAME_WIDTH, FRAME_HEIGHT
)

from maps import handle_maps_call
from map_server import app, map_state
from context import SYSTEM_PROMPT

load_dotenv()

# ---- Gemini Config ----
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
    ),
    types.FunctionDeclaration(
        name="start_navigation",
        description="Starts turn-by-turn walking navigation to a specific destination.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "destination": types.Schema(type=types.Type.STRING, description="The name or address of the place to go to."),
            },
            required=["destination"]
        )
    )
])

# Porting OutdoorNavigator logic to server state
class NavigationState:
    def __init__(self):
        self.is_active = False
        self.destination = None
        self.dest_lat = None
        self.dest_lng = None
        self.route = None
        self.current_step_index = 0
        self.last_step_announced = -1

    def reset(self):
        self.__init__()

nav_state = NavigationState()

async def handle_start_navigation(destination):
    """Bridge for start_navigation tool."""
    from maps import maps_service
    # 1. Geocode
    target = await maps_service.geocode_address(destination)
    if not target:
        return {"error": f"Could not find destination: {destination}"}
    
    # 2. Get directions from current location
    route = await maps_service.get_directions(
        map_state["lat"], map_state["lng"],
        target["lat"], target["lng"]
    )
    if not route:
        return {"error": f"No walking route found to {destination}"}
    
    # 3. Update global navigation state
    nav_state.is_active = True
    nav_state.destination = destination
    nav_state.dest_lat = target["lat"]
    nav_state.dest_lng = target["lng"]
    nav_state.route = route
    nav_state.current_step_index = 0
    nav_state.last_step_announced = -1
    
    # 4. Notify map dashboard
    import httpx
    async with httpx.AsyncClient() as client:
        await client.post("http://localhost:8001/start_navigation", json={
            "destination": destination,
            "lat": target["lat"],
            "lng": target["lng"],
            "steps": route["steps"]
        })
    
    return {
        "status": "Navigation started",
        "total_distance": route["total_distance"],
        "total_duration": route["total_duration"],
        "first_step": route["steps"][0]["instruction"]
    }

# ---- Offline Model Initialization ----
print("[Init] Loading Reflex Models (from channel_ii)...")
detector = ObjectDetector(YOLO_MODEL, YOLO_CONFIDENCE)
obstacle_detector = ObstacleDetector(
    obstacle_classes=OBSTACLE_CLASSES,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    close_threshold=CLOSE_THRESHOLD,
    medium_threshold=MEDIUM_THRESHOLD,
    safety_stop_threshold=SAFETY_STOP_THRESHOLD
)
navigator = Navigator(cooldown=SPEECH_COOLDOWN)
executor = ThreadPoolExecutor(max_workers=2)

import concurrent.futures

# Performance Optimization: Run ML on smaller frames
ML_RES = (320, 240)

def process_reflexes(frame_bytes):
    """Optimized ML Pipeline (Channel II)"""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return None, None

    # Downscale for ML to save 4x processing power
    ml_frame = cv2.resize(frame, ML_RES)

    # Run YOLO (Depth-Anything-V2 removed in Channel II for speed)
    detections = detector.detect(ml_frame)

    # Process obstacles
    obstacles = obstacle_detector.process(detections)
    
    # Generate instruction
    instruction = navigator.decide(obstacles)

    report = {
        "obstacles": [o["label"] for o in obstacles],
        "hazards": [], # Depth hazards removed in this version
        "instruction": instruction["type"] if instruction else "NONE"
    }
    return report, instruction

def calculate_distance(lat1, lon1, lat2, lat2_lng):
    # Simple distance in meters
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # Radius of earth in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lat2_lng - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket accepted")
    map_state["status"] = "Connected"

    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-latest",
            config=types.LiveConnectConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[maps_tool],
                response_modalities=["AUDIO"],
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
            map_state["status"] = "Ready"

            # Initial Greeting
            await session.send_client_content(
                turns=[types.Content(parts=[types.Part(text="Greet the user warmly. Say you are ready to help them walk safely.")], role="user")],
                turn_complete=True
            )

            async def send_loop():
                audio_sent_this_turn = False
                last_report_time = 0
                last_proactive_time = time.time()
                last_instruction_type = "NONE"
                
                while True:
                    try:
                        data = json.loads(await websocket.receive_text())

                        if "lat" in data and "lng" in data:
                            map_state["lat"], map_state["lng"] = data["lat"], data["lng"]
                            map_state["status"] = "Walking..."
                            
                            # Update Navigation Step Tracking
                            if nav_state.is_active and nav_state.route:
                                steps = nav_state.route["steps"]
                                curr_step = steps[nav_state.current_step_index]
                                dist_to_end = calculate_distance(
                                    map_state["lat"], map_state["lng"],
                                    curr_step["end_lat"], curr_step["end_lng"]
                                )
                                if dist_to_end < 20:
                                    if nav_state.current_step_index < len(steps) - 1:
                                        nav_state.current_step_index += 1
                                        print(f"📍 Navigation: Moved to step {nav_state.current_step_index + 1}")
                                    else:
                                        print("🏁 Navigation: Arrived")
                                        nav_state.is_active = False

                        if "frame" in data:
                            frame_bytes = base64.b64decode(data["frame"])
                            
                            report, instruction = await asyncio.get_event_loop().run_in_executor(
                                executor, process_reflexes, frame_bytes
                            )

                            if report:
                                if instruction and instruction["type"] == "STOP":
                                    print(f"🚨 REFLEX ALERT: {instruction['message']}")
                                    map_state["status"] = f"HAZARD: {instruction['type']}"
                                    await websocket.send_text(json.dumps({
                                        "transcript": f"REFLEX: {instruction['message']}"
                                    }))

                                now = time.time()
                                current_inst = report.get("instruction", "NONE")
                                
                                # Proactive trigger if safety changed, time passed, or navigation step changed
                                significant_change = (current_inst != last_instruction_type)
                                time_for_update = (now - last_proactive_time > 15.0)
                                nav_step_change = (nav_state.is_active and nav_state.current_step_index != nav_state.last_step_announced)

                                if now - last_report_time > 2.0:
                                    should_trigger = significant_change or time_for_update or nav_step_change
                                    
                                    # Include GPS and Navigation info in report
                                    report["gps"] = {"lat": map_state["lat"], "lng": map_state["lng"]}
                                    if nav_state.is_active:
                                        curr_step = nav_state.route["steps"][nav_state.current_step_index]
                                        report["navigation"] = {
                                            "is_active": True,
                                            "destination": nav_state.destination,
                                            "instruction": curr_step["instruction"],
                                            "distance_to_turn": curr_step["distance"],
                                            "step_number": nav_state.current_step_index + 1,
                                            "total_steps": len(nav_state.route["steps"])
                                        }
                                    
                                    await session.send_client_content(
                                        turns=[types.Content(
                                            parts=[types.Part(text=f"SITUATION_REPORT: {json.dumps(report)}")],
                                            role="user"
                                        )],
                                        turn_complete=should_trigger
                                    )
                                    
                                    last_report_time = now
                                    if should_trigger:
                                        last_proactive_time = now
                                        last_instruction_type = current_inst
                                        if nav_state.is_active:
                                            nav_state.last_step_announced = nav_state.current_step_index
                                        print(f"🧠 Proactive Trigger: change={significant_change}, time={time_for_update}, nav={nav_step_change}")

                            await session.send_realtime_input(
                                video=types.Blob(data=frame_bytes, mime_type="image/jpeg")
                            )

                        if data.get("speech_start"):
                            print("🎙️ ActivityStart")
                            await session.send_realtime_input(activity_start=types.ActivityStart())

                        if "audio" in data:
                            await session.send_realtime_input(
                                audio=types.Blob(data=base64.b64decode(data["audio"]), mime_type="audio/pcm;rate=16000")
                            )
                            audio_sent_this_turn = True

                        if (data.get("speech_end") or data.get("turn_complete")) and audio_sent_this_turn:
                            print("🔚 ActivityEnd")
                            await session.send_realtime_input(activity_end=types.ActivityEnd())
                            audio_sent_this_turn = False

                    except Exception as e:
                        print(f"❌ Send error: {e}")
                        break

            async def receive_loop():
                while True:
                    try:
                        async for response in session.receive():
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if hasattr(part, 'text') and part.text:
                                        print(f"💬 Gemini: {part.text}")
                                        await websocket.send_text(json.dumps({"transcript": part.text}))

                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    print(f"🗺️ Maps Tool: {fc.name}")
                                    
                                    if fc.name == "start_navigation":
                                        result = await handle_start_navigation(fc.args["destination"])
                                    else:
                                        # Developer Debug: Push the exact query to the map preview
                                        map_state["last_query"] = fc.args
                                        map_state["status"] = f"Searching: {fc.args.get('place_type', 'nearby')}"
                                        result = await handle_maps_call(fc)
                                        map_state["places"] = result.get("places", [])
                                        map_state["status"] = f"Found {len(map_state['places'])} places"
                                    
                                    await session.send_tool_response(
                                        function_responses=[types.FunctionResponse(name=fc.name, id=fc.id, response=result)]
                                    )

                            if response.data:
                                await websocket.send_text(json.dumps({"audio": base64.b64encode(response.data).decode()}))

                            if response.server_content and response.server_content.turn_complete:
                                print("✅ Turn complete")
                                await websocket.send_text(json.dumps({"turn_complete": True}))

                    except Exception as e:
                        print(f"❌ Receive error: {e}")
                        break

            await asyncio.gather(send_loop(), receive_loop())

    except Exception as e:
        print(f"❌ Gemini session error: {e}")
        map_state["status"] = "Error"
        await websocket.close()