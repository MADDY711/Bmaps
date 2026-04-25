# maps.py
import httpx, os, re
from dotenv import load_dotenv
load_dotenv()

class GoogleMapsService:
    """Robust Google Maps Service ported from Channel II."""
    BASE_PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    BASE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
    BASE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    BASE_PLACE_TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            print("❌ GOOGLE_MAPS_API_KEY not set")

    async def find_nearby_places(self, lat, lng, place_type=None, radius=500):
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "key": self.api_key
        }
        if place_type:
            params["type"] = place_type

        async with httpx.AsyncClient() as client:
            r = await client.get(self.BASE_PLACES_URL, params=params, timeout=5.0)
            data = r.json()
            if data.get("status") not in ("OK", "ZERO_RESULTS"):
                return {"error": data.get("status"), "message": data.get("error_message")}
            
            results = data.get("results", [])[:5]
            return [
                {"name": p['name'], "vicinity": p.get('vicinity', 'nearby'), "lat": p['geometry']['location']['lat'], "lng": p['geometry']['location']['lng']}
                for p in results
            ]

    async def get_directions(self, origin_lat, origin_lng, dest_lat, dest_lng, mode="walking"):
        params = {
            "origin": f"{origin_lat},{origin_lng}",
            "destination": f"{dest_lat},{dest_lng}",
            "mode": mode,
            "key": self.api_key
        }
        async with httpx.AsyncClient() as client:
            r = await client.get(self.BASE_DIRECTIONS_URL, params=params, timeout=5.0)
            data = r.json()
            if data.get("status") != "OK":
                return None

            route = data["routes"][0]
            leg = route["legs"][0]
            steps = []
            for step in leg["steps"]:
                instruction = re.sub(r'<[^>]+>', '', step["html_instructions"])
                steps.append({
                    "instruction": instruction,
                    "distance": step["distance"]["text"],
                    "distance_meters": step["distance"]["value"],
                    "start_lat": step["start_location"]["lat"],
                    "start_lng": step["start_location"]["lng"],
                    "end_lat": step["end_location"]["lat"],
                    "end_lng": step["end_location"]["lng"],
                })

            return {
                "total_distance": leg["distance"]["text"],
                "total_duration": leg["duration"]["text"],
                "steps": steps,
                "end_address": leg["end_address"]
            }

    async def geocode_address(self, address):
        params = {"address": address, "key": self.api_key}
        async with httpx.AsyncClient() as client:
            r = await client.get(self.BASE_GEOCODE_URL, params=params, timeout=5.0)
            data = r.json()
            if data.get("status") != "OK" or not data.get("results"):
                return None
            res = data["results"][0]
            return {
                "lat": res["geometry"]["location"]["lat"],
                "lng": res["geometry"]["location"]["lng"],
                "address": res["formatted_address"]
            }

maps_service = GoogleMapsService()

async def handle_maps_call(function_call):
    """Bridge for Gemini Tool Calls."""
    args = function_call.args
    lat, lng = args['latitude'], args['longitude']
    
    if function_call.name == "get_nearby_landmarks":
        results = await maps_service.find_nearby_places(lat, lng, args.get("place_type"), args.get("radius_meters", 500))
        if isinstance(results, dict) and "error" in results:
            return {"places": [f"Error: {results['error']}"]}
        return {"places": [f"{p['name']} ({p['vicinity']})" for p in results]}
    
    return {"error": "Unknown tool"}
