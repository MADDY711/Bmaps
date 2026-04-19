# maps.py
import httpx, os
from dotenv import load_dotenv
load_dotenv()

async def handle_maps_call(function_call):
    args = function_call.args
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        print("❌ GOOGLE_MAPS_API_KEY not set")
        return {"places": ["Maps unavailable — API key not configured"]}

    params = {
        "location": f"{args['latitude']},{args['longitude']}",
        "radius": args.get("radius_meters", 200),
        "key": api_key
    }
    if "place_type" in args:
        params["type"] = args["place_type"]

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params,
                timeout=5.0   # don't hang the session if Maps is slow
            )
            data = r.json()

            if data.get("status") not in ("OK", "ZERO_RESULTS"):
                print(f"❌ Maps API error: {data.get('status')} — {data.get('error_message', '')}")
                return {"places": ["Could not fetch nearby places right now"]}

            results = data.get("results", [])[:5]
            places = [
                f"{p['name']} ({p.get('vicinity', 'nearby')})"
                for p in results
            ]
            return {"places": places if places else ["No places found nearby"]}

    except Exception as e:
        print(f"❌ Maps call failed: {e}")
        return {"places": ["Maps lookup failed — check connection"]}