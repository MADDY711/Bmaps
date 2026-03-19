# maps.py
import httpx, os

async def handle_maps_call(function_call):
    args = function_call.args
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    params = {
        "location": f"{args['latitude']},{args['longitude']}",
        "radius": args.get("radius_meters", 200),
        "key": api_key
    }
    if "place_type" in args:
        params["type"] = args["place_type"]

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params=params
        )
        results = r.json().get("results", [])[:5]  # Top 5 only
        
        places = [
            f"{p['name']} ({p.get('vicinity', 'nearby')})"
            for p in results
        ]
        return {"places": places}