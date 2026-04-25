# Run once at startup, reuse cache_name per session
from google.genai import caching
import datetime

cache = client.caches.create(
    model="gemini-2.0-flash-exp",
    contents=[types.Content(parts=[types.Part(text=SYSTEM_PROMPT)], role="user")],
    ttl=datetime.timedelta(hours=1),
    display_name="walking-assistant-context"
)
# Pass cache.name into your Live session config