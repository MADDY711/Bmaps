import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: API Key not found.")
else:
    client = genai.Client(api_key=api_key)
    print("\n--- YOUR AVAILABLE MODELS ---")
    try:
        # Just list everything. No filtering means no errors.
        for m in client.models.list():
            print(f"- {m.name}")
    except Exception as e:
        print(f"Error: {e}")