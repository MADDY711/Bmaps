import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def test_vision():
    """Test Gemini Vision with a sample image"""
    
    print("🔍 Testing Gemini Vision API...")
    print("=" * 60)
    
    try:
        # Load a test image from URL
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Use Gemini 1.5 Flash (fastest, cheapest)
        model = genai.GenerativeModel('gemini-2.5-flash-image')
        
        prompt = """You are a helpful AI companion for a visually impaired person.
        Describe what you see in this image clearly and concisely.
        Focus on objects, their positions, and any potential obstacles."""
        
        response = model.generate_content([prompt, img])
        
        print("✅ SUCCESS! Gemini Response:")
        print("-" * 60)
        print(response.text)
        print("-" * 60)
        
        # Show usage
        print(f"\n📊 Usage Info:")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
        print(f"Total tokens: {response.usage_metadata.total_token_count}")
        
        # Calculate cost (Flash pricing)
        cost = (response.usage_metadata.total_token_count / 1_000_000) * 0.075
        print(f"💰 Cost: ${cost:.6f} (basically free!)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Make sure billing is enabled at https://console.cloud.google.com/billing")
        print("3. Enable Gemini API at https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
        return False

def test_text_generation():
    """Test Gemini for conversation"""
    
    print("\n\n💬 Testing Gemini Conversation...")
    print("=" * 60)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        chat = model.start_chat(history=[])
        
        # Simulate a conversation
        messages = [
            "Hello! I'm visually impaired and need help walking.",
            "What should I be careful about when walking indoors?",
        ]
        
        for msg in messages:
            print(f"\n👤 User: {msg}")
            response = chat.send_message(msg)
            print(f"🤖 AI: {response.text}")
        
        print("\n✅ Conversation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_streaming():
    """Test streaming responses (for real-time feel)"""
    
    print("\n\n⚡ Testing Streaming Responses...")
    print("=" * 60)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = "Describe a typical office environment for someone who can't see it."
        
        print("🤖 AI: ", end="", flush=True)
        
        response = model.generate_content(prompt, stream=True)
        
        for chunk in response:
            print(chunk.text, end="", flush=True)
        
        print("\n\n✅ Streaming test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  VISION COMPANION - GEMINI API TEST")
    print("=" * 60)
    print()
    
    # Run all tests
    vision_ok = test_vision()
    text_ok = test_text_generation()
    stream_ok = test_streaming()
    
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"Vision API:       {'✅ PASS' if vision_ok else '❌ FAIL'}")
    print(f"Conversation API: {'✅ PASS' if text_ok else '❌ FAIL'}")
    print(f"Streaming API:    {'✅ PASS' if stream_ok else '❌ FAIL'}")
    print()
    
    if vision_ok and text_ok and stream_ok:
        print("🎉 ALL TESTS PASSED!")
        print("\n✨ Gemini is working perfectly!")
        print("\n📋 Next Steps:")
        print("1. ✅ API confirmed working")
        print("2. 🔨 Build FastAPI backend")
        print("3. 📱 Create Android app")
        print("4. 🔗 Connect them together")
        print("\n🚀 Ready to start building!")
    else:
        print("⚠️  Some tests failed.")
        print("Check the troubleshooting steps above.")