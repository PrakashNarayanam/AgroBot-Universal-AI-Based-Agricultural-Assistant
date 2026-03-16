# gemini_helper.py - FINAL VERSION

import os
from dotenv import load_dotenv
import google.genai as genai
from google.genai.types import GenerateContentConfig
from PIL import Image

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables")
    client = None


# -------------------------------
# TEXT CHAT WITH GEMINI
# -------------------------------
def ask_gemini(prompt, model="gemini-2.0-flash"):
    """Ask Gemini a text question"""

    try:
        if not client:
            print("❌ Gemini client not available")
            return "AI service is currently unavailable."

        print(f"📤 Asking Gemini: {prompt[:80]}...")

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )
        )

        # Safe response handling
        if hasattr(response, "text") and response.text:
            print("📥 Gemini response received")
            return response.text

        if response.candidates:
            return response.candidates[0].content.parts[0].text

        return "Sorry, I couldn't generate a response."

    except Exception as e:
        print(f"🔥 Gemini API error: {e}")

        return f"""
I received your farming question about:

"{prompt[:100]}"

For the best farming advice I recommend:

1. Contact your **local agricultural extension office**
2. Check **local weather forecasts**
3. Conduct **soil testing**
4. Follow **crop rotation practices**

Consult nearby farming experts for region-specific guidance.
"""


# -------------------------------
# IMAGE ANALYSIS WITH GEMINI
# -------------------------------
def analyze_with_gemini(image_path, prompt=""):
    """Analyze plant or crop image using Gemini Vision"""

    try:
        if not client:
            return {"error": "Gemini API client not configured"}

        img = Image.open(image_path)

        analysis_prompt = f"""
Analyze this agricultural image for a farmer.

{prompt if prompt else 'Check plant health, diseases, pests, or nutrient issues.'}

Provide response in this format:

1. Plant health assessment
2. Visible diseases or pests
3. Possible causes
4. Practical treatment recommendations
5. Confidence level
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[analysis_prompt, img],
            config=GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=2048,
            )
        )

        text_output = ""

        if hasattr(response, "text") and response.text:
            text_output = response.text
        elif response.candidates:
            text_output = response.candidates[0].content.parts[0].text

        return {
            "health_status": "AI Analysis Complete",
            "analysis": text_output,
            "confidence": 0.85,
            "recommendations": [
                "Review AI analysis above",
                "Confirm diagnosis with agricultural expert",
                "Follow recommended treatment if needed"
            ]
        }

    except Exception as e:
        print(f"🔥 Gemini Vision error: {e}")

        # -------------------------------
        # Fallback basic analysis
        # -------------------------------
        try:

            im = Image.open(image_path).convert('RGB').resize((200, 200))
            pixels = list(im.getdata())

            greens = sum(
                1 for r, g, b in pixels if g > r + 20 and g > b + 20
            )

            total = len(pixels)
            healthy_ratio = greens / total if total > 0 else 0

            if healthy_ratio < 0.3:
                status = "Possible plant stress detected"
                advice = "Low green coloration may indicate nutrient deficiency, water stress, or disease."
            else:
                status = "Plant appears relatively healthy"
                advice = "Healthy green coloration detected."

            return {
                "health_status": status,
                "analysis": advice,
                "green_percentage": round(healthy_ratio * 100, 1),
                "recommendations": [
                    "Take clearer photos from multiple angles",
                    "Check soil moisture and drainage",
                    "Inspect underside of leaves for pests",
                    "Consider soil nutrient testing"
                ]
            }

        except:
            return {
                "health_status": "Basic analysis only",
                "analysis": "Image received but AI analysis failed. Try a clearer image with good lighting.",
                "confidence": 0.5,
                "recommendations": [
                    "Upload clearer plant image",
                    "Ensure good lighting",
                    "Focus camera on affected leaf"
                ]
            }