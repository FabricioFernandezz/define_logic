"""Run: python list_models.py  — lists all Gemini models available for this API key."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / "config" / ".env")

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== Models supporting generateContent ===")
for m in client.models.list():
    methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
    if "generateContent" in str(methods):
        print(f"  {m.name}")
