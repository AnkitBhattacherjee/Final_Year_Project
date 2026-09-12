import sys
sys.path.insert(0, '.')
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-lite-latest"]:
    try:
        model = genai.GenerativeModel(m)
        resp = model.generate_content("Ping")
        print(f"Model {m}: SUCCESS -> {resp.text.strip()}")
    except Exception as e:
        print(f"Model {m}: FAILED -> {e}")
