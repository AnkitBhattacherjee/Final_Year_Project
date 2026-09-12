import sys
sys.path.insert(0, '.')
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

models_to_test = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-3.5-flash-lite"]

for m in models_to_test:
    try:
        model = genai.GenerativeModel(m)
        resp = model.generate_content("Ping")
        print(f"Model {m}: SUCCESS -> {resp.text.strip()}")
    except Exception as e:
        print(f"Model {m}: FAILED -> {e}")
