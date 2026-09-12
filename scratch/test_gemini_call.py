import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
print(f"API Key: {api_key[:10]}... (len: {len(api_key) if api_key else 0})")
print(f"Model: {model_name}")

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Hello, reply with 1 word: 'Connected'.")
    print("Response:", response.text)
except Exception as e:
    print("Error:", type(e), e)
