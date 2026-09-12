"""
VoiceTranscriber — Gemini Audio Transcription Service
======================================================

Accepts raw audio bytes from the browser (WebM/OGG/WAV via MediaRecorder),
sends them to Gemini as inline_data with the appropriate MIME type,
and returns the transcribed text.

Supported languages: English, Bengali (বাংলা), Hindi (हिन्दी / Hinglish).
Gemini 1.5 Flash / 2.0 Flash handle Indic language audio natively.

Token cost: ~200-400 tokens per short utterance (audio inline_data).
"""

import os
import base64
from typing import Optional


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
    """
    Transcribe audio bytes using the Gemini API.

    Args:
        audio_bytes: Raw audio bytes captured from the browser MediaRecorder.
        mime_type:   MIME type of the audio (e.g., 'audio/webm', 'audio/wav',
                     'audio/ogg', 'audio/mp4').

    Returns:
        dict with keys:
            text  – transcribed text (or None on failure)
            error – error message (or None on success)
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return {"text": None, "error": "GEMINI_API_KEY not configured."}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError:
        return {"text": None, "error": "google-generativeai package not installed."}

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Gemini accepts inline audio via base64 encoded content
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    transcription_prompt = (
        "Transcribe the speech in this audio clip exactly as spoken. "
        "The speaker may use English, Bengali (বাংলা), Hindi (हिन्दी), or a mix. "
        "Output ONLY the transcribed text — no labels, no translation, no explanation."
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([
            transcription_prompt,
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": audio_b64,
                }
            }
        ])
        text = response.text.strip() if response.text else ""
        if not text:
            return {"text": None, "error": "No speech detected in audio."}
        return {"text": text, "error": None}

    except Exception as exc:
        return {"text": None, "error": f"Transcription failed: {str(exc)}"}
