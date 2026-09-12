import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from services.data_analyzer import DataAnalyzer
from services.visualizer import DataVisualizer
from services.ai_agent import DataAIAgent
from services.code_executor import get_fallback_log_entries
from services.voice_transcriber import transcribe_audio

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-12345")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory cache: { file_path: { "df": DataFrame, "schema": str } }
# Schema is computed ONCE at upload time and reused on every chat request
# so DataAIAgent never recomputes it per question.
DATASET_CACHE = {}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_dataframe_safely(file_path: str) -> pd.DataFrame:
    """Robust loader that handles large CSVs, different encodings (UTF-8, Latin-1, CP1252), and Excel."""
    if file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False, on_bad_lines="skip")
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            break
    return pd.read_csv(file_path, low_memory=False)

def get_current_df() -> tuple[pd.DataFrame | None, str | None]:
    """Returns the cached DataFrame for the active session file."""
    file_path = session.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return None, "No active dataset uploaded. Please upload a dataset first."

    if file_path in DATASET_CACHE:
        return DATASET_CACHE[file_path]["df"], None

    try:
        df = load_dataframe_safely(file_path)
        analyzer = DataAnalyzer(df)
        DATASET_CACHE[file_path] = {
            "df": df,
            "schema": analyzer.get_schema_summary()
        }
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"

def get_cached_schema() -> str | None:
    """Returns the pre-built schema string for the active session file, or None."""
    file_path = session.get("file_path")
    if file_path and file_path in DATASET_CACHE:
        return DATASET_CACHE[file_path].get("schema")
    return None

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the 500 MB limit. Please upload a smaller file."}), 413

@app.errorhandler(500)
def handle_500(error):
    return jsonify({"error": "Internal Server Error. Please check server logs."}), 500

@app.errorhandler(400)
def handle_400(error):
    return jsonify({"error": "Bad Request."}), 400


# -----------------------------------------------------------------------------
# Web & API Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "dataset" not in request.files:
            return jsonify({"error": "No file part in upload request."}), 400
            
        file = request.files["dataset"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected."}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported format. Allowed formats: CSV, XLSX, XLS."}), 400

        import time, uuid
        original_name = file.filename
        clean_name = secure_filename(original_name)
        if not clean_name or clean_name.startswith("."):
            base = "dataset"
            ext = os.path.splitext(original_name)[1].lower() or ".csv"
            clean_name = f"{base}_{int(time.time())}{ext}"

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_name)

        # Handle Windows file locking if the active file is re-uploaded
        try:
            file.save(save_path)
        except Exception:
            base, ext = os.path.splitext(clean_name)
            clean_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_name)
            file.save(save_path)
            
        session["file_path"] = save_path
        session["file_name"] = clean_name
        file_bytes = os.path.getsize(save_path)
        if file_bytes >= 1024 * 1024:
            session["file_size"] = f"{round(file_bytes / (1024 * 1024), 1)} MB"
        else:
            session["file_size"] = f"{round(file_bytes / 1024, 1)} KB"

        # Load dataframe into memory and build schema cache
        df = load_dataframe_safely(save_path)

        analyzer = DataAnalyzer(df)
        metrics = analyzer.get_summary_metrics()
        schema = analyzer.get_schema_summary()   # Cached once — reused on every /chat

        DATASET_CACHE[save_path] = {"df": df, "schema": schema}

        return jsonify({
            "message": "Dataset uploaded and analyzed successfully!",
            "filename": clean_name,
            "filesize": session["file_size"],
            "metrics": metrics
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Upload processing failed: {str(e)}"}), 500


@app.route("/dataset", methods=["GET"])
def get_dataset_info():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    analyzer = DataAnalyzer(df)
    return jsonify({
        "metrics": analyzer.get_summary_metrics(),
        "preview": analyzer.get_preview(15),
        "columns": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns)
    })


@app.route("/profile", methods=["GET"])
def get_profile():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    analyzer = DataAnalyzer(df)
    return jsonify({"profiles": analyzer.get_column_profiles()})


@app.route("/quality", methods=["GET"])
def get_quality():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    analyzer = DataAnalyzer(df)
    return jsonify({"quality": analyzer.get_quality_report()})


@app.route("/chat", methods=["POST"])
def chat():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    data = request.json or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question provided."}), 400

    # Pass cached schema so the agent never recomputes it
    ai_agent = DataAIAgent(df, cached_schema=get_cached_schema())
    result = ai_agent.ask(question)

    # Chart JSON is already built inside ask() — just pass it through
    return jsonify({
        "answer": result.get("answer"),
        "chart": result.get("chart")
    })


@app.route("/suggestions", methods=["GET"])
def suggestions():
    df, error = get_current_df()
    if error:
        return jsonify({"suggestions": []})

    ai_agent = DataAIAgent(df, cached_schema=get_cached_schema())
    return jsonify({"suggestions": ai_agent.generate_suggested_questions()})


@app.route("/insights", methods=["GET"])
def insights():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    ai_agent = DataAIAgent(df, cached_schema=get_cached_schema())
    return jsonify({"insights": ai_agent.generate_automated_insights()})


@app.route("/visualize", methods=["POST"])
def visualize():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    data = request.json or {}
    chart_type = data.get("chart_type", "bar")
    x_col = data.get("x_col")
    y_col = data.get("y_col")

    if not x_col or x_col not in df.columns:
        return jsonify({"error": "Invalid X column selection."}), 400

    analyzer = DataAnalyzer(df)
    chart_data = df
    
    # Pre-aggregate bar/pie charts if y_col is numeric
    if chart_type.lower() in ["bar", "pie"] and y_col and y_col in df.columns:
        chart_data = analyzer.safe_aggregate(x_col, y_col, "sum", top_n=15)

    chart_json = DataVisualizer.create_chart(chart_data, chart_type, x_col, y_col)
    return jsonify({"chart": chart_json})


@app.route("/fallback-log", methods=["GET"])
def fallback_log():
    """
    Admin endpoint: returns the last N questions that fell through to Pattern B.
    Use this to see which new analysis functions to add to the library.
    GET /fallback-log?n=50
    """
    n = int(request.args.get("n", 50))
    entries = get_fallback_log_entries(last_n=n)
    return jsonify({
        "total_logged": len(entries),
        "entries": entries,
        "log_path": "logs/fallback_queries.jsonl",
        "note": "These are questions that had no Pattern A function match. Add functions for recurring patterns."
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts a raw audio file upload from the browser's MediaRecorder.
    Passes it to Gemini for multilingual speech-to-text transcription.
    Returns: { "text": "...", "error": null } or { "text": null, "error": "..." }
    """
    if "audio" not in request.files:
        return jsonify({"text": None, "error": "No audio file in request."}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({"text": None, "error": "Empty audio file received."}), 400

    # Determine MIME type from content type header or filename extension
    mime_type = audio_file.content_type or "audio/webm"
    # Normalize: browsers may send 'audio/webm;codecs=opus' — strip codec info
    mime_type = mime_type.split(";")[0].strip()
    if not mime_type.startswith("audio/"):
        mime_type = "audio/webm"

    result = transcribe_audio(audio_bytes, mime_type)
    status_code = 200 if result["text"] else 422
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(debug=True, port=5000)