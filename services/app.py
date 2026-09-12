import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from services.data_analyzer import DataAnalyzer
from services.visualizer import DataVisualizer
from services.ai_agent import DataAIAgent

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-12345")

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory session DataFrame cache store
DATASET_CACHE = {}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_df() -> tuple[pd.DataFrame | None, str | None]:
    file_path = session.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return None, "No active dataset uploaded. Please upload a dataset first."
        
    if file_path in DATASET_CACHE:
        return DATASET_CACHE[file_path], None
        
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        DATASET_CACHE[file_path] = df
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"


# -----------------------------------------------------------------------------
# Web & API Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "dataset" not in request.files:
        return jsonify({"error": "No file part in upload request."}), 400
        
    file = request.files["dataset"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        
        session["file_path"] = save_path
        session["file_name"] = filename
        session["file_size"] = f"{round(os.path.getsize(save_path) / 1024, 1)} KB"

        # Load dataframe into memory
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(save_path)
            else:
                df = pd.read_excel(save_path)
            DATASET_CACHE[save_path] = df
            
            analyzer = DataAnalyzer(df)
            metrics = analyzer.get_summary_metrics()
            
            return jsonify({
                "message": "Dataset uploaded and analyzed successfully!",
                "filename": filename,
                "filesize": session["file_size"],
                "metrics": metrics
            })
        except Exception as e:
            return jsonify({"error": f"Corrupted or unreadable file: {str(e)}"}), 400

    return jsonify({"error": "Unsupported format. Allowed formats: CSV, XLSX, XLS."}), 400


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

    ai_agent = DataAIAgent(df)
    result = ai_agent.ask(question)
    
    # If the response requests a chart, generate Plotly JSON spec
    chart_json = None
    if result.get("chart_recommended"):
        visualizer = DataVisualizer()
        group_col = result.get("x_col")
        target_col = result.get("y_col")
        analyzer = DataAnalyzer(df)
        agg_df = analyzer.safe_aggregate(group_col, target_col, "sum")
        chart_json = visualizer.create_chart(agg_df, result.get("chart_type", "bar"), group_col, target_col, result.get("title"))

    return jsonify({
        "answer": result.get("answer"),
        "chart": chart_json
    })


@app.route("/suggestions", methods=["GET"])
def suggestions():
    df, error = get_current_df()
    if error:
        return jsonify({"suggestions": []})

    ai_agent = DataAIAgent(df)
    return jsonify({"suggestions": ai_agent.generate_suggested_questions()})


@app.route("/insights", methods=["GET"])
def insights():
    df, error = get_current_df()
    if error:
        return jsonify({"error": error}), 400

    ai_agent = DataAIAgent(df)
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)