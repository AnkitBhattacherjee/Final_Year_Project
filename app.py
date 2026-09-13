import sys
import os
import re
import time
import uuid
import json
from typing import Optional, Tuple

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory cache:
# { file_path: { "df": DataFrame, "schema": str } }
DATASET_CACHE = {}


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_dataframe_safely(file_path: str) -> pd.DataFrame:
    """Load CSV/XLS/XLSX with common encoding fallbacks."""
    if file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    for enc in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=enc,
                low_memory=False,
                on_bad_lines="skip",
            )
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            break

    return pd.read_csv(file_path, low_memory=False)


def get_current_df() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return the cached DataFrame for the active session."""
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
            "schema": analyzer.get_schema_summary(),
        }

        return df, None

    except Exception as exc:
        return None, f"Failed to load dataset: {str(exc)}"


def get_cached_schema() -> Optional[str]:
    file_path = session.get("file_path")

    if file_path and file_path in DATASET_CACHE:
        return DATASET_CACHE[file_path].get("schema")

    return None


# -----------------------------------------------------------------------------
# Universal visualization aggregation
# -----------------------------------------------------------------------------
SUPPORTED_AGGREGATIONS = {"auto", "sum", "mean", "count", "min", "max"}

# Words that normally describe additive measures.
ADDITIVE_KEYWORDS = {
    "sales", "sale", "revenue", "turnover", "income", "earning", "earnings",
    "profit", "cost", "expense", "spend", "spending", "amount", "value",
    "quantity", "qty", "units", "unit", "orders", "order count", "transactions",
    "transaction count", "volume", "stock", "inventory", "total",
}

# Words that normally describe non-additive measures.
AVERAGE_KEYWORDS = {
    "margin", "ratio", "percentage", "percent", "rate", "average", "avg",
    "mean", "median", "rating", "score", "satisfaction", "discount",
    "discount rate", "profit margin", "profit ratio", "conversion rate",
    "growth rate", "share", "percentage", "pct", "age", "temperature",
    "duration", "days", "hours", "time", "lead time",
}

ID_KEYWORDS = {
    "id", "identifier", "key", "code", "zip", "zipcode", "postal", "phone",
    "latitude", "longitude", "lat", "lon", "index", "unnamed",
}


def _normalized_column_name(column: str) -> str:
    """Normalize a column name for semantic matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()


def _contains_semantic_keyword(column: str, keywords: set) -> bool:
    normalized = _normalized_column_name(column)
    tokens = set(normalized.split())

    for keyword in keywords:
        keyword_norm = _normalized_column_name(keyword)

        if keyword_norm in normalized:
            return True

        if keyword_norm in tokens:
            return True

    return False


def _looks_like_identifier(series: pd.Series, column: str) -> bool:
    """
    Detect numeric identifier-like columns so AUTO does not sum IDs.

    Name-based detection is combined with uniqueness. A highly unique integer
    column called Customer_ID/Order_ID should be counted rather than summed.
    """
    if _contains_semantic_keyword(column, ID_KEYWORDS):
        return True

    non_null = series.dropna()

    if non_null.empty:
        return False

    unique_ratio = non_null.nunique(dropna=True) / len(non_null)

    # High-cardinality numeric columns are often IDs/codes.
    if unique_ratio >= 0.98 and len(non_null) >= 20:
        return True

    return False


def _is_datetime_like(series: pd.Series, column: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    normalized = _normalized_column_name(column)

    if not any(k in normalized for k in
               ("date", "datetime", "timestamp", "time", "day", "month", "year")):
        return False

    # Only classify as datetime if parsing succeeds for a substantial sample.
    sample = series.dropna().head(100)

    if sample.empty:
        return False

    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.80
    except Exception:
        return False


def choose_auto_aggregation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[str, str]:
    """
    Decide the safest aggregation for a visualization.

    Returns:
        (aggregation, reason)

    Priority:
      1. Non-numeric/date Y -> count
      2. Identifier-like numeric Y -> count
      3. Explicit ratio/rate/percentage/average-style semantic -> mean
      4. Explicit additive business measure -> sum
      5. Generic numeric measure -> mean

    The final generic-numeric fallback is MEAN because arbitrary numeric
    measures (age, price, temperature, scores, etc.) are not safely additive.
    Users can explicitly choose SUM when totalization is desired.
    """
    series = df[y_col]

    if _is_datetime_like(series, y_col):
        return "count", "Y column is date/time-like, so counting observations is used."

    if not pd.api.types.is_numeric_dtype(series):
        return "count", "Y column is categorical/text, so frequency count is used."

    if _looks_like_identifier(series, y_col):
        return "count", "Y column looks like an identifier/code, so summing it is avoided."

    if _contains_semantic_keyword(y_col, AVERAGE_KEYWORDS):
        return "mean", "Y column represents a ratio, rate, percentage, score, average, or non-additive measure."

    if _contains_semantic_keyword(y_col, ADDITIVE_KEYWORDS):
        return "sum", "Y column represents an additive business measure."

    # Generic numeric fallback.
    return "mean", "Generic numeric measure: mean is safer than assuming values are additive."


def aggregate_for_visualization(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggregation: str = "auto",
    top_n: int = 15,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Universal aggregation engine for charts.

    Returns:
        chart_data, actual_aggregation, decision_reason

    Important:
      - The selected y_col is ALWAYS retained as the chart metric column.
      - COUNT creates a column with the exact selected y_col name so the
        existing DataVisualizer can consume it without frontend changes.
      - Manual aggregation always wins over AUTO.
    """
    if x_col not in df.columns:
        raise ValueError(f"Invalid X column selection: {x_col}")

    if y_col:
        if y_col not in df.columns:
            raise ValueError(f"Invalid Y column selection: {y_col}")
    else:
        raise ValueError("A Y column is required for this chart.")

    if x_col == y_col:
        raise ValueError("X and Y columns must be different.")

    requested = str(aggregation or "auto").strip().lower()

    if requested not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Choose one of: {', '.join(sorted(SUPPORTED_AGGREGATIONS))}."
        )

    if requested == "auto":
        actual_agg, reason = choose_auto_aggregation(df, x_col, y_col)
    else:
        actual_agg = requested
        reason = f"Manual aggregation selected: {actual_agg}."

    # Work only with selected columns. This prevents an accidental fallback
    # to another metric such as Profit when Y is Profit_Margin.
    working = df[[x_col, y_col]].copy()

    # Drop rows where X is missing. For Y, SUM/MEAN/MIN/MAX naturally ignore
    # missing numeric values; COUNT below counts non-null Y observations.
    working = working.dropna(subset=[x_col])

    if actual_agg == "count":
        grouped = (
            working.groupby(x_col, dropna=False)[y_col]
            .count()
            .reset_index()
        )
    else:
        numeric_y = pd.to_numeric(working[y_col], errors="coerce")
        working = working.assign(**{y_col: numeric_y})
        working = working.dropna(subset=[y_col])

        if working.empty:
            raise ValueError(f"Y column '{y_col}' contains no usable numeric values.")

        grouped = (
            working.groupby(x_col, dropna=False)[y_col]
            .agg(actual_agg)
            .reset_index()
        )

    # Remove NaN/inf values that can break chart serialization.
    if pd.api.types.is_numeric_dtype(grouped[y_col]):
        grouped[y_col] = grouped[y_col].replace([float("inf"), float("-inf")], pd.NA)
        grouped = grouped.dropna(subset=[y_col])

    # Sorting should reflect the selected aggregation.
    ascending = actual_agg == "min"

    grouped = grouped.sort_values(
        by=y_col,
        ascending=ascending,
        kind="stable",
    )

    # Keep charts readable. For "min", the smallest values are shown first;
    # for the other common rankings, the largest values are shown first.
    grouped = grouped.head(max(1, int(top_n))).reset_index(drop=True)

    return grouped, actual_agg, reason


def get_aggregation_metadata(df: pd.DataFrame, y_col: str) -> dict:
    """
    Metadata endpoint/helper for the frontend.

    Gives the UI a recommended AUTO aggregation plus the available manual
    overrides. This allows the frontend to show the real aggregation rather
    than guessing from a column name.
    """
    if y_col not in df.columns:
        raise ValueError(f"Invalid Y column: {y_col}")

    auto_agg, reason = choose_auto_aggregation(df, "", y_col)

    return {
        "requested": "auto",
        "resolved": auto_agg,
        "reason": reason,
        "options": ["auto", "sum", "mean", "count", "min", "max"],
    }


# -----------------------------------------------------------------------------
# Error handlers
# -----------------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File size exceeds the 500 MB limit. Please upload a smaller file."
    }), 413


@app.errorhandler(500)
def handle_500(error):
    return jsonify({
        "error": "Internal Server Error. Please check server logs."
    }), 500


@app.errorhandler(400)
def handle_400(error):
    return jsonify({"error": "Bad Request."}), 400


# -----------------------------------------------------------------------------
# Web & API routes
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
            return jsonify({
                "error": "Unsupported format. Allowed formats: CSV, XLSX, XLS."
            }), 400

        original_name = file.filename
        clean_name = secure_filename(original_name)

        if not clean_name or clean_name.startswith("."):
            base = "dataset"
            ext = os.path.splitext(original_name)[1].lower() or ".csv"
            clean_name = f"{base}_{int(time.time())}{ext}"

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_name)

        # Handle Windows file locking if the active file is re-uploaded.
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

        # Load dataframe and build schema once.
        df = load_dataframe_safely(save_path)
        analyzer = DataAnalyzer(df)

        metrics = analyzer.get_summary_metrics()
        schema = analyzer.get_schema_summary()

        DATASET_CACHE[save_path] = {
            "df": df,
            "schema": schema,
        }

        return jsonify({
            "message": "Dataset uploaded and analyzed successfully!",
            "filename": clean_name,
            "filesize": session["file_size"],
            "metrics": metrics,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": f"Upload processing failed: {str(exc)}"
        }), 500


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
        "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
    })


@app.route("/profile", methods=["GET"])
def get_profile():
    df, error = get_current_df()

    if error:
        return jsonify({"error": error}), 400

    analyzer = DataAnalyzer(df)

    return jsonify({
        "profiles": analyzer.get_column_profiles()
    })


@app.route("/quality", methods=["GET"])
def get_quality():
    df, error = get_current_df()

    if error:
        return jsonify({"error": error}), 400

    analyzer = DataAnalyzer(df)

    return jsonify({
        "quality": analyzer.get_quality_report()
    })


@app.route("/chat", methods=["POST"])
def chat():
    df, error = get_current_df()

    if error:
        return jsonify({"error": error}), 400

    data = request.json or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Empty question provided."}), 400

    # Cached schema avoids recomputation on every question.
    ai_agent = DataAIAgent(
        df,
        cached_schema=get_cached_schema(),
    )

    result = ai_agent.ask(question)

    # The agent is the single source of truth for AI questions.
    # IMPORTANT: result_df is the already-computed analysis result. It is
    # serialized here so the frontend can send ONLY these precomputed rows to
    # /visualize/analysis. The visualization layer never re-aggregates the raw
    # dataset for an AI question.
    result_df = result.get("result_df")
    result_data = []
    if isinstance(result_df, pd.DataFrame):
        try:
            result_data = json.loads(
                result_df.to_json(orient="records", date_format="iso")
            )
        except Exception:
            result_data = []

    return jsonify({
        "answer": result.get("answer"),
        "chart": result.get("chart"),
        "calculation_status": result.get("calculation_status", "calculated"),
        "reason": result.get("reason"),
        "x_col": result.get("x_col"),
        "y_col": result.get("y_col"),
        "chart_type": result.get("chart_type"),
        "chart_title": result.get("chart_title"),
        "chart_recommended": result.get(
            "chart_recommended",
            bool(result.get("chart") or result_data)
        ),
        "result_data": result_data,
        "source": "agent_precomputed_result",
    })


@app.route("/suggestions", methods=["GET"])
def suggestions():
    df, error = get_current_df()

    if error:
        return jsonify({"suggestions": []})

    ai_agent = DataAIAgent(
        df,
        cached_schema=get_cached_schema(),
    )

    return jsonify({
        "suggestions": ai_agent.generate_suggested_questions()
    })


@app.route("/insights", methods=["GET"])
def insights():
    df, error = get_current_df()

    if error:
        return jsonify({"error": error}), 400

    ai_agent = DataAIAgent(
        df,
        cached_schema=get_cached_schema(),
    )

    return jsonify({
        "insights": ai_agent.generate_automated_insights()
    })


# -----------------------------------------------------------------------------
# Visualization metadata endpoint
# -----------------------------------------------------------------------------
@app.route("/visualize/aggregation-options", methods=["GET"])
def visualization_aggregation_options():
    """
    Frontend helper.

    GET /visualize/aggregation-options?y_col=Profit_Margin

    Returns the AUTO recommendation and all manual choices.
    """
    df, error = get_current_df()

    if error:
        return jsonify({"error": error}), 400

    y_col = request.args.get("y_col")

    if not y_col or y_col not in df.columns:
        return jsonify({"error": "Invalid Y column selection."}), 400

    try:
        return jsonify({
            "y_col": y_col,
            **get_aggregation_metadata(df, y_col),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# -----------------------------------------------------------------------------
# Agent-result visualization route — SOURCE-OF-TRUTH PATH
# -----------------------------------------------------------------------------
@app.route("/visualize/analysis", methods=["POST"])
def visualize_analysis_result():
    """
    Render a chart from an analysis result that has ALREADY been calculated
    by DataAIAgent.

    This endpoint deliberately does NOT load or aggregate the raw dataset.
    The caller must provide the precomputed result rows plus the exact X/Y
    fields selected by the agent. This guarantees that the chart is a visual
    representation of the agent's answer, not a second calculation.

    Request body:
    {
        "chart_type": "bar",
        "x_col": "Payment_Method",
        "y_col": "Average_Order_Profit",
        "title": "Average Order Profit by Payment Method",
        "result_data": [
            {"Payment_Method": "Credit Card", "Average_Order_Profit": 15256.09}
        ]
    }

    For unavailable calculations, send calculation_status="unavailable".
    No chart is generated in that case.
    """
    data = request.get_json(silent=True) or {}

    if str(data.get("calculation_status", "calculated")).lower() == "unavailable":
        return jsonify({
            "chart": None,
            "calculation_status": "unavailable",
            "reason": data.get("reason"),
        })

    chart_type = str(data.get("chart_type", "bar")).strip()
    x_col = data.get("x_col")
    y_col = data.get("y_col")
    title = data.get("title") or data.get("chart_title")
    result_data = data.get("result_data")

    if not x_col:
        return jsonify({"error": "Agent result is missing x_col."}), 400
    if not y_col:
        return jsonify({"error": "Agent result is missing y_col."}), 400
    if not isinstance(result_data, list):
        return jsonify({
            "error": "Agent result must contain result_data as a list of precomputed rows."
        }), 400
    if not result_data:
        return jsonify({"chart": None, "calculation_status": "calculated", "rows_used": 0})

    try:
        # IMPORTANT: no raw DataFrame and no aggregation_for_visualization()
        # are used here. The DataFrame consists only of agent-computed values.
        chart_data = pd.DataFrame(result_data)

        if x_col not in chart_data.columns:
            return jsonify({"error": f"Agent result does not contain X column '{x_col}'."}), 400
        if y_col not in chart_data.columns:
            return jsonify({"error": f"Agent result does not contain Y column '{y_col}'."}), 400

        chart_json = DataVisualizer.create_chart(
            chart_data,
            chart_type,
            x_col,
            y_col,
            title=title,
        )

        return jsonify({
            "chart": chart_json,
            "calculation_status": "calculated",
            "x_col": x_col,
            "y_col": y_col,
            "chart_type": chart_type,
            "chart_title": title,
            "rows_used": int(len(chart_data)),
            "source": "agent_precomputed_result",
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Visualization failed: {str(exc)}"}), 500


# -----------------------------------------------------------------------------
# Manual/custom visualization route
# -----------------------------------------------------------------------------
@app.route("/visualize", methods=["POST"])
def visualize():
    """
    Generate a chart for the manual/custom Visualization Studio.

    NOTE: This route is intentionally separate from /visualize/analysis.
    AI Analyst charts MUST use /visualize/analysis so the agent's precomputed
    result remains the source of truth. This route is for user-selected manual
    X/Y/aggregation exploration only.

    Request body:
    {
        "chart_type": "bar",
        "x_col": "Category",
        "y_col": "Profit_Margin",
        "aggregation": "auto"
    }

    aggregation can be:
        auto | sum | mean | count | min | max

    Response contains both requested and resolved aggregation so the frontend
    can display exactly what was used to build the chart.
    """
    df, error = get_current_df()

    if error:
        return jsonify({"error": error}), 400

    data = request.get_json(silent=True) or {}

    chart_type = str(data.get("chart_type", "bar")).strip()
    x_col = data.get("x_col")
    y_col = data.get("y_col")

    # New frontend-compatible field.
    # Defaults to AUTO so old frontend requests continue to work.
    requested_aggregation = str(
        data.get("aggregation", "auto")
    ).strip().lower()

    if not x_col or x_col not in df.columns:
        return jsonify({
            "error": "Invalid X column selection."
        }), 400

    if not y_col or y_col not in df.columns:
        return jsonify({
            "error": "Invalid Y column selection."
        }), 400

    if requested_aggregation not in SUPPORTED_AGGREGATIONS:
        return jsonify({
            "error": (
                f"Unsupported aggregation '{requested_aggregation}'. "
                "Use auto, sum, mean, count, min, or max."
            ),
            "supported_aggregations": sorted(SUPPORTED_AGGREGATIONS),
        }), 400

    try:
        # Bar/pie charts need category-level aggregation.
        # Other chart types can still use the same universal aggregation when
        # they are category-based. Line charts also benefit from aggregated data.
        aggregate_chart_types = {
            "bar",
            "pie",
            "line",
            "area",
            "scatter",
        }

        if chart_type.lower() in aggregate_chart_types:
            chart_data, actual_aggregation, decision_reason = (
                aggregate_for_visualization(
                    df=df,
                    x_col=x_col,
                    y_col=y_col,
                    aggregation=requested_aggregation,
                    top_n=15,
                )
            )
        else:
            # Preserve the original raw-data behavior for specialized chart
            # implementations that may expect row-level data.
            chart_data = df[[x_col, y_col]].copy()
            actual_aggregation = "none"
            decision_reason = (
                f"Raw row-level data passed to specialized '{chart_type}' chart."
            )

        # Critical safeguard:
        # DataVisualizer receives the EXACT selected y_col. It is never replaced
        # by Profit, Sales, or another default metric.
        chart_json = DataVisualizer.create_chart(
            chart_data,
            chart_type,
            x_col,
            y_col,
            title=data.get("title"),
        )

        return jsonify({
            "chart": chart_json,

            # Frontend-compatible aggregation metadata.
            "aggregation": actual_aggregation,
            "requested_aggregation": requested_aggregation,
            "aggregation_label": actual_aggregation.replace("_", " ").title(),
            "aggregation_reason": decision_reason,

            # Echo the exact columns used to generate the chart.
            "x_col": x_col,
            "y_col": y_col,
            "metric_label": y_col,

            # Useful for frontend debugging/verification.
            "chart_type": chart_type,
            "rows_used": int(len(chart_data)),
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": f"Visualization failed: {str(exc)}",
            "x_col": x_col,
            "y_col": y_col,
            "requested_aggregation": requested_aggregation,
        }), 500


@app.route("/fallback-log", methods=["GET"])
def fallback_log():
    """
    Admin endpoint:
    GET /fallback-log?n=50
    """
    try:
        n = int(request.args.get("n", 50))
    except ValueError:
        n = 50

    entries = get_fallback_log_entries(last_n=n)

    return jsonify({
        "total_logged": len(entries),
        "entries": entries,
        "log_path": "logs/fallback_queries.jsonl",
        "note": (
            "These are questions that had no Pattern A function match. "
            "Add functions for recurring patterns."
        ),
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accept a raw audio file from the browser MediaRecorder and return text.
    """
    if "audio" not in request.files:
        return jsonify({
            "text": None,
            "error": "No audio file in request.",
        }), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if not audio_bytes:
        return jsonify({
            "text": None,
            "error": "Empty audio file received.",
        }), 400

    mime_type = audio_file.content_type or "audio/webm"
    mime_type = mime_type.split(";")[0].strip()

    if not mime_type.startswith("audio/"):
        mime_type = "audio/webm"

    result = transcribe_audio(audio_bytes, mime_type)
    status_code = 200 if result["text"] else 422

    return jsonify(result), status_code


# -----------------------------------------------------------------------------
# Local development
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
