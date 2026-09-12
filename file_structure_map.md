# 🗺️ Complete File Structure & Component Map: AI Data Analyst Web Application

This document provides a **complete, part-by-part breakdown of every file in the codebase**. It details what each file contains, what each internal section does, and where specific features live so anyone can easily locate and tweak code.

---

## 🗂️ High-Level Directory Overview

```
AI-Data-Analyst-Agent-web-application-main/
│
├── 📄 app.py                           # Flask server, route controller & in-memory cache
├── 📄 architecture_of_ai_data_analyst.md # System architecture & execution lifecycle reference
├── 📄 file_structure_map.md             # This file: Comprehensive file-by-file component map
├── 📄 requirements.txt                 # Python dependencies
├── 📄 Procfile                         # Cloud deployment entry point
├── 📄 .env                             # Environment variables & API keys
├── 📄 .env.example                     # Configuration template
├── 📄 README.md                        # Project introduction & setup instructions
│
├── 📁 services/                        # Backend intelligence, analysis & visualization core
│   ├── 📄 __init__.py                  # Service package exports
│   ├── 📄 ai_agent.py                  # Agent orchestrator, semantic classifier, local router & Pass 1/2
│   ├── 📄 analysis_functions.py        # Library of 20+ parameterized business & statistical functions
│   ├── 📄 visualizer.py                # Plotly chart generation & base64 buffer decoder
│   ├── 📄 data_analyzer.py             # Data health profiler, schema extractor & preview builder
│   ├── 📄 code_executor.py             # AST sandboxed Python executor & fallback logger
│   └── 📄 voice_transcriber.py         # Speech-to-text audio transcriber (Gemini Audio)
│
├── 📁 templates/                       # Frontend HTML
│   └── 📄 index.html                   # Jinja2 Single Page Application UI layout
│
├── 📁 static/                          # Frontend Assets
│   ├── 📁 css/
│   │   └── 📄 style.css                # SaaS design system, layout grid & glassmorphism theme
│   └── 📁 js/
│       └── 📄 script.js                # AJAX handlers, audio recorder, Plotly lifecycle & UI controller
│
├── 📁 logs/                            # Audit Logs
│   └── 📄 fallback_queries.jsonl       # Pattern B code generation fallback log
│
└── 📁 uploads/                         # Temporary storage for uploaded CSV / Excel datasets
```

---

## 📂 Detailed File-by-File Component Map

---

### 1. `app.py` (Main Flask Application)
* **File Path:** [`app.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/app.py)
* **Role:** Web server entry point, REST API routes, session management, and dataset caching.

| Section / Function | Lines (Approx.) | Description & Responsibilities |
| :--- | :--- | :--- |
| **Imports & App Config** | `1 – 25` | Loads environment variables (`.env`), sets up Flask, upload folder (`uploads/`), and max upload limit (100 MB). |
| **In-Memory Caching** | `26 – 64` | `DATASET_CACHE` dictionary stores parsed DataFrames and pre-computed schema strings. `get_current_df()` and `get_cached_schema()` retrieve cached data by session key. |
| `GET /` (`index`) | `69 – 72` | Serves the main Single Page Application from `templates/index.html`. |
| `POST /upload` (`upload_file`) | `74 – 115` | Validates `.csv`/`.xlsx` files, saves to `uploads/`, runs initial profiling via `DataAnalyzer`, builds schema cache, and returns summary metrics. |
| `GET /dataset` (`get_dataset_info`) | `117 – 130` | Returns first 15 rows preview, column names list, and numeric column lists. |
| `GET /profile` (`get_profile`) | `132 – 140` | Returns column-level profiling statistics (null counts, unique counts, min/max/mean). |
| `GET /quality` (`get_quality`) | `142 – 150` | Returns data quality audit metrics (missing rates, duplicate row counts). |
| `POST /chat` (`chat`) | `152 – 172` | Main chat endpoint: parses question JSON, initializes `DataAIAgent` with cached schema, runs analysis, and returns markdown answer + Plotly chart JSON. |
| `GET /suggestions` (`suggestions`) | `174 – 182` | Returns dynamic question suggestion chips based on dataset columns. |
| `GET /insights` (`insights`) | `184 – 192` | Returns automated rule-based business insights and health highlights. |
| `POST /visualize` (`visualize`) | `194 – 217` | Manual Visual Studio endpoint: creates custom charts based on selected X-axis, Y-axis, and chart type. |
| `GET /fallback-log` (`fallback_log`)| `219 – 234` | Admin endpoint: returns recent queries that triggered Pattern B code generation. |
| `POST /transcribe` (`transcribe`) | `236 – 262` | Receives raw audio from browser `MediaRecorder`, normalizes MIME type, and invokes `transcribe_audio()`. |
| `__main__` Server Runner | `264 – 265` | Runs Flask development server on `port=5000` with `debug=True`. |

---

### 2. `services/ai_agent.py` (AI Analyst Orchestration Engine)
* **File Path:** [`services/ai_agent.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/ai_agent.py)
* **Role:** Semantic column classifier, zero-API local regex router, Gemini Pass 1 function calling, Pass 2 narrative generation, and Pattern B sandboxed code generation.

| Section / Function | Lines (Approx.) | Description & Responsibilities |
| :--- | :--- | :--- |
| **Tool Schemas (`ANALYSIS_TOOLS`)** | `51 – 490` | JSON function definitions passed to Gemini Function Calling (parameters, types, descriptions for all 20+ analytical tools). |
| **Dispatch Table (`ANALYSIS_FUNCTIONS`)** | `491 – 523` | Dictionary mapping tool names (e.g., `"get_top_n"`) to python functions in `services/analysis_functions.py`. |
| `_is_indic_script(text)` | `525 – 538` | Detects Devanagari (Hindi) or Bengali unicode characters to preserve language in narrative responses. |
| `_BoundedDict` & `_ANSWER_CACHE` | `540 – 560` | LRU in-memory cache (max 256 items) storing previous question answers to return repeated queries instantly. |
| `DataAIAgent.__init__()` | `583 – 618` | Initializes agent, checks `GEMINI_API_KEY`, prepares Gemini GenerativeModel, and invokes column classification. |
| `_build_col_roles()` | `622 – 677` | Semantic Column Classifier: filters out IDs/keys and identifies business roles (`sales`, `quantity`, `discount`, `cost`, `profit`, `delivery`, `product`, `category`, `region`, `customer`, `date_col`, etc.). |
| `_local_intent_router()` | `679 – 1225` | **25+ Regex Intent Pattern Groups** running zero-API analysis: |
| ↳ *Pattern 0a–0f* | `745 – 896` | Financial & Sales KPIs (Totals, Growth rate YoY/MoM/QoQ, Seasonality, Profit margins, Inventory value/low stock). |
| ↳ *Pattern 1–4* | `897 – 927` | Percentiles, Median, Threshold exceedance, and IQR Outliers. |
| ↳ *Pattern 5–8* | `928 – 1014` | Top N / Bottom N rankings, Distributions, Category breakdowns, Treemaps, Waterfalls, Grouped Bars. |
| ↳ *Pattern 9 & 22* | `1015 – 1037, 1130 – 1148` | Correlation, Impact, Cause-and-Effect, and Scatter Plots with 2-variable coordinate extraction. |
| ↳ *Pattern 10–15* | `1038 – 1076` | Missing values, Duplicates, Data overview, Summary statistics, Item search, Date bounds. |
| ↳ *Pattern 16–21* | `1077 – 1128` | Pareto 80/20, Conversion Funnel, Rolling Averages, Bubble Charts, Cohort Retention, KPI Gauges vs Target. |
| ↳ *Pattern 24* | `1157 – 1178` | Customer metrics (top spenders, spend rankings). |
| `_execute_function()` | `1227 – 1245` | Executes the selected analytical function on the full DataFrame. |
| `_pass1_function_call()` | `1247 – 1298` | Calls Gemini API to select a function when no local regex pattern matches. |
| `_pass2_generate_answer()` | `1300 – 1331` | Generates a 1-2 sentence business summary in the user's matching language (English, Hindi, Bengali). |
| `_pattern_b_fallback()` | `1333 – 1386` | Prompts Gemini to write raw Pandas code, runs it in `code_executor.py`, and logs the outcome. |
| `_local_heuristic_fallback()` | `1388 – 1481` | Resilient zero-API fallback executed if Gemini API quota (429) is exceeded. |
| `_build_chart()` | `1483 – 1519` | Dispatches chart generation to `DataVisualizer.create_chart()` or extracts pre-built `chart_json`. |
| `ask()` | `1521 – 1680` | **Primary Entry Point:** Cache check -> Fast paths -> Local router -> Pass 1 -> Execution -> Chart -> Pass 2. |
| `generate_suggested_questions()`| `1682 – 1740` | Generates 5 dynamic dataset-specific questions for frontend suggestion chips. |
| `generate_automated_insights()` | `1742 – 1811` | Generates 4 automated rule-based business insights from the dataset. |

---

### 3. `services/analysis_functions.py` (Analytical Calculation Library)
* **File Path:** [`services/analysis_functions.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/analysis_functions.py)
* **Role:** 20+ deterministic Pandas analytical functions operating against the full dataset in memory.

| Section / Function | Lines (Approx.) | Purpose & Output |
| :--- | :--- | :--- |
| **Helper Utilities** | `34 – 96` | `_result()` (builds standardized envelope), `_error()`, `_fmt()` (smart number/currency formatter), `_validate_cols()`, `_safe_agg()`. |
| `fn_get_missing_values()` | `100 – 125` | Checks missing/null cells per column; returns health summary. |
| `fn_get_duplicates()` | `127 – 148` | Detects and counts exact duplicate rows in the dataset. |
| `fn_get_data_overview()` | `150 – 180` | High-level dataset summary (row count, column count, memory size, sample columns). |
| `fn_get_top_n()` | `185 – 235` | Computes Top N or Bottom N entities by metric; calculates gap between highest and lowest; returns Bar Chart. |
| `fn_get_category_breakdown()` | `237 – 270` | Calculates percentage share of total for categories; returns Donut/Bar Chart. |
| `fn_get_distribution()` | `272 – 311` | Computes value counts and frequency distributions for categorical or discrete numeric fields. |
| `fn_get_filtered_summary()` | `314 – 352` | Filters dataset where `col == value` and aggregates target metric (e.g. sales in East region). |
| `fn_get_describe()` | `354 – 377` | Computes mean, std, min, max, and quartiles for numeric columns. |
| `fn_get_correlation()` | `383 – 450` | Pearson correlation $r$. For 2 variables: downsamples to 1,000 points and returns **Scatter Plot** (`chart_type="scatter"`). For $>2$ variables: returns **Correlation Heatmap**. |
| `fn_get_outliers()` | `452 – 490` | Uses IQR rule ($Q1 - 1.5 \times IQR$, $Q3 + 1.5 \times IQR$) to find outliers; returns Box Plot. |
| `fn_get_single_percentile()` | `492 – 515` | Computes exact N-th percentile value (e.g. 90th percentile of order value). |
| `fn_get_median()` | `517 – 535` | Computes 50th percentile (median) of numeric column. |
| `fn_get_pct_above_threshold()` | `537 – 560` | Computes count and percentage of records exceeding a numeric value. |
| `fn_get_profit_margin()` | `610 – 660` | Gross & operating profit margin percentages grouped by category. |
| `fn_get_customer_metrics()` | `662 – 710` | Total revenue per customer, top spending customers, and spending gap. |
| `fn_get_inventory_value()` | `712 – 740` | Total inventory valuation ($Quantity \times Cost$). |
| `fn_get_low_stock()` | `742 – 770` | Items with inventory quantity below threshold. |
| `fn_get_date_bounds()` | `780 – 815` | Earliest and latest transaction timestamps. |
| `fn_get_time_series()` | `820 – 870` | Aggregates metric over time (daily, weekly, monthly, yearly); returns Line/Area Chart. |
| `fn_get_growth_rate()` | `872 – 930` | Period-over-period growth rate (YoY, MoM, QoQ); returns Growth Bar Chart. |
| `fn_get_monthly_pattern()` | `932 – 975` | Monthly seasonality pattern across years; returns Seasonal Bar Chart. |
| `fn_get_pareto()` | `980 – 1030` | 80/20 cumulative contribution curve; returns Pareto Combo Chart. |
| `fn_get_treemap()` | `1032 – 1070`| Hierarchical nested category shares; returns Treemap. |
| `fn_get_waterfall()` | `1072 – 1115`| Revenue / Cost variance bridges; returns Waterfall Chart. |
| `fn_get_heatmap_cross()` | `1120 – 1165`| 2-dimensional cross-tabulation frequency matrix; returns 2D Heatmap. |
| `fn_get_cohort_retention()`| `1170 – 1220`| Customer monthly cohort retention rates; returns Cohort Heatmap. |
| `fn_get_gauge_vs_target()` | `1222 – 1260`| Current value vs target KPI achievement; returns Gauge Chart. |

---

### 4. `services/visualizer.py` (Plotly Visualization Engine)
* **File Path:** [`services/visualizer.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/visualizer.py)
* **Role:** Generates Plotly charts and cleans serialized JSON data.

| Section / Function | Lines (Approx.) | Description & Responsibilities |
| :--- | :--- | :--- |
| **Theme & Layout Config** | `16 – 33` | `_COLORS` (14 curated theme hex colors), `_FONT` (Inter typography), `_LAYOUT_BASE` (transparent background, white canvas, clean margins). |
| `_apply_layout()` | `35 – 40` | Injects theme fonts, margins, and title styling into Plotly figures. |
| `_decode_bdata()` | `42 – 61` | **Base64 Buffer Decoder:** Recursively decodes binary `bdata` buffers and numpy arrays into standard native Python lists/floats so frontend Plotly.js renders scatter plots and points visibly. |
| `_to_json()` | `63 – 66` | Converts Plotly figure to clean, browser-ready JSON specification. |
| `DataVisualizer.create_chart()` | `68 – 163` | Dispatches chart generation for `bar`, `hbar`, `grouped_bar`, `stacked_bar`, `line`, `area`, `pie`, `donut`, `scatter`, `scatter_trend`, `bubble`, `histogram`, `box`, `funnel`. |
| `build_pareto()` | `168 – 195` | Creates dual-axis Pareto chart (bar values on left axis, cumulative % line on right axis). |
| `build_heatmap()` | `197 – 220` | Builds interactive correlation matrix heatmap with diverging colorscale. |
| `build_treemap()` | `222 – 245` | Builds hierarchical nested treemap. |
| `build_waterfall()` | `247 – 275` | Builds positive/negative revenue & cost bridge waterfall charts. |
| `build_gauge()` | `277 – 305` | Builds radial KPI gauge with target indicators and color bands. |
| `build_radar()` | `307 – 330` | Builds closed polar radar charts for multi-metric comparison. |
| `build_cohort_heatmap()` | `332 – 360` | Builds cohort retention grid with percentage annotations. |

---

### 5. `services/data_analyzer.py` (Dataset Profiling & Schema Extraction)
* **File Path:** [`services/data_analyzer.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/data_analyzer.py)
* **Role:** Profiling tabular data during upload, generating token-efficient schema summaries, and column health checks.

| Function | Description & Responsibilities |
| :--- | :--- |
| `__init__(df)` | Stores raw DataFrame and computes column metadata. |
| `get_summary_metrics()` | Calculates total rows, columns, duplicate rows, missing cell counts, and memory footprint in KB/MB. |
| `get_schema_summary()` | Builds a compact text schema (column names, types, non-null counts, sample values) cached once at upload time. |
| `get_preview(n=15)` | Returns sanitized dictionary of the first 15 rows for the preview table. |
| `get_column_profiles()` | Computes per-column health profiles: data types, distinct count, null count, min, max, mean, top values. |
| `get_quality_report()` | Audits dataset quality and generates warnings for missing values or single-value columns. |
| `safe_aggregate()` | Pre-aggregates large datasets for manual chart studio rendering. |

---

### 6. `services/code_executor.py` (Safe Sandboxed Code Execution)
* **File Path:** [`services/code_executor.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/code_executor.py)
* **Role:** Sandboxed execution engine for Pattern B Gemini-generated Python snippets and fallback audit logging.

| Function / Class | Description & Responsibilities |
| :--- | :--- |
| `_SecurityVisitor` | AST (Abstract Syntax Tree) scanner that blocks dangerous calls (`import`, `open`, `eval`, `exec`, `os`, `sys`, `subprocess`, `__`). |
| `execute_sandboxed()` | Runs Python code within a restricted global scope (`df`, `pd`, `np`) on a read-only copy of the DataFrame with an 8-second execution timeout. |
| `log_fallback_query()` | Records question, generated code, execution status, and errors into `logs/fallback_queries.jsonl`. |
| `get_fallback_log_entries()` | Reads the last $N$ entries from the fallback log for admin auditing. |

---

### 7. `services/voice_transcriber.py` (Speech-to-Text Transcription)
* **File Path:** [`services/voice_transcriber.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/voice_transcriber.py)
* **Role:** Multilingual voice query transcription using Gemini's native audio understanding.

| Function | Description & Responsibilities |
| :--- | :--- |
| `transcribe_audio()` | Accepts audio byte buffer and MIME type (`audio/webm`), sends payload to Gemini Multimodal Audio model, and returns clean transcribed text across English, Hindi, and Bengali. |

---

### 8. `templates/index.html` (Frontend SPA Layout)
* **File Path:** [`templates/index.html`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/templates/index.html)
* **Role:** Complete Single Page Application interface template.

| DOM Section / ID | Description & Responsibilities |
| :--- | :--- |
| **Top Navbar & Upload Dropzone** | Header containing project title, dataset name badge, and file upload dropzone. |
| **Executive KPI Grid** | 4 summary metric cards: `#kpi-rows`, `#kpi-cols`, `#kpi-duplicates`, `#kpi-missing`. |
| **Workspace Tab Navigation** | Tab switch buttons: Chat Analyst (`#tab-chat`), Data Preview (`#tab-preview`), Quality Report (`#tab-quality`), Visual Studio (`#tab-visualizer`). |
| `#tab-chat` (Chat Pane) | Conversation feed (`#chat-messages`), suggestion chip container (`#suggested-chips`), query input box (`#chat-input`), send button, and voice record button (`#voice-btn`). |
| `#tab-preview` (Data Table) | Scrollable interactive table displaying dataset records. |
| `#tab-quality` (Health Cards) | Column health breakdown bars and data quality audits. |
| `#tab-visualizer` (Chart Studio) | Dropdowns to select chart type, X column, Y column, and "Generate Visualization" button. |
| `#plotly-chart-container` | Dedicated Plotly chart viewport where all charts are rendered. |

---

### 9. `static/js/script.js` (Frontend Controller & Plotly Engine)
* **File Path:** [`static/js/script.js`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/static/js/script.js)
* **Role:** Manages client-side state, AJAX requests, voice recording, chat rendering, and Plotly lifecycle.

| Section / Function | Description & Responsibilities |
| :--- | :--- |
| **State & Initialization** | Initializes active dataset metadata, chat state, and tab listeners. |
| `switchTab(tabId)` | Switches active tab and triggers `Plotly.Plots.resize()` so charts adjust properly. |
| `handleFileUpload(file)` | Sends file via `fetch("/upload")`, renders KPI cards, loads suggestion chips, and opens preview table. |
| `sendMessage(text)` | Appends user bubble, displays typing indicator, calls `fetch("/chat")`, parses markdown response, and renders Plotly chart via `Plotly.newPlot()`. |
| `initVoiceRecording()` | Connects to browser `navigator.mediaDevices.getUserMedia`, handles recording state, sends audio blob to `fetch("/transcribe")`, and places text into chat input. |
| `loadSuggestions()` | Fetches dynamic question chips from `GET /suggestions` and attaches click-to-ask handlers. |
| `renderVisualStudio()` | Handles manual chart generation via `fetch("/visualize")` and renders output in Plotly container. |

---

### 10. `static/css/style.css` (Visual Design System)
* **File Path:** [`static/css/style.css`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/static/css/style.css)
* **Role:** Modern light SaaS design system, CSS variables, glassmorphism, responsive grids, and animations.

| CSS Section | Description & Responsibilities |
| :--- | :--- |
| `:root` Variables | Color tokens (`--primary: #2563EB`, `--bg: #F8FAFC`, `--text: #334155`, `--border: #E2E8F0`). |
| Layout & Grid System | Flex and CSS grid layout for header, KPI cards, sidebar, and chart viewport. |
| Chat Component Styles | Chat message bubbles (user vs AI), markdown formatting, bold text highlights, code blocks. |
| Audio Recording Pulse | Micro-animations and glowing red pulse effects for active microphone recording. |
| Chart Container Styling | Responsive sizing and centering for `#plotly-chart-container`. |
| Responsive Breakpoints | Mobile and tablet media queries (`max-width: 768px` and `max-width: 1024px`). |

---

## 🎯 Quick Navigation: Where Do I Change...?

| To Change / Add... | Edit This File | Specific Function or Section |
| :--- | :--- | :--- |
| **Add a new business KPI calculation** | [`services/analysis_functions.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/analysis_functions.py) | Add `fn_new_metric(df, ...)` & return `_result(...)` |
| **Add a new question regex pattern** | [`services/ai_agent.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/ai_agent.py) | Add pattern block in `_local_intent_router()` |
| **Add a new chart type** | [`services/visualizer.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/visualizer.py) | Add `elif ct == "..."` in `create_chart()` |
| **Add or edit REST API endpoints** | [`app.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/app.py) | Add `@app.route(...)` |
| **Change UI colors or theme** | [`static/css/style.css`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/static/css/style.css) | Edit `:root` color tokens |
| **Change UI HTML layout or tabs** | [`templates/index.html`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/templates/index.html) | Add/modify HTML section |
| **Adjust Gemini prompt or model** | [`services/ai_agent.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/ai_agent.py) | Edit prompt strings in `_pass1_function_call()` or `_pass2_generate_answer()` |
| **Change speech recognition settings**| [`services/voice_transcriber.py`](file:///c:/Users/ANKIT/Downloads/AI-Data-Analyst-Agent-web-application-main/services/voice_transcriber.py) | Edit `transcribe_audio()` prompt |
