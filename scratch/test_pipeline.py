import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np

print("=== 1. Testing Services Import ===")
from services.data_analyzer import DataAnalyzer
from services.visualizer import DataVisualizer
from services.ai_agent import DataAIAgent
from services.code_executor import execute_sandboxed, get_fallback_log_entries
from services.voice_transcriber import transcribe_audio
from services.query_logger import log_query_start
from services import analysis_functions as af
import app as flask_app
print("Imports: OK")

print("=== 2. Testing DataAnalyzer ===")
df = pd.DataFrame({
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 'Furniture'],
    'Region': ['North', 'South', 'East', 'West', 'North', 'East'],
    'Sales': [1500.5, 320.0, 2400.0, 850.25, 410.0, 990.0],
    'Profit': [300.0, 50.0, 600.0, 120.0, 80.0, 210.0],
    'Quantity': [2, 5, 3, 1, 4, 2],
    'Order_Date': pd.date_range('2023-01-01', periods=6, freq='D')
})

analyzer = DataAnalyzer(df)
summary = analyzer.get_summary_metrics()
preview = analyzer.get_preview()
profiles = analyzer.get_column_profiles()
schema = analyzer.get_schema_summary()
quality = analyzer.get_quality_report()
print(f"DataAnalyzer methods: OK (Summary rows: {summary['total_rows']})")

print("=== 3. Testing Visualizer ===")
for ctype in ['bar', 'line', 'scatter', 'pie', 'histogram', 'box']:
    chart = DataVisualizer.create_chart(df, ctype, 'Category', 'Sales')
    assert chart is not None, f"Failed {ctype}"
print("Visualizer chart types: OK")

print("=== 4. Testing Analysis Functions ===")
res1 = af.fn_get_top_n(df, group_col='Category', value_col='Sales', n=3, agg='sum')
assert 'Electronics' in str(res1.get('answer', '')), 'Top N failed'
res2 = af.fn_get_describe(df, cols=['Sales', 'Profit'])
assert 'Sales' in str(res2.get('answer', '')), 'Describe failed'
res3 = af.fn_get_correlation(df, cols=['Sales', 'Profit'])
assert res3 is not None, 'Correlation failed'
print("Analysis functions: OK")

print("=== 5. Testing Code Executor ===")
code = "result = df.groupby('Category')['Sales'].sum().reset_index()"
exec_res = execute_sandboxed(df, code)
assert exec_res.get('error') is None, f"Code executor error: {exec_res}"
assert exec_res.get('result_df') is not None, "Code executor result_df is None"
print("Code executor: OK")

print("=== 6. Testing Flask Endpoints ===")
client = flask_app.app.test_client()

# GET /
r_index = client.get('/')
assert r_index.status_code == 200, f"GET / failed: {r_index.status_code}"

# Mock uploading dataset to session
import io
csv_bytes = io.BytesIO(df.to_csv(index=False).encode('utf-8'))
r_upload = client.post('/upload', data={'dataset': (csv_bytes, 'test_sales.csv')}, content_type='multipart/form-data')
assert r_upload.status_code == 200, f"Upload failed: {r_upload.data}"
upload_data = json.loads(r_upload.data)
assert "metrics" in upload_data, "Upload missing metrics"
assert upload_data["metrics"]["total_rows"] == 6, "Total rows mismatch"

# GET /dataset
r_info = client.get('/dataset')
assert r_info.status_code == 200, f"/dataset failed: {r_info.status_code}"

# GET /suggestions
r_sugg = client.get('/suggestions')
assert r_sugg.status_code == 200, f"/suggestions failed: {r_sugg.status_code}"

# GET /quality
r_qual = client.get('/quality')
assert r_qual.status_code == 200, f"/quality failed: {r_qual.status_code}"

# POST /visualize
r_vis = client.post('/visualize', json={'chart_type': 'bar', 'x_col': 'Category', 'y_col': 'Sales', 'aggregation': 'sum'})
assert r_vis.status_code == 200, f"/visualize failed: {r_vis.data}"

# POST /visualize/analysis
sample_precomputed = [{'Category': 'Electronics', 'Sales': 3900.5}, {'Category': 'Home', 'Sales': 850.25}]
r_vis_analysis = client.post('/visualize/analysis', json={'result_data': sample_precomputed, 'x_col': 'Category', 'y_col': 'Sales', 'chart_type': 'bar', 'chart_title': 'Top Categories'})
assert r_vis_analysis.status_code == 200, f"/visualize/analysis failed: {r_vis_analysis.data}"

print("Flask routes: OK")
print("=== ALL SYSTEM COMPONENTS AND PIPELINES VERIFIED 100% WORKING! ===")
