import sys
sys.path.insert(0, '.')
import json
import pandas as pd
from app import app, DATASET_CACHE

# Setup test client
client = app.test_client()

# Load a cached dataset for session
file_path = 'uploads/sales_data_sample-selected-columns.csv'
df = pd.read_csv(file_path)
from services.data_analyzer import DataAnalyzer
DATASET_CACHE[file_path] = {"df": df, "schema": DataAnalyzer(df).get_schema_summary()}

with client.session_transaction() as sess:
    sess['file_path'] = file_path
    sess['file_name'] = 'sales_data_sample-selected-columns.csv'

print("\n--- Sending request to /chat endpoint ---")
response = client.post('/chat', json={"question": "Calculate monthly total sales and percentage growth compared to previous month"})
print("Flask Response Status:", response.status_code)
data = response.get_json()
print("Flask Answer Preview:", data.get("answer")[:100] if data else "None")
