import sys
sys.path.insert(0, '.')
import pandas as pd

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=20)
for col in ['Order date (DateOrders)', 'Shipping date (DateOrders)', 'Order date']:
    if col in df.columns:
        print(f"Col '{col}': sample values -> {df[col].tolist()[:5]}")
        parsed = pd.to_datetime(df[col], errors='coerce')
        print(f"Parsed non-null count: {parsed.notna().sum()}/{len(df)}")
