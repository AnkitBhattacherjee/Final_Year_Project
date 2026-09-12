import os
import pandas as pd

for filename in os.listdir('uploads'):
    if filename.endswith(('.csv', '.xlsx', '.xls')):
        path = os.path.join('uploads', filename)
        print(f"\n--- Dataset: {filename} ({os.path.getsize(path)/1024:.1f} KB) ---")
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(path, nrows=5000)
            else:
                df = pd.read_excel(path, nrows=5000)
            print("Columns:", df.columns.tolist()[:10])
            date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'year', 'month', 'day'])]
            num_cols = [c for c in df.columns if any(k in c.lower() for k in ['sale', 'revenue', 'total', 'price', 'amount'])]
            print("Date columns candidate:", date_cols)
            print("Sales/num columns candidate:", num_cols)
        except Exception as e:
            print("Error reading:", e)
