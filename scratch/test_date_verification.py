import sys
sys.path.insert(0, '.')
import pandas as pd

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=50)

date_cols = []
for c in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        date_cols.append(c)
    elif not pd.api.types.is_numeric_dtype(df[c]):
        if any(k in c.lower() for k in ["date", "time", "day", "year", "month", "period"]):
            # verify that at least some values can be parsed as timestamps
            try:
                sample = df[c].dropna().head(10)
                if not sample.empty:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() >= len(sample) * 0.5:
                        date_cols.append(c)
            except Exception:
                pass

print("Valid parsed date cols:", date_cols)
