import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from services.analysis_functions import _fmt, _FREQ_LABELS

def enhanced_growth_rate(df, date_col, value_col, freq="ME", from_date=None, to_date=None, n_periods=None):
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])
    if ts.empty:
        return {"error": "No valid dates found."}
    
    ts = ts.set_index(date_col)
    grouped = ts[value_col].resample(freq).sum().dropna()
    
    if n_periods is not None and n_periods > 0:
        grouped = grouped.tail(int(n_periods))
    elif from_date is None and to_date is None:
        grouped = grouped.tail(12)  # Last 12 periods for readable table
        
    if len(grouped) < 2:
        return {"error": "Need at least 2 time periods to calculate growth rate."}
        
    growth = grouped.pct_change() * 100
    
    # Format periods
    if freq in ["ME", "M"]:
        period_format = "%Y-%m"
    elif freq in ["YE", "Y", "A"]:
        period_format = "%Y"
    elif freq in ["QE", "Q"]:
        period_format = "%Y-Q%q"
    else:
        period_format = "%Y-%m-%d"
        
    table_rows = []
    for idx, (dt, val) in enumerate(grouped.items()):
        dt_str = dt.strftime("%Y-%m") if freq in ["ME", "M"] else dt.strftime("%Y-%m-%d")
        g_val = growth.iloc[idx]
        if pd.isna(g_val):
            g_str = "—"
        else:
            sign = "+" if g_val > 0 else ""
            g_str = f"{sign}{g_val:.2f}%"
        table_rows.append(f"| {dt_str} | {_fmt(val)} | {g_str} |")
        
    avg_growth = round(float(growth.dropna().mean()), 2)
    latest_growth = round(float(growth.dropna().iloc[-1]), 2)
    latest_val = float(grouped.iloc[-1])
    prev_val = float(grouped.iloc[-2])
    
    prev_period = grouped.index[-2].strftime("%b %Y") if freq in ["ME", "M"] else str(grouped.index[-2])[:10]
    curr_period = grouped.index[-1].strftime("%b %Y") if freq in ["ME", "M"] else str(grouped.index[-1])[:10]
    
    sign = "+" if latest_growth > 0 else ""
    avg_sign = "+" if avg_growth > 0 else ""
    freq_label = _FREQ_LABELS.get(freq, "Period").title()
    
    table_md = "\n".join(table_rows)
    
    answer = (
        f"**{freq_label} Total {value_col} & Percentage Growth Rate:**\n\n"
        f"- **Latest ({curr_period})**: **{_fmt(latest_val)}** ({sign}{latest_growth:.2f}% vs {prev_period} of {_fmt(prev_val)})\n"
        f"- **Average {freq_label} Growth**: **{avg_sign}{avg_growth:.2f}%**\n\n"
        f"| Month | Total {value_col} | Growth % (vs Prev Month) |\n"
        f"| :--- | :--- | :--- |\n"
        f"{table_md}"
    )
    
    growth_df = pd.DataFrame({
        date_col: [dt.strftime("%Y-%m") if freq in ["ME", "M"] else dt.strftime("%Y-%m-%d") for dt in grouped.index],
        f"Total {value_col}": grouped.values,
        "Growth%": [0.0 if pd.isna(x) else round(x, 2) for x in growth.values]
    })
    
    return {"answer": answer, "growth_df": growth_df}

# Test on datasets
df1 = pd.read_csv('uploads/sales_data_sample-selected-columns.csv')
res1 = enhanced_growth_rate(df1, 'ORDERDATE', 'SALES', freq='ME')
print(res1['answer'])
print("\nDataFrame Preview:\n", res1['growth_df'].head())
