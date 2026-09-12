import sys
sys.path.insert(0, '.')
import pandas as pd
from services.analysis_functions import _fmt, _FREQ_LABELS, _validate_cols, _parse_date_bound, _result, _error
from typing import Optional, Dict

def fn_get_growth_rate(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    n_periods: Optional[int] = None,
) -> Dict:
    err = _validate_cols(df, date_col, value_col)
    if err:
        return _error(err)
    freq = freq if freq in _FREQ_LABELS else "ME"
    try:
        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col])

        # Apply user-specified date filters
        start_ts = _parse_date_bound(from_date, end=False)
        end_ts = _parse_date_bound(to_date, end=True)
        if start_ts is not None:
            ts = ts[ts[date_col] >= start_ts]
        if end_ts is not None:
            ts = ts[ts[date_col] <= end_ts]

        if ts.empty:
            return _error("No data found in the specified date range.")

        ts = ts.set_index(date_col)
        grouped = ts[value_col].resample(freq).sum().dropna()

        if n_periods is not None and n_periods > 0:
            grouped = grouped.tail(int(n_periods))
        elif from_date is None and to_date is None:
            grouped = grouped.tail(12)

        if len(grouped) < 2:
            return _error("Need at least 2 time periods to calculate growth rate.")

        growth = grouped.pct_change() * 100
        avg_growth = round(float(growth.dropna().mean()), 2)
        latest_growth = round(float(growth.dropna().iloc[-1]), 2)
        latest_val = float(grouped.iloc[-1])
        prev_val = float(grouped.iloc[-2])

        prev_period = grouped.index[-2].strftime("%b %Y") if freq in ["ME", "M"] else str(grouped.index[-2])[:10]
        curr_period = grouped.index[-1].strftime("%b %Y") if freq in ["ME", "M"] else str(grouped.index[-1])[:10]

        sign = "+" if latest_growth > 0 else ""
        avg_sign = "+" if avg_growth > 0 else ""
        range_label = f" ({from_date or ''}-{to_date or ''})" if (from_date or to_date) else ""
        freq_label = _FREQ_LABELS[freq].title()

        table_rows = []
        for idx, (dt, val) in enumerate(grouped.items()):
            dt_str = dt.strftime("%Y-%m") if freq in ["ME", "M"] else dt.strftime("%Y-%m-%d")
            g_val = growth.iloc[idx]
            if pd.isna(g_val):
                g_str = "—"
            else:
                s = "+" if g_val > 0 else ""
                g_str = f"{s}{g_val:.2f}%"
            table_rows.append(f"| {dt_str} | {_fmt(val)} | {g_str} |")

        table_md = "\n".join(table_rows)

        answer = (
            f"**{freq_label} Total {value_col} & Growth Rate**{range_label}:\n\n"
            f"- **Latest Period ({curr_period})**: Total **{_fmt(latest_val)}** (**{sign}{latest_growth:.2f}%** vs {prev_period} of {_fmt(prev_val)})\n"
            f"- **Average {freq_label} Growth**: **{avg_sign}{avg_growth:.2f}%**\n\n"
            f"| Period | Total {value_col} | MoM Growth % |\n"
            f"| :--- | :--- | :--- |\n"
            f"{table_md}"
        )

        growth_df = pd.DataFrame({
            date_col: [dt.strftime("%Y-%m") if freq in ["ME", "M"] else dt.strftime("%Y-%m-%d") for dt in grouped.index],
            f"Total {value_col}": grouped.values,
            "Growth%": [0.0 if pd.isna(x) else round(float(x), 2) for x in growth.values]
        })

        chart_type = "line" if len(growth_df) > 1 else None
        return _result(
            answer,
            summary=f"{freq_label} Growth Rate: {value_col} ({sign}{latest_growth:.2f}% latest, {avg_sign}{avg_growth:.2f}% avg)",
            result_df=growth_df if chart_type else None,
            chart_type=chart_type,
            x_col=date_col if chart_type else None,
            y_col="Growth%" if chart_type else None,
            chart_title=f"Period-over-Period Growth: {value_col} ({_FREQ_LABELS[freq]}){range_label}" if chart_type else None,
        )
    except Exception as e:
        return _error(f"Growth rate error: {str(e)}")

# Test on Cleaned_data.csv and restaurant_sales_data.csv
df_rest = pd.read_csv('uploads/restaurant_sales_data.csv')
print("Restaurant data growth:")
res = fn_get_growth_rate(df_rest, 'Order Date', 'Order Total', freq='ME')
print(res['answer'])
