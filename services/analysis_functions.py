"""
Pattern A — Pre-written, Parameterized Analysis Function Library
================================================================
Design rules:
  - Every function accepts explicit column names + parameters (no hardcoded assumptions)
  - Runs against the FULL DataFrame locally (zero API cost)
  - Returns an AnalysisResult dict:
        answer       →  template-formatted business answer (no 2nd LLM call needed)
        summary      →  compact data string sent to LLM only when needs_llm=True
        result_df    →  DataFrame for chart rendering (or None)
        chart_*      →  Plotly chart metadata
        needs_llm    →  True = result is complex enough to benefit from LLM narrative
  - Every function is independently testable

Functions (20 total):
  Data Quality   : get_missing_values, get_duplicates, get_data_overview
  Ranking        : get_top_n, get_category_breakdown, get_distribution,
                   get_filtered_summary, get_describe
  Statistical    : get_correlation, get_outliers, compare_segments, get_percentile_breakdown
  Financial      : get_profit_margin, get_inventory_value, get_low_stock
  Time-based     : get_time_series, get_date_range_summary, get_growth_rate, get_monthly_pattern
  Search         : search_items
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _result(
    answer: str,
    *,
    summary: Optional[str] = None,
    result_df: Optional[pd.DataFrame] = None,
    chart_type: Optional[str] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    chart_title: Optional[str] = None,
    chart_kwargs: Optional[Dict[str, Any]] = None,
    chart_json: Optional[Dict[str, Any]] = None,
    needs_llm: bool = False,
) -> Dict[str, Any]:
    """
    Standard result envelope for all analysis functions.

    chart_json  : Pre-built Plotly JSON dict (for Pareto, Treemap, Waterfall,
                  Gauge, Radar, Combo, Cohort). When set, _build_chart() uses it
                  directly and skips create_chart().
    chart_kwargs: Extra kwargs passed to DataVisualizer.create_chart() when
                  chart_type is set (e.g. color_col, size_col, barmode).
    """
    return {
        "answer": answer,
        "summary": summary or answer,
        "result_df": result_df,
        "chart_type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "chart_title": chart_title,
        "chart_kwargs": chart_kwargs or {},
        "chart_json": chart_json,
        "needs_llm": needs_llm,
    }

def _error(msg: str) -> Dict[str, Any]:
    return _result(f"\u26a0\ufe0f {msg}")

def _fmt(val) -> str:
    """Format a number cleanly for display in template answers."""
    if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
        return str(val)
    if isinstance(val, float):
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"{val:,.2f}"
        return f"{val:.2f}"
    return f"{val:,}"

def _validate_cols(df: pd.DataFrame, *cols) -> Optional[str]:
    """Returns None if all cols exist, else an error string."""
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        available = ", ".join(df.columns[:10])
        return f"Column(s) not found: {', '.join(missing)}. Available: {available}"
    return None

def _safe_agg(series: pd.Series, agg: str) -> float:
    agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"
    return float(getattr(series, agg)())


# ===========================================================================
# Category 1 — Data Quality
# ===========================================================================

def fn_get_missing_values(df: pd.DataFrame) -> Dict:
    """Count missing/null values per column and report as % of total."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return _result("No missing values — dataset is 100% complete.")
    total = int(missing.sum())
    pct = round(total / df.size * 100, 2)
    lines = [
        f"- **{col}**: {cnt:,} missing ({round(cnt/len(df)*100,1)}%)"
        for col, cnt in missing.items()
    ]
    answer = (
        f"Found **{total:,} missing values** ({pct}% of dataset) "
        f"across **{len(missing)} column(s)**:\n" + "\n".join(lines)
    )
    return _result(answer, summary=missing.to_string())


def fn_get_duplicates(df: pd.DataFrame) -> Dict:
    """Count exact duplicate rows in the dataset."""
    n = int(df.duplicated().sum())
    if n == 0:
        return _result("No duplicate rows detected — all records are unique.")
    pct = round(n / len(df) * 100, 2)
    return _result(
        f"Found **{n:,} duplicate rows** ({pct}% of dataset). "
        "Remove these before aggregating to avoid double-counting."
    )


def fn_get_data_overview(df: pd.DataFrame) -> Dict:
    """High-level dataset summary: shape, column types, completeness, duplicates."""
    rows, cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    missing_pct = round(df.isna().sum().sum() / df.size * 100, 1) if df.size > 0 else 0
    dups = int(df.duplicated().sum())

    answer = (
        f"**Dataset Overview — {rows:,} rows x {cols} columns**\n"
        f"- Numeric ({len(num_cols)}): {', '.join(num_cols[:6])}{'...' if len(num_cols)>6 else ''}\n"
        f"- Categorical ({len(cat_cols)}): {', '.join(cat_cols[:6])}{'...' if len(cat_cols)>6 else ''}\n"
        + (f"- Date ({len(date_cols)}): {', '.join(date_cols)}\n" if date_cols else "")
        + f"- Missing data: **{missing_pct}%**  |  Duplicate rows: **{dups:,}**"
    )
    return _result(answer)


# ===========================================================================
# Category 2 — Ranking & Aggregation
# ===========================================================================

def fn_get_top_n(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n: int = 10,
    agg: str = "sum",
    ascending: bool = False,
    filter_col: Optional[str] = None,
    filter_val: Optional[str] = None,
) -> Dict:
    """
    Top / Bottom N items by aggregated value with performance gap analysis.
    Use for: 'which X has highest Y', 'rank products by revenue', 'best/worst performers',
    'gap between top and bottom performers', filtered ranking ('top product in Consumer segment').
    """
    if filter_col and filter_val:
        if filter_col in df.columns:
            filtered_df = df[df[filter_col].astype(str).str.strip().str.lower() == str(filter_val).strip().lower()]
            if not filtered_df.empty:
                df = filtered_df

    err = _validate_cols(df, group_col, value_col)
    if err:
        return _error(err)

    n = min(int(n) if n else 10, 20)
    agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"

    result = (
        df.groupby(group_col, as_index=False)[value_col]
        .agg(agg)
        .sort_values(value_col, ascending=bool(ascending))
        .head(n)
    )
    if result.empty:
        return _error("No data to aggregate.")

    direction = "Bottom" if ascending else "Top"
    perf_label = "Lowest Performer" if ascending else "Top Performer"
    filter_label = f" (for {filter_val})" if filter_val else ""
    top_name = str(result.iloc[0][group_col])
    top_val = float(result.iloc[0][value_col])
    last_name = str(result.iloc[-1][group_col])
    last_val = float(result.iloc[-1][value_col])
    total = float(df[value_col].sum()) if agg == "sum" and pd.api.types.is_numeric_dtype(df[value_col]) else None

    lines = []
    if ascending:
        lines.append(f"**{perf_label}{filter_label}:** **{top_name}** has the lowest {value_col} with **{_fmt(top_val)}**.")
    elif total and total > 0:
        pct = round(top_val / total * 100, 1)
        lines.append(f"**Top Performer{filter_label}:** **{top_name}** leads with **{_fmt(top_val)}** {value_col} (**{pct}%** of total).")
    else:
        lines.append(f"**Top Performer{filter_label}:** **{top_name}** ranks 1st with **{_fmt(top_val)}** ({agg} of {value_col}).")

    if len(result) > 1:
        items_summary = "\n".join([
            f"{i+1}. **{str(r[group_col])}**: {_fmt(float(r[value_col]))}" + (f" ({round(float(r[value_col])/total*100, 1)}% of total)" if total and total > 0 else "")
            for i, (_, r) in enumerate(result.iterrows())
        ])
        lines.append(f"\n**{direction} {len(result)} Rankings:**\n{items_summary}")

        gap = abs(top_val - last_val)
        pct_diff = round((top_val - last_val) / top_val * 100, 1) if top_val > 0 else 0
        ratio = round(top_val / last_val, 1) if last_val > 0 else None

        if ratio and ratio > 1:
            lines.append(
                f"\n**Performance Gap:** The gap between #{1} (**{top_name}**: {_fmt(top_val)}) "
                f"and #{len(result)} (**{last_name}**: {_fmt(last_val)}) is **{_fmt(gap)}** "
                f"({pct_diff}% drop, **{ratio}x** difference)."
            )
        else:
            lines.append(
                f"\n**Performance Gap:** The difference between #{1} (**{top_name}**) "
                f"and #{len(result)} (**{last_name}**) is **{_fmt(gap)}**."
            )

    answer = "\n".join(lines)

    # Only render a chart when there is more than one bar to compare
    chart_type = "bar" if len(result) > 1 else None
    return _result(
        answer,
        summary=result.to_string(index=False),
        result_df=result if chart_type else None,
        chart_type=chart_type,
        x_col=group_col if chart_type else None,
        y_col=value_col if chart_type else None,
        chart_title=f"{direction} {len(result)} {group_col} by {agg.title()} {value_col}" if chart_type else None,
    )


def fn_get_category_breakdown(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg: str = "sum",
) -> Dict:
    """Full breakdown of a metric across all categories (not just top N)."""
    err = _validate_cols(df, group_col, value_col)
    if err:
        return _error(err)

    agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"
    result = (
        df.groupby(group_col, as_index=False)[value_col]
        .agg(agg)
        .sort_values(value_col, ascending=False)
    )
    total = float(result[value_col].sum()) if agg == "sum" else None
    n_cats = len(result)

    answer = f"**{n_cats} {group_col}** categories ranked by {agg} of {value_col}:\n"
    for _, row in result.head(6).iterrows():
        share = (
            f" ({round(float(row[value_col])/total*100,1)}%)"
            if total and total > 0 else ""
        )
        answer += f"- **{row[group_col]}**: {_fmt(float(row[value_col]))}{share}\n"
    if n_cats > 6:
        answer += f"  *(and {n_cats-6} more categories)*"

    # Only render chart when there are multiple categories to compare
    chart_type = "bar" if len(result) > 1 else None
    return _result(
        answer.strip(),
        summary=result.to_string(index=False),
        result_df=result if chart_type else None,
        chart_type=chart_type,
        x_col=group_col if chart_type else None,
        y_col=value_col if chart_type else None,
        chart_title=f"{agg.title()} of {value_col} by {group_col}" if chart_type else None,
    )


def fn_get_distribution(df: pd.DataFrame, col: str, top_n: int = 15) -> Dict:
    """
    Value frequency counts for a column.
    Use for: 'how many of each', 'breakdown', 'count per category'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    top_n = min(int(top_n) if top_n else 15, 20)

    vc = df[col].value_counts().head(top_n)
    dist_df = vc.reset_index()
    dist_df.columns = [col, "count"]

    top_val = str(vc.index[0])
    top_cnt = int(vc.iloc[0])
    pct = round(top_cnt / len(df) * 100, 1)

    answer = (
        f"**{top_val}** is the most frequent value in **{col}** "
        f"with **{top_cnt:,} occurrences** ({pct}%). "
        f"Showing top {len(vc)} of {df[col].nunique()} unique values."
    )
    # Suppress chart when there's only one unique value
    chart_type = "bar" if len(dist_df) > 1 else None
    return _result(
        answer,
        summary=vc.to_string(),
        result_df=dist_df if chart_type else None,
        chart_type=chart_type,
        x_col=col if chart_type else None,
        y_col="count" if chart_type else None,
        chart_title=f"Distribution of {col}" if chart_type else None,
    )


def fn_get_filtered_summary(
    df: pd.DataFrame,
    filter_col: str,
    filter_value: str,
    target_col: Optional[str] = None,
    agg: str = "sum",
) -> Dict:
    """
    Filter rows matching a value, then optionally aggregate.
    Use for: 'total sales in East region', 'orders from category X', 'how many orders are at risk of late delivery'.
    """
    if filter_col not in df.columns:
        return _error(f"Column '{filter_col}' not found.")

    filtered = df[
        df[filter_col].astype(str).str.strip().str.lower() == str(filter_value).lower().strip()
    ]
    if filtered.empty:
        return _error(f"No rows found where {filter_col} = '{filter_value}'.")

    n = len(filtered)
    total_len = len(df)
    pct = round(n / total_len * 100, 1) if total_len > 0 else 0

    if (target_col and target_col in filtered.columns
            and pd.api.types.is_numeric_dtype(filtered[target_col])):
        agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"
        val = _safe_agg(filtered[target_col].dropna(), agg)
        answer = (
            f"For **{filter_col} = '{filter_value}'** ({n:,} records, **{pct}%** of total):\n"
            f"- **{agg.title()} of {target_col}**: **{_fmt(val)}**"
        )
    else:
        answer = (
            f"Found **{n:,} records** (**{pct}%** of all {total_len:,} records) "
            f"matching **{filter_col} = '{filter_value}'**."
        )
    return _result(answer, summary=answer)


def fn_get_describe(df: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict:
    """
    Descriptive statistics (mean, std, min, max, quartiles) for numeric columns.
    Use for: 'summary stats', 'average and range', 'spread of values'.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols:
        cols = [c for c in cols if c in df.columns and c in num_cols]
    cols = cols or num_cols
    if not cols:
        return _error("No numeric columns found.")

    desc = df[cols].describe().round(2)
    lines = []
    for col in cols[:6]:
        s = desc[col]
        lines.append(
            f"- **{col}**: mean={_fmt(s['mean'])}, "
            f"min={_fmt(s['min'])}, max={_fmt(s['max'])}, "
            f"std={_fmt(s['std'])}"
        )
    answer = f"**Statistical Summary** ({len(cols)} column(s)):\n" + "\n".join(lines)
    return _result(answer, summary=desc.to_string())


# ===========================================================================
# Category 3 — Statistical Analysis
# ===========================================================================

def fn_get_correlation(df: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict:
    """Pearson correlation matrix for numeric columns. Highlights relationship strength and direction."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols:
        cols = [c for c in cols if c in df.columns and c in num_cols]
    cols = cols or num_cols
    if len(cols) < 2:
        return _error("Need at least 2 numeric columns for correlation analysis.")

    corr = df[cols].corr().round(3)
    arr = corr.abs().to_numpy().copy()   # writable numpy array for fill_diagonal
    np.fill_diagonal(arr, 0)
    corr_abs_mat = pd.DataFrame(arr, index=corr.index, columns=corr.columns)

    flat_max_idx = corr_abs_mat.values.argmax()
    n = len(cols)
    r, c = divmod(flat_max_idx, n)
    max_val = arr[r, c]

    if max_val > 0 or len(cols) == 2:
        col_a, col_b = (cols[0], cols[1]) if len(cols) == 2 else (cols[r], cols[c])
        val = float(corr.loc[col_a, col_b])
        strength = "strong" if abs(val) > 0.7 else ("moderate" if abs(val) > 0.4 else ("weak" if abs(val) > 0.15 else "very weak / negligible"))
        direction = "positive" if val > 0 else "negative"

        if len(cols) == 2:
            if abs(val) < 0.15:
                answer = (
                    f"**Correlation between {col_a} and {col_b}:** **r = {val:.3f}** ({strength} relationship).\n"
                    f"- Higher values of **{col_a}** do **not** appear to significantly increase or drive **{col_b}**.\n"
                    f"- There is little to no linear association between these two metrics."
                )
            elif val > 0:
                answer = (
                    f"**Correlation between {col_a} and {col_b}:** **r = {val:.3f}** ({strength} positive relationship).\n"
                    f"- Higher values of **{col_a}** tend to be associated with higher **{col_b}**."
                )
            else:
                answer = (
                    f"**Correlation between {col_a} and {col_b}:** **r = {val:.3f}** ({strength} negative / inverse relationship).\n"
                    f"- Higher values of **{col_a}** tend to be associated with lower **{col_b}**."
                )

            # Sample points for scatter plot so frontend renders fast and smoothly
            scatter_sample = df[[col_a, col_b]].dropna()
            if len(scatter_sample) > 1000:
                scatter_sample = scatter_sample.sample(1000, random_state=42)

            return _result(
                answer,
                summary=corr.to_string(),
                result_df=scatter_sample,
                chart_type="scatter",
                x_col=col_a,
                y_col=col_b,
                chart_title=f"Scatter Plot: {col_a} vs {col_b} (r = {val:.3f})",
                needs_llm=False,
            )
        else:
            answer = (
                f"Strongest correlation: **{col_a}** <-> **{col_b}** "
                f"(r = **{val:.3f}**, {strength} {direction})."
            )
            from services.visualizer import DataVisualizer
            chart_json = DataVisualizer.build_heatmap(
                corr, title="Correlation Matrix Heatmap"
            )
            return _result(answer, summary=corr.to_string(), chart_json=chart_json, needs_llm=False)
    else:
        answer = "No significant correlations detected among numeric columns."
        return _result(answer, summary=corr.to_string(), needs_llm=False)


def fn_get_customer_metrics(
    df: pd.DataFrame,
    customer_col: str,
    value_col: str,
) -> Dict:
    """
    Detailed customer metrics: revenue per customer (mean, median), active customer count,
    spend range, and top customer rankings.
    Use for: 'how much revenue does each customer generate', 'revenue per customer', 'customer spend'.
    """
    err = _validate_cols(df, customer_col, value_col)
    if err:
        return _error(err)
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' must be a numeric column.")

    cust_grp = df.groupby(customer_col)[value_col].sum()
    n_cust = len(cust_grp)
    if n_cust == 0:
        return _error("No customer records found.")

    total_rev = float(cust_grp.sum())
    mean_rev = float(cust_grp.mean())
    median_rev = float(cust_grp.median())
    min_rev = float(cust_grp.min())
    max_rev = float(cust_grp.max())

    top5 = cust_grp.nlargest(5)
    top_df = top5.reset_index()
    top_df.columns = [customer_col, value_col]

    lines = [
        f"**Customer Revenue Metrics ({n_cust:,} Unique Customers):**",
        f"- **Average Revenue Per Customer (Mean):** **{_fmt(mean_rev)}**",
        f"- **Median Spend Per Customer:** **{_fmt(median_rev)}**",
        f"- **Total Revenue Generated:** **{_fmt(total_rev)}**",
        f"- **Spend Range:** {_fmt(min_rev)} to {_fmt(max_rev)}",
        f"\n**Top 5 Customers by Total Revenue:**"
    ]
    for i, (cid, val) in enumerate(top5.items()):
        pct = round(val / total_rev * 100, 1) if total_rev > 0 else 0
        lines.append(f"{i+1}. **{cid}**: {_fmt(val)} ({pct}% of total)")

    answer = "\n".join(lines)
    return _result(
        answer,
        summary=answer,
        result_df=top_df,
        chart_type="bar",
        x_col=customer_col,
        y_col=value_col,
        chart_title=f"Top Customers by {value_col}",
        needs_llm=False
    )



def fn_get_outliers(df: pd.DataFrame, col: str) -> Dict:
    """
    Detect outliers using IQR method.
    Use for: 'unusual values', 'anomalies', 'extreme records'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        return _error(f"'{col}' is not a numeric column.")

    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return _result(f"No outliers detected in **{col}** (zero spread — IQR = 0).")

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    n = len(outliers)

    if n == 0:
        return _result(
            f"No outliers in **{col}** (normal range: {_fmt(lower)} to {_fmt(upper)})."
        )

    pct = round(n / len(s) * 100, 1)
    answer = (
        f"Found **{n:,} outliers** ({pct}% of values) in **{col}**.\n"
        f"- Normal IQR range: **{_fmt(lower)}** to **{_fmt(upper)}**\n"
        f"- Outlier range: {_fmt(float(outliers.min()))} to {_fmt(float(outliers.max()))}"
    )
    return _result(answer, summary=answer)


def fn_compare_segments(
    df: pd.DataFrame,
    segment_col: str,
    val1: str,
    val2: str,
    metric_col: str,
    agg: str = "mean",
) -> Dict:
    """
    Compare two segment values on a metric.
    Use for: 'East vs West revenue', 'Category A vs B performance'.
    """
    err = _validate_cols(df, segment_col, metric_col)
    if err:
        return _error(err)
    agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "mean"

    def get_val(v):
        mask = df[segment_col].astype(str).str.lower() == v.lower()
        s = df[mask][metric_col].dropna()
        return _safe_agg(s, agg) if not s.empty else None

    v1, v2 = get_val(val1), get_val(val2)
    if v1 is None: return _error(f"No data for {segment_col} = '{val1}'")
    if v2 is None: return _error(f"No data for {segment_col} = '{val2}'")

    diff = v1 - v2
    pct_diff = round((diff / v2) * 100, 1) if v2 != 0 else 0
    arrow = "higher" if diff > 0 else "lower"

    answer = (
        f"**{val1}** vs **{val2}** ({agg} of {metric_col}):\n"
        f"- **{val1}**: {_fmt(v1)}\n"
        f"- **{val2}**: {_fmt(v2)}\n"
        f"- {'Up' if diff>0 else 'Down'} **{_fmt(abs(diff))}** "
        f"({abs(pct_diff)}% {arrow})"
    )
    cmp_df = pd.DataFrame({segment_col: [val1, val2], metric_col: [v1, v2]})
    return _result(
        answer,
        summary=answer,
        result_df=cmp_df,
        chart_type="bar",
        x_col=segment_col,
        y_col=metric_col,
        chart_title=f"{val1} vs {val2}: {agg.title()} {metric_col}",
    )


def fn_get_percentile_breakdown(df: pd.DataFrame, col: str) -> Dict:
    """Quartile and percentile breakdown for a numeric column."""
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        return _error(f"'{col}' is not numeric.")

    s = df[col].dropna()
    pcts = {10: 0.10, 25: 0.25, 50: 0.50, 75: 0.75, 90: 0.90, 95: 0.95, 99: 0.99}
    vals = {p: float(s.quantile(q)) for p, q in pcts.items()}
    lines = "\n".join(f"- **{p}th percentile**: {_fmt(v)}" for p, v in vals.items())
    return _result(f"**Percentile Breakdown of {col}**:\n{lines}", summary=lines)


def fn_get_single_percentile(df: pd.DataFrame, col: str, percentile: int = 90) -> Dict:
    """
    Return a single specific percentile value for a numeric column.
    Use for: '90th percentile of sales', 'median order value', '75th percentile of revenue'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        return _error(f"'{col}' is not a numeric column.")
    percentile = max(1, min(int(percentile), 99))
    s = df[col].dropna()
    val = float(s.quantile(percentile / 100))
    mean_val = float(s.mean())
    pct_above = round((s > val).sum() / len(s) * 100, 1)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10 if percentile not in (11, 12, 13) else 0, "th")
    answer = (
        f"The **{percentile}{suffix} percentile** of **{col}** is **{_fmt(val)}**.\n"
        f"- This means {100 - pct_above}% of records have {col} at or below this value.\n"
        f"- Mean for reference: **{_fmt(mean_val)}**"
    )
    return _result(answer, summary=answer, needs_llm=False)


def fn_get_pct_above_threshold(
    df: pd.DataFrame,
    col: str,
    threshold: float,
    group_col: Optional[str] = None,
) -> Dict:
    """
    Calculate the percentage of records with a column value above a given threshold.
    Use for: 'what % of customers spend more than 10000', 'how many orders exceed 500',
    'percentage of sales above 1000', 'share of orders over threshold'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        return _error(f"'{col}' is not a numeric column.")

    s = df[col].dropna()
    total = len(s)
    above = int((s > threshold).sum())
    pct = round(above / total * 100, 2) if total > 0 else 0

    if group_col and group_col in df.columns:
        grp = (
            df.groupby(group_col)[col]
            .apply(lambda x: round((x.dropna() > threshold).sum() / len(x.dropna()) * 100, 1) if len(x.dropna()) > 0 else 0)
            .sort_values(ascending=False)
            .reset_index()
        )
        grp.columns = [group_col, f"% above {_fmt(threshold)}"]
        top = grp.iloc[0]
        answer = (
            f"**{above:,} of {total:,} records** ({pct}%) have **{col} > {_fmt(threshold)}**.\n"
            f"By {group_col}: highest in **{top[group_col]}** ({top[f'% above {_fmt(threshold)}']:.1f}%)."
        )
        return _result(answer, summary=grp.to_string(index=False), needs_llm=False)

    answer = (
        f"**{above:,} out of {total:,} records** ({pct}%) have **{col}** above **{_fmt(threshold)}**.\n"
        f"- That leaves {total - above:,} records ({round(100 - pct, 2)}%) at or below the threshold."
    )
    return _result(answer, summary=answer, needs_llm=False)


def fn_get_entities_above_threshold(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    threshold: float,
    agg: str = "sum",
    op: str = ">=",
    top_n: int = 15,
) -> Dict[str, Any]:
    """
    Finds entities (e.g. countries, products, customers) whose aggregated metric
    meets or exceeds/falls below a specified threshold (e.g. 'countries with sales > $5M',
    'products with profit ratio below zero').
    """
    err = _validate_cols(df, group_col, value_col)
    if err:
        return _error(err)
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not a numeric column.")

    agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"
    clean_df = df.dropna(subset=[group_col, value_col])
    if clean_df.empty:
        return _error("No valid non-null records available for this query.")

    grouped = clean_df.groupby(group_col, as_index=False)[value_col].agg(agg)

    is_less = False
    if op in (">=", "greater than or equal to", "at least"):
        filtered = grouped[grouped[value_col] >= threshold]
        op_label = "at least"
    elif op in (">", "more than", "above", "exceed", "greater than", "over"):
        filtered = grouped[grouped[value_col] > threshold]
        op_label = "more than"
    elif op in ("<=", "at most", "less than or equal to"):
        filtered = grouped[grouped[value_col] <= threshold]
        op_label = "at most"
        is_less = True
    elif op in ("<", "less than", "below", "under"):
        filtered = grouped[grouped[value_col] < threshold]
        op_label = "less than"
        is_less = True
    else:
        filtered = grouped[grouped[value_col] >= threshold]
        op_label = "at least"

    # Sort descending for > queries, ascending (worst first) for < queries
    filtered = filtered.sort_values(by=value_col, ascending=is_less)
    count = len(filtered)
    total_groups = len(grouped)

    is_ratio = any(k in value_col.lower() for k in ["ratio", "margin", "rate", "percent", "%"])
    is_currency = not is_ratio and any(k in value_col.lower() for k in ["sale", "profit", "price", "cost", "revenue", "amount", "total"])
    thresh_prefix = "$" if is_currency else ""
    thresh_suffix = "%" if (is_ratio and abs(threshold) <= 1.0 and threshold != 0) else ""

    def _fmt_metric(val):
        if is_ratio and abs(val) <= 1.0:
            return f"{val * 100:.2f}%" if "%" in value_col.lower() or is_ratio else f"{val:.4f}"
        return f"{thresh_prefix}{_fmt(val)}"

    thresh_str = f"{thresh_prefix}{_fmt(threshold)}{thresh_suffix}"

    entity_label = group_col
    if entity_label.lower().startswith("order "):
        entity_label = entity_label[6:]
    elif entity_label.lower().startswith("customer "):
        entity_label = entity_label[9:]
    plural_label = (entity_label[:-1] + "ies") if entity_label.lower().endswith("y") else (entity_label + "s")

    if count == 0:
        sorted_grouped = grouped.sort_values(by=value_col, ascending=is_less)
        extreme_row = sorted_grouped.iloc[0]
        extreme_entity = extreme_row[group_col]
        extreme_val = _fmt_metric(extreme_row[value_col])
        extreme_label = "Lowest" if is_less else "Highest"
        answer = (
            f"**No {plural_label.lower()}** had {op_label} **{thresh_str}** in {value_col} ({agg}).\n"
            f"- {extreme_label} in dataset: **{extreme_entity}** with **{extreme_val}** (out of {total_groups:,} total {plural_label.lower()})."
        )
        return _result(answer, summary=answer, needs_llm=False)

    lines = [
        f"**{count:,} {plural_label.lower()}** have {op_label} **{thresh_str}** in {value_col} ({agg}) "
        f"(out of {total_groups:,} total {plural_label.lower()}):\n"
    ]
    for i, (_, row) in enumerate(filtered.head(top_n).iterrows(), start=1):
        v_str = _fmt_metric(row[value_col])
        lines.append(f"{i}. **{row[group_col]}**: {v_str}")

    if count > top_n:
        lines.append(f"\n*(Showing first {top_n} of {count} matching {plural_label.lower()})*")

    answer = "\n".join(lines)
    return _result(
        answer,
        summary=answer,
        result_df=filtered.head(top_n),
        chart_type="bar" if count > 1 else None,
        x_col=group_col,
        y_col=value_col,
        chart_title=f"{group_col} with {value_col} {op_label} {thresh_str}",
        needs_llm=False
    )


def fn_dual_metric_ranking(
    df: pd.DataFrame,
    group_col: str,
    metric1: str,
    agg1: str = "sum",
    ascending1: bool = False,
    metric2: Optional[str] = None,
    agg2: str = "mean",
    ascending2: bool = True,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Computes multi-metric aggregations across a categorical dimension to answer
    compound comparative questions like:
    'Which country has the highest total profit but the lowest average profit per order?'
    """
    metric2 = metric2 or metric1
    err = _validate_cols(df, group_col, metric1, metric2)
    if err:
        return _error(err)

    if not pd.api.types.is_numeric_dtype(df[metric1]):
        return _error(f"'{metric1}' is not numeric.")
    if not pd.api.types.is_numeric_dtype(df[metric2]):
        return _error(f"'{metric2}' is not numeric.")

    agg1 = agg1 if agg1 in ["sum", "mean", "count", "min", "max", "median"] else "sum"
    agg2 = agg2 if agg2 in ["sum", "mean", "count", "min", "max", "median"] else "mean"

    clean_df = df.dropna(subset=[group_col, metric1, metric2])
    if clean_df.empty:
        return _error("No valid non-null records available for analysis.")

    # Group by entity and compute both aggregations
    grouped = clean_df.groupby(group_col).agg(
        val1=(metric1, agg1),
        val2=(metric2, agg2)
    ).reset_index()

    if grouped.empty:
        return _error("No data to aggregate.")

    col1_label = f"{agg1.title()} of {metric1}"
    col2_label = f"{agg2.title()} of {metric2}"
    if col1_label == col2_label:
        col2_label = f"{col2_label} (Secondary)"

    # Metric 1 leader/lowest
    sorted1 = grouped.sort_values(by="val1", ascending=bool(ascending1))
    winner1 = sorted1.iloc[0]
    w1_name = str(winner1[group_col])
    w1_v1 = float(winner1["val1"])
    w1_v2 = float(winner1["val2"])

    # Metric 2 leader/lowest
    sorted2 = grouped.sort_values(by="val2", ascending=bool(ascending2))
    winner2 = sorted2.iloc[0]
    w2_name = str(winner2[group_col])
    w2_v1 = float(winner2["val1"])
    w2_v2 = float(winner2["val2"])

    label1_dir = "Lowest" if ascending1 else "Highest"
    label2_dir = "Lowest" if ascending2 else "Highest"

    lines = []
    if w1_name == w2_name:
        lines.append(
            f"**{w1_name}** satisfies both criteria simultaneously:\n"
            f"- **{label1_dir} {col1_label}:** **{_fmt(w1_v1)}**\n"
            f"- **{label2_dir} {col2_label}:** **{_fmt(w1_v2)}**"
        )
    else:
        lines.append(
            f"No single **{group_col}** satisfies both conditions simultaneously. Here is the comparative breakdown:\n"
            f"- **{label1_dir} {col1_label}:** **{w1_name}** ({_fmt(w1_v1)}), with {col2_label} of **{_fmt(w1_v2)}**.\n"
            f"- **{label2_dir} {col2_label}:** **{w2_name}** ({_fmt(w2_v2)}), with {col1_label} of **{_fmt(w2_v1)}**."
        )

    # Add a top comparison table
    display_df = grouped.rename(columns={"val1": col1_label, "val2": col2_label})
    sorted1_disp = display_df.sort_values(by=col1_label, ascending=bool(ascending1))
    sorted2_disp = display_df.sort_values(by=col2_label, ascending=bool(ascending2))
    top_combined = pd.concat([sorted1_disp.head(5), sorted2_disp.head(5)]).drop_duplicates().head(top_n)

    table_lines = [f"\n**Top Performance Comparison ({group_col}):**"]
    for i, (_, r) in enumerate(top_combined.iterrows(), 1):
        table_lines.append(f"{i}. **{r[group_col]}**: {col1_label} = **{_fmt(float(r[col1_label]))}** | {col2_label} = **{_fmt(float(r[col2_label]))}**")
    lines.extend(table_lines)

    answer = "\n".join(lines)
    return _result(
        answer,
        summary=top_combined.to_string(index=False),
        result_df=top_combined,
        chart_type="bar",
        x_col=group_col,
        y_col=col1_label,
        chart_title=f"{group_col}: {label1_dir} {col1_label} vs {label2_dir} {col2_label}",
        needs_llm=False
    )


def fn_get_median(df: pd.DataFrame, col: str) -> Dict:
    """
    Return the median value of a numeric column plus context (mean, IQR).
    Use for: 'median of X', 'middle value of Y', 'median delivery time', 'median order value'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        return _error(f"'{col}' is not a numeric column.")

    s = df[col].dropna()
    median = float(s.median())
    mean = float(s.mean())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    skew_dir = "right-skewed (high outliers pull mean up)" if mean > median * 1.05 else (
               "left-skewed (low outliers pull mean down)" if mean < median * 0.95 else "approximately symmetric")

    answer = (
        f"The **median** of **{col}** is **{_fmt(median)}**.\n"
        f"- Mean: **{_fmt(mean)}** (distribution is {skew_dir})\n"
        f"- Middle 50% of values (IQR): **{_fmt(q1)}** to **{_fmt(q3)}**"
    )
    return _result(answer, summary=answer, needs_llm=False)


# ===========================================================================
# Category 4 — Financial / Business Metrics
# ===========================================================================

def fn_get_profit_margin(
    df: pd.DataFrame,
    cost_col: str,
    revenue_col: str,
    group_col: Optional[str] = None,
) -> Dict:
    """
    Calculate profit margin % = (revenue - cost) / revenue * 100.
    Use for: 'profit margin', 'margin by category', 'which product is most profitable'.
    """
    err = _validate_cols(df, cost_col, revenue_col)
    if err:
        return _error(err)

    work = df.copy()
    work["_margin_pct"] = (
        (work[revenue_col] - work[cost_col])
        / work[revenue_col].replace(0, np.nan)
        * 100
    )

    if group_col and group_col in df.columns:
        result = (
            work.groupby(group_col)
            .agg(_margin_pct=("_margin_pct", "mean"),
                 **{revenue_col: (revenue_col, "sum"),
                    cost_col: (cost_col, "sum")})
            .rename(columns={"_margin_pct": "Margin%"})
            .round(2)
            .sort_values("Margin%", ascending=False)
            .reset_index()
        )
        avg = round(float(work["_margin_pct"].mean()), 1)
        top, bot = result.iloc[0], result.iloc[-1]
        answer = (
            f"**Profit Margin by {group_col}** (avg: **{avg}%**):\n"
            f"- Highest: **{top[group_col]}** at **{top['Margin%']:.1f}%**\n"
            f"- Lowest: **{bot[group_col]}** at **{bot['Margin%']:.1f}%**"
        )
        return _result(
            answer,
            summary=result.to_string(index=False),
            result_df=result,
            chart_type="bar",
            x_col=group_col,
            y_col="Margin%",
            chart_title=f"Profit Margin % by {group_col}",
        )

    total_rev = float(df[revenue_col].sum())
    total_cost = float(df[cost_col].sum())
    total_profit = total_rev - total_cost
    avg_margin = round(float(work["_margin_pct"].mean()), 1)
    answer = (
        f"**Profit Summary (all records)**:\n"
        f"- Total Revenue: **{_fmt(total_rev)}**\n"
        f"- Total Cost: **{_fmt(total_cost)}**\n"
        f"- Total Profit: **{_fmt(total_profit)}**\n"
        f"- Average Margin: **{avg_margin}%**"
    )
    return _result(answer, summary=answer)


def fn_get_inventory_value(
    df: pd.DataFrame,
    qty_col: str,
    cost_col: str,
    group_col: Optional[str] = None,
) -> Dict:
    """
    Total inventory value = quantity x cost price.
    Use for: 'stock value', 'how much is inventory worth', 'capital locked in stock'.
    """
    err = _validate_cols(df, qty_col, cost_col)
    if err:
        return _error(err)

    work = df.copy()
    work["_stock_value"] = work[qty_col] * work[cost_col]
    total = float(work["_stock_value"].sum())

    if group_col and group_col in df.columns:
        result = (
            work.groupby(group_col)["_stock_value"]
            .sum().sort_values(ascending=False).reset_index()
        )
        result.columns = [group_col, "Stock Value"]
        top = result.iloc[0]
        answer = (
            f"**Total Inventory Value: {_fmt(total)}**\n"
            f"- **{top[group_col]}** holds the largest share: "
            f"**{_fmt(float(top['Stock Value']))}** "
            f"({round(float(top['Stock Value'])/total*100,1)}%)"
        )
        return _result(
            answer,
            summary=result.to_string(index=False),
            result_df=result,
            chart_type="bar",
            x_col=group_col,
            y_col="Stock Value",
            chart_title=f"Inventory Value by {group_col}",
        )

    return _result(
        f"**Total Inventory Value: {_fmt(total)}** ({qty_col} x {cost_col})",
        summary=f"Total stock value: {_fmt(total)}"
    )


def fn_get_low_stock(
    df: pd.DataFrame,
    qty_col: str,
    threshold: float = 10,
    name_col: Optional[str] = None,
) -> Dict:
    """
    Items at or below a stock threshold.
    Use for: 'low stock alerts', 'products needing reorder', 'stock below X units'.
    """
    if qty_col not in df.columns:
        return _error(f"Column '{qty_col}' not found.")

    low = df[df[qty_col] <= float(threshold)].copy()
    n = len(low)

    if n == 0:
        return _result(f"All items have stock above **{threshold}** units — no alerts.")

    answer = f"**{n} item(s)** are at or below **{threshold}** units:\n"
    if name_col and name_col in df.columns:
        for _, row in low.sort_values(qty_col).head(10).iterrows():
            qty = int(row[qty_col]) if pd.notna(row[qty_col]) else 0
            answer += f"- **{row[name_col]}**: {qty} units\n"
        if n > 10:
            answer += f"  *(and {n-10} more items)*"
    else:
        avg_qty = round(float(low[qty_col].mean()), 1)
        answer += f"Average stock level of flagged items: **{avg_qty} units**."

    return _result(answer.strip(), summary=f"{n} items below threshold of {threshold}")


# ===========================================================================
# Category 5 — Time-based Analysis
# ===========================================================================

_FREQ_LABELS = {"D": "daily", "W": "weekly", "ME": "monthly", "QE": "quarterly", "YE": "yearly"}


def _parse_date_bound(val: str, end: bool = False) -> Optional[pd.Timestamp]:
    """
    Parse a flexible date string for boundary clamping.
    - A 4-digit year string like '2022' becomes 2022-01-01 (start) or 2022-12-31 (end).
    - ISO date strings like '2022-06-01' are parsed directly.
    """
    if val is None:
        return None
    val = str(val).strip()
    if val.isdigit() and len(val) == 4:
        return pd.Timestamp(f"{val}-12-31 23:59:59") if end else pd.Timestamp(f"{val}-01-01")
    try:
        return pd.Timestamp(val)
    except Exception:
        return None


def fn_get_date_bounds(
    df: pd.DataFrame,
    date_col: str,
    target: str = "first",
) -> Dict:
    """
    Get the earliest (first) date, latest (last) date, or overall time coverage of a dataset.
    Use for: 'when was the first order', 'what is the earliest date', 'what year did orders begin',
    'when was the last/latest transaction', 'what is the date span/range'.
    """
    if date_col not in df.columns:
        return _error(f"Column '{date_col}' not found.")
    try:
        series = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if series.empty:
            return _error(f"No valid dates could be parsed in column '{date_col}'.")

        min_date = series.min()
        max_date = series.max()

        min_str = min_date.strftime("%B %d, %Y") if hasattr(min_date, "strftime") else str(min_date)
        max_str = max_date.strftime("%B %d, %Y") if hasattr(max_date, "strftime") else str(max_date)
        min_iso = min_date.strftime("%Y-%m-%d") if hasattr(min_date, "strftime") else str(min_date)[:10]
        max_iso = max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date)[:10]
        min_year = getattr(min_date, "year", "")
        max_year = getattr(max_date, "year", "")

        t = (target or "first").lower()
        if any(w in t for w in ["first", "early", "start", "begin", "oldest", "min"]):
            answer = (
                f"The **first recorded date** in **{date_col}** is **{min_str}** (`{min_iso}`) "
                f"in the year **{min_year}**."
            )
        elif any(w in t for w in ["last", "late", "recent", "end", "max", "newest"]):
            answer = (
                f"The **latest recorded date** in **{date_col}** is **{max_str}** (`{max_iso}`) "
                f"in the year **{max_year}**."
            )
        else:
            days_span = (max_date - min_date).days
            answer = (
                f"The dataset covers dates from **{min_str}** (`{min_iso}`, year {min_year}) "
                f"to **{max_str}** (`{max_iso}`, year {max_year}), "
                f"spanning **{days_span:,} days** across **{len(series):,} records**."
            )

        return _result(answer, summary=answer, needs_llm=False)
    except Exception as e:
        return _error(f"Date boundary calculation error: {str(e)}")


def fn_get_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
    agg: str = "sum",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    n_periods: Optional[int] = None,
) -> Dict:
    """
    Aggregate a numeric column over time periods.
    Use for: 'trend', 'monthly revenue', 'daily orders', 'growth over time'.
    Supports optional from_date / to_date / n_periods to restrict the timeframe.
    """
    err = _validate_cols(df, date_col, value_col)
    if err:
        return _error(err)

    freq = freq if freq in _FREQ_LABELS else "ME"
    agg = agg if agg in ["sum", "mean", "count"] else "sum"

    try:
        ts = df.copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col])

        # Apply user-specified date filters BEFORE resampling
        start_ts = _parse_date_bound(from_date, end=False)
        end_ts = _parse_date_bound(to_date, end=True)
        if start_ts is not None:
            ts = ts[ts[date_col] >= start_ts]
        if end_ts is not None:
            ts = ts[ts[date_col] <= end_ts]

        if ts.empty:
            return _error("No data found in the specified date range.")

        ts = ts.set_index(date_col)
        result = ts[value_col].resample(freq).agg(agg).dropna().reset_index()
        result.columns = [date_col, value_col]

        # If n_periods specified, take the last N periods; otherwise cap at 24 when no range set
        if n_periods is not None and n_periods > 0:
            result = result.tail(int(n_periods))
        elif from_date is None and to_date is None:
            result = result.tail(24)

        result[date_col] = result[date_col].dt.strftime("%Y-%m-%d")

        if len(result) < 2:
            return _error("Not enough data points for time-series analysis in the specified range.")

        first_val = float(result.iloc[0][value_col])
        last_val = float(result.iloc[-1][value_col])
        trend_pct = round((last_val - first_val) / first_val * 100, 1) if first_val != 0 else 0
        arrow = "up" if trend_pct > 0 else "down"

        range_label = ""
        if from_date or to_date:
            range_label = f" ({from_date or ''}–{to_date or ''})"

        answer = (
            f"**{_FREQ_LABELS[freq].title()} {agg} of {value_col}**{range_label} over {len(result)} periods.\n"
            f"- Latest: **{_fmt(last_val)}** | Earliest: **{_fmt(first_val)}**\n"
            f"- Overall trend: **{'+' if trend_pct>0 else ''}{trend_pct}%** ({arrow})"
        )
        return _result(
            answer,
            summary=result.to_string(index=False),
            result_df=result,
            chart_type="line",
            x_col=date_col,
            y_col=value_col,
            chart_title=f"{agg.title()} of {value_col} over Time ({_FREQ_LABELS[freq]}){range_label}",
            needs_llm=True,
        )
    except Exception as e:
        return _error(f"Time-series error: {str(e)}")


def fn_get_date_range_summary(
    df: pd.DataFrame,
    date_col: str,
    from_date: str,
    to_date: str,
    value_col: Optional[str] = None,
    agg: str = "sum",
) -> Dict:
    """
    Filter by date range then aggregate.
    Use for: 'revenue in Q1', 'orders between Jan and Mar', 'last 30 days'.
    """
    if date_col not in df.columns:
        return _error(f"Column '{date_col}' not found.")
    try:
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        mask = (work[date_col] >= pd.to_datetime(from_date)) & (work[date_col] <= pd.to_datetime(to_date))
        filtered = work[mask]
        n = len(filtered)

        if n == 0:
            return _result(f"No records found between **{from_date}** and **{to_date}**.")

        if value_col and value_col in filtered.columns and pd.api.types.is_numeric_dtype(filtered[value_col]):
            agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"
            val = _safe_agg(filtered[value_col].dropna(), agg)
            answer = (
                f"**{from_date}** to **{to_date}**: **{n:,} records**, "
                f"{agg} of **{value_col}** = **{_fmt(val)}**"
            )
        else:
            answer = f"**{from_date}** to **{to_date}**: **{n:,} records** found."

        return _result(answer, summary=answer)
    except Exception as e:
        return _error(f"Date range error: {str(e)}")


def fn_get_growth_rate(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "ME",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    n_periods: Optional[int] = None,
) -> Dict:
    """
    Period-over-period growth rate with totals.
    Use for: 'month-over-month growth', 'YoY change', 'is revenue growing?', 'monthly total sales and percentage growth'.
    Supports optional from_date / to_date / n_periods to restrict the timeframe.
    """
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
                g_str = "N/A"
            else:
                s = "+" if g_val > 0 else ""
                g_str = f"{s}{g_val:.2f}%"
            table_rows.append(f"| {dt_str} | {_fmt(val)} | {g_str} |")

        table_md = "\n".join(table_rows)

        answer = (
            f"**{freq_label} Total {value_col} & Percentage Growth Rate**{range_label}:\n\n"
            f"- **Latest Period ({curr_period})**: Total **{_fmt(latest_val)}** (**{sign}{latest_growth:.2f}%** vs {prev_period} of {_fmt(prev_val)})\n"
            f"- **Average {freq_label} Growth**: **{avg_sign}{avg_growth:.2f}%**\n\n"
            f"| Period | Total {value_col} | Growth % (vs Prev Period) |\n"
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


def fn_get_monthly_pattern(df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
    """
    Average value by calendar month (seasonal pattern).
    Use for: 'which month is busiest', 'seasonality', 'peak sales month'.
    """
    err = _validate_cols(df, date_col, value_col)
    if err:
        return _error(err)
    try:
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        work["_month_num"] = work[date_col].dt.month
        work["_month"] = work[date_col].dt.strftime("%b")

        monthly = (
            work.groupby(["_month_num", "_month"])[value_col]
            .mean().reset_index().sort_values("_month_num")
        )
        monthly = monthly[["_month", value_col]].rename(columns={"_month": "Month"})

        peak = monthly.loc[monthly[value_col].idxmax()]
        low = monthly.loc[monthly[value_col].idxmin()]

        answer = (
            f"**Monthly Pattern of {value_col}**:\n"
            f"- Peak month: **{peak['Month']}** (avg: {_fmt(float(peak[value_col]))})\n"
            f"- Lowest month: **{low['Month']}** (avg: {_fmt(float(low[value_col]))})"
        )
        return _result(
            answer,
            summary=monthly.to_string(index=False),
            result_df=monthly,
            chart_type="bar",
            x_col="Month",
            y_col=value_col,
            chart_title=f"Average {value_col} by Month (Seasonal Pattern)",
            needs_llm=True,
        )
    except Exception as e:
        return _error(f"Monthly pattern error: {str(e)}")


# ===========================================================================
# Category 6 — Search
# ===========================================================================

def fn_search_items(
    df: pd.DataFrame,
    keyword: str,
    search_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Keyword search across text columns.
    Use for: 'find product X', 'show me rows containing Y', 'search for Z'.
    """
    if not keyword:
        return _error("Keyword is required for search.")

    cols = (
        [c for c in search_cols if c in df.columns]
        if search_cols
        else df.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    if not cols:
        return _error("No text columns available to search.")

    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask |= df[col].astype(str).str.lower().str.contains(keyword.lower(), na=False)

    results = df[mask].head(20)
    n = int(mask.sum())

    if n == 0:
        return _result(
            f"No results found for '**{keyword}**' in: {', '.join(cols)}."
        )

    answer = (
        f"Found **{n} record(s)** matching '**{keyword}**' "
        f"(searched: {', '.join(cols)})."
    )
    return _result(answer, summary=results.to_string(index=False), result_df=results)


# ===========================================================================
# Category 8 — Advanced Charts (locally computed, zero-API)
# ===========================================================================

def fn_get_pareto(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n: int = 20,
) -> Dict:
    """
    80/20 Pareto analysis — bar chart of sorted values + cumulative % line.
    Use for: 'which products generate 80% of revenue', 'pareto of sales', '80-20 analysis'.
    """
    err = _validate_cols(df, group_col, value_col)
    if err:
        return _error(err)
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not numeric.")

    n = min(max(int(n), 5), 30)
    grouped = (
        df.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    total = float(grouped.sum())
    cumsum = grouped.cumsum()
    cum_pct = (cumsum / total * 100).round(1)

    cutoff = int((cum_pct <= 80).sum()) + 1
    cutoff = min(cutoff, len(grouped))

    pareto_df = pd.DataFrame({
        group_col: grouped.index,
        value_col: grouped.values,
        "Cumulative %": cum_pct.values,
    })

    top3 = ", ".join(f"**{str(r)}**" for r in grouped.head(3).index)
    answer = (
        f"**Pareto Analysis — {value_col} by {group_col}**\n"
        f"- Top **{cutoff} {group_col}(s)** account for **80%** of total {value_col}.\n"
        f"- Top 3: {top3}.\n"
        f"- Total: **{_fmt(total)}** across {len(grouped)} categories shown."
    )

    from services.visualizer import DataVisualizer
    chart_json = DataVisualizer.build_pareto(
        pareto_df, group_col, value_col,
        title=f"Pareto: {value_col} by {group_col}"
    )
    return _result(answer, summary=pareto_df.to_string(index=False),
                   result_df=pareto_df, chart_json=chart_json, needs_llm=False)


def fn_get_heatmap_cross(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: Optional[str] = None,
    agg: str = "sum",
) -> Dict:
    """
    Cross-tab / pivot heatmap of two categorical columns optionally weighted by a numeric.
    Use for: 'heatmap of sales by product and region', 'orders by day and hour'.
    """
    if row_col not in df.columns:
        return _error(f"Column '{row_col}' not found.")
    if col_col not in df.columns:
        return _error(f"Column '{col_col}' not found.")

    try:
        if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
            agg = agg if agg in ["sum", "mean", "count", "min", "max"] else "sum"
            pivot = df.pivot_table(index=row_col, columns=col_col,
                                   values=value_col, aggfunc=agg, fill_value=0)
        else:
            pivot = pd.crosstab(df[row_col], df[col_col])

        n_rows, n_cols = pivot.shape
        max_val = float(pivot.values.max())
        top_row = str(pivot.sum(axis=1).idxmax())
        top_col = str(pivot.sum(axis=0).idxmax())
        value_label = f"{agg} of {value_col}" if value_col else "count"

        answer = (
            f"**Cross-tab Heatmap** ({n_rows} {row_col} × {n_cols} {col_col})\n"
            f"- Highest {value_label}: **{_fmt(max_val)}**\n"
            f"- Most active {row_col}: **{top_row}**\n"
            f"- Most active {col_col}: **{top_col}**"
        )

        from services.visualizer import DataVisualizer
        chart_json = DataVisualizer.build_heatmap(
            pivot, title=f"{value_label} by {row_col} × {col_col}"
        )
        return _result(answer, summary=pivot.to_string(), chart_json=chart_json, needs_llm=False)

    except Exception as exc:
        return _error(f"Heatmap computation failed: {exc}")


def fn_get_treemap(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    agg: str = "sum",
) -> Dict:
    """
    Hierarchical treemap of value by one or two grouping columns.
    Use for: 'treemap of revenue by category', 'show revenue by product line and product'.
    """
    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        return _error("No valid group columns found.")
    if value_col not in df.columns:
        return _error(f"Column '{value_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not numeric.")

    agg = agg if agg in ["sum", "mean", "count"] else "sum"
    agg_df = df.groupby(valid_cols, as_index=False)[value_col].agg(agg)
    total = float(agg_df[value_col].sum())
    top = agg_df.sort_values(value_col, ascending=False).iloc[0]

    answer = (
        f"**Treemap — {value_col} by {' > '.join(valid_cols)}**\n"
        f"- Largest segment: **{str(top[valid_cols[-1]])}** with **{_fmt(float(top[value_col]))}** "
        f"({round(float(top[value_col])/total*100,1)}% of total).\n"
        f"- {len(agg_df)} segments shown."
    )

    from services.visualizer import DataVisualizer
    chart_json = DataVisualizer.build_treemap(
        agg_df, valid_cols, value_col,
        title=f"{value_col} by {' > '.join(valid_cols)}"
    )
    return _result(answer, summary=agg_df.to_string(index=False),
                   result_df=agg_df, chart_json=chart_json, needs_llm=False)


def fn_get_funnel(
    df: pd.DataFrame,
    stage_col: str,
    count_col: Optional[str] = None,
    ordered_stages: Optional[List[str]] = None,
) -> Dict:
    """
    Funnel chart for pipeline / conversion step analysis.
    Use for: 'show conversion funnel', 'where are customers dropping off', 'pipeline funnel'.
    """
    if stage_col not in df.columns:
        return _error(f"Column '{stage_col}' not found.")

    if count_col and count_col in df.columns and pd.api.types.is_numeric_dtype(df[count_col]):
        funnel_df = df.groupby(stage_col)[count_col].sum().reset_index()
        funnel_df.columns = [stage_col, "Count"]
        count_col = "Count"
    else:
        vc = df[stage_col].value_counts()
        funnel_df = vc.reset_index()
        funnel_df.columns = [stage_col, "Count"]
        count_col = "Count"

    if ordered_stages:
        stage_order = {s: i for i, s in enumerate(ordered_stages)}
        funnel_df["_ord"] = funnel_df[stage_col].map(stage_order).fillna(999)
        funnel_df = funnel_df.sort_values("_ord").drop("_ord", axis=1)
    else:
        funnel_df = funnel_df.sort_values(count_col, ascending=False)

    top_stage = str(funnel_df.iloc[0][stage_col])
    top_count = int(funnel_df.iloc[0][count_col])
    if len(funnel_df) > 1:
        bottom_count = int(funnel_df.iloc[-1][count_col])
        conv_rate = round(bottom_count / top_count * 100, 1) if top_count else 0
        answer = (
            f"**Funnel Analysis — {stage_col}**\n"
            f"- Top stage: **{top_stage}** ({_fmt(top_count)} records)\n"
            f"- Overall conversion: **{conv_rate}%** (top → bottom stage)"
        )
    else:
        answer = f"Only one stage found: **{top_stage}** ({_fmt(top_count)})."

    from services.visualizer import DataVisualizer
    chart_json = DataVisualizer.build_funnel(
        funnel_df, stage_col, count_col,
        title=f"Funnel: {stage_col}"
    )
    return _result(answer, summary=funnel_df.to_string(index=False),
                   result_df=funnel_df, chart_json=chart_json, needs_llm=False)


def fn_get_waterfall(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    n: int = 15,
) -> Dict:
    """
    Waterfall chart showing contribution of each category to a total.
    Use for: 'waterfall of revenue by product', 'contribution breakdown', 'what drives total sales'.
    """
    err = _validate_cols(df, label_col, value_col)
    if err:
        return _error(err)
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not numeric.")

    agg_df = (
        df.groupby(label_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
    total = float(agg_df.sum())
    top = str(agg_df.index[0])
    top_val = float(agg_df.iloc[0])

    labels = list(agg_df.index)
    values = [float(v) for v in agg_df.values]

    answer = (
        f"**Waterfall — {value_col} contributions by {label_col}**\n"
        f"- Largest contributor: **{top}** (+{_fmt(top_val)})\n"
        f"- Total ({len(labels)} categories): **{_fmt(total)}**"
    )

    from services.visualizer import DataVisualizer
    chart_json = DataVisualizer.build_waterfall(
        labels, values,
        title=f"{value_col} Waterfall by {label_col}"
    )
    return _result(answer, summary=agg_df.to_string(), chart_json=chart_json, needs_llm=False)


def fn_get_rolling_average(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int = 30,
) -> Dict:
    """
    Rolling / moving average trend line.
    Use for: 'rolling 30-day sales', '7-day moving average', 'smoothed trend of revenue'.
    """
    if date_col not in df.columns:
        return _error(f"Date column '{date_col}' not found.")
    if value_col not in df.columns:
        return _error(f"Value column '{value_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not numeric.")

    window = max(2, min(int(window), 90))
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col)
    if ts.empty:
        return _error("No valid date rows found.")

    daily = ts.groupby(date_col)[value_col].sum()
    rolling_col = f"{window}d Avg {value_col}"
    rolling = daily.rolling(window=window, min_periods=1).mean()

    result_df = pd.DataFrame({
        date_col: daily.index,
        value_col: daily.values,
        rolling_col: rolling.values,
    })

    overall_trend = "upward" if float(rolling.iloc[-1]) > float(rolling.iloc[0]) else "downward"
    answer = (
        f"**{window}-Day Rolling Average — {value_col}**\n"
        f"- Overall trend: **{overall_trend}**\n"
        f"- Latest {window}d avg: **{_fmt(float(rolling.iloc[-1]))}**\n"
        f"- Period: {ts[date_col].min().date()} to {ts[date_col].max().date()}"
    )
    return _result(
        answer, summary=answer, result_df=result_df,
        chart_type="line", x_col=date_col, y_col=rolling_col,
        chart_title=f"{window}-Day Rolling Avg of {value_col}",
        needs_llm=False,
    )


def fn_get_bubble_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    cat_col: Optional[str] = None,
    n: int = 200,
) -> Dict:
    """
    Bubble chart comparing three numeric variables (optionally coloured by a category).
    Use for: 'bubble chart of revenue vs profit sized by orders', 'plot X vs Y sized by Z'.
    """
    req_cols = [x_col, y_col, size_col]
    if cat_col:
        req_cols.append(cat_col)
    err = _validate_cols(df, *req_cols)
    if err:
        return _error(err)
    for c in [x_col, y_col, size_col]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return _error(f"'{c}' must be numeric for a bubble chart.")
    # Build sample with deduplicated columns
    seen_c, unique_cols = set(), []
    for c in req_cols:
        if c and c not in seen_c:
            unique_cols.append(c)
            seen_c.add(c)
    sample = df[unique_cols].dropna().head(n)
    # Correlate using original Series (not potentially-duplicate slice)
    try:
        corr = float(df[x_col].corr(df[y_col]))
    except Exception:
        corr = 0.0
    strength = "strong" if abs(corr) > 0.7 else ("moderate" if abs(corr) > 0.4 else "weak")
    direction = "positive" if corr > 0 else "negative"

    answer = (
        f"**Bubble Chart — {y_col} vs {x_col} (sized by {size_col})**\n"
        f"- {y_col} vs {x_col} correlation: **r = {corr:.3f}** ({strength} {direction})\n"
        f"- {len(sample)} data points plotted."
    )
    return _result(
        answer, summary=answer, result_df=sample,
        chart_type="bubble", x_col=x_col, y_col=y_col,
        chart_title=f"{y_col} vs {x_col} (size = {size_col})",
        chart_kwargs={"size_col": size_col, "color_col": cat_col},
        needs_llm=False,
    )


def fn_get_grouped_comparison(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    value_col: str,
    agg: str = "sum",
) -> Dict:
    """
    Grouped / stacked bar comparing a metric across two categorical dimensions.
    Use for: 'sales by year and product line', 'compare regions by category', 'grouped bar'.
    """
    err = _validate_cols(df, x_col, group_col, value_col)
    if err:
        return _error(err)
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not numeric.")

    agg = agg if agg in ["sum", "mean", "count"] else "sum"
    result = df.groupby([x_col, group_col], as_index=False)[value_col].agg(agg)
    result = result.sort_values(value_col, ascending=False)

    n_x = result[x_col].nunique()
    n_g = result[group_col].nunique()
    top = result.iloc[0]

    answer = (
        f"**Grouped Comparison — {value_col} by {x_col} and {group_col}**\n"
        f"- {n_x} {x_col} × {n_g} {group_col} combinations\n"
        f"- Top: **{top[x_col]}** / **{top[group_col]}** = **{_fmt(float(top[value_col]))}**"
    )
    return _result(
        answer, summary=result.to_string(index=False), result_df=result,
        chart_type="grouped_bar",
        x_col=x_col, y_col=value_col,
        chart_title=f"{agg.title()} {value_col} by {x_col} and {group_col}",
        chart_kwargs={"color_col": group_col},
        needs_llm=False,
    )


def fn_get_cohort_retention(
    df: pd.DataFrame,
    date_col: str,
    customer_col: str,
) -> Dict:
    """
    Monthly cohort retention heatmap.
    Use for: 'cohort analysis', 'customer retention by cohort', 'repeat purchase by month'.
    """
    if date_col not in df.columns:
        return _error(f"Date column '{date_col}' not found.")
    if customer_col not in df.columns:
        return _error(f"Customer column '{customer_col}' not found.")

    try:
        ts = df[[date_col, customer_col]].copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
        ts = ts.dropna(subset=[date_col])
        if ts.empty:
            return _error("No valid date rows for cohort analysis.")

        ts["cohort_month"] = ts[date_col].dt.to_period("M")
        ts["order_month"]  = ts[date_col].dt.to_period("M")

        first_purchase = (
            ts.groupby(customer_col)["cohort_month"].min()
            .rename("first_cohort")
        )
        ts = ts.join(first_purchase, on=customer_col)
        ts["period"] = (ts["cohort_month"] - ts["first_cohort"]).apply(
            lambda x: x.n if hasattr(x, "n") else 0
        )

        cohort_sizes = ts.groupby("first_cohort")[customer_col].nunique()
        cohort_table = (
            ts.groupby(["first_cohort", "period"])[customer_col]
            .nunique()
            .unstack()
        )
        retention = (cohort_table.divide(cohort_sizes, axis=0) * 100).round(1)

        n_cohorts = len(retention)
        avg_m1_ret = float(retention[1].mean()) if 1 in retention.columns else 0
        answer = (
            f"**Cohort Retention Analysis**\n"
            f"- {n_cohorts} monthly cohorts analysed.\n"
            f"- Average Month-1 retention: **{avg_m1_ret:.1f}%**\n"
            f"- (Green = high retention, Red = high churn)"
        )

        retention.index = retention.index.astype(str)
        retention.columns = [f"Month {c}" for c in retention.columns]

        from services.visualizer import DataVisualizer
        chart_json = DataVisualizer.build_cohort_heatmap(
            retention, title="Cohort Retention Heatmap (%)"
        )
        return _result(answer, summary=retention.to_string(), chart_json=chart_json, needs_llm=False)

    except Exception as exc:
        return _error(f"Cohort computation failed: {exc}")


def fn_get_aggregate(
    df: pd.DataFrame,
    col: str,
    agg: str = "sum",
    label: Optional[str] = None,
) -> Dict:
    """
    Simple scalar aggregation — sum, mean, count, min, max of one column.
    Returns a SINGLE NUMBER answer with NO chart (intentional).
    Use for: 'total sales', 'total revenue', 'average order value', 'count of orders',
             'what is our net profit', 'total net sales', 'grand total'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        return _error(f"'{col}' is not numeric.")

    agg = agg if agg in ["sum", "mean", "count", "min", "max", "median"] else "sum"
    label = label or col

    series = df[col].dropna()
    if agg == "sum":
        val = float(series.sum())
        agg_name = "Total"
    elif agg == "mean":
        val = float(series.mean())
        agg_name = "Average"
    elif agg == "count":
        val = float(len(series))
        agg_name = "Count"
    elif agg == "min":
        val = float(series.min())
        agg_name = "Minimum"
    elif agg == "max":
        val = float(series.max())
        agg_name = "Maximum"
    elif agg == "median":
        val = float(series.median())
        agg_name = "Median"
    else:
        val = float(series.sum())
        agg_name = "Total"

    if agg == "count":
        answer = f"**Total {label}:** **{int(val):,}**"
    elif agg == "mean":
        answer = f"**Average {label}:** **{_fmt(val)}**"
    elif agg == "min":
        answer = f"**Minimum {label}:** **{_fmt(val)}**"
    elif agg == "max":
        answer = f"**Maximum {label}:** **{_fmt(val)}**"
    elif agg == "median":
        answer = f"**Median {label}:** **{_fmt(val)}**"
    else:
        answer = f"**Total {label}:** **{_fmt(val)}**"
    # Intentionally NO chart_type / chart_json — scalar answers don't need charts
    return _result(answer, summary=f"{agg_name} {label} = {_fmt(val)}", needs_llm=False)


def fn_count_unique(df: pd.DataFrame, col: str, label: Optional[str] = None) -> Dict:
    """
    Count the number of DISTINCT values in a column.
    Use for: 'how many unique customers', 'how many distinct products',
             'total unique regions', 'how many customers do we have',
             'number of different order statuses'.
    """
    if col not in df.columns:
        return _error(f"Column '{col}' not found.")
    label = label or col
    n_unique = int(df[col].nunique())
    n_total = len(df[col].dropna())
    sample = df[col].dropna().unique()[:5]
    sample_str = ", ".join(str(v) for v in sample)
    answer = (
        f"There are **{n_unique:,} unique {label}** in the dataset "
        f"(out of {n_total:,} total records).\n"
        f"- Sample values: {sample_str}{'...' if n_unique > 5 else ''}"
    )
    return _result(answer, summary=f"Unique {label} = {n_unique:,}", needs_llm=False)


def fn_get_multi_aggregate(
    df: pd.DataFrame,
    metrics: List[Dict[str, str]],
) -> Dict:
    """
    Compute multiple scalar metrics in one combined answer.
    Use for: 'total sales and profit margin', 'avg order value and total revenue',
             'what is our sales, profit, and quantity', multi-metric questions.

    Args:
        metrics: list of dicts, each with keys:
            - 'col':   column name in df
            - 'agg':   aggregation type ('sum', 'mean', 'count', 'min', 'max')
            - 'label': display label (optional, defaults to col name)
    """
    if not metrics:
        return _error("No metrics specified.")

    agg_map = {"sum": "Total", "mean": "Average", "count": "Count",
                "min": "Minimum", "max": "Maximum", "median": "Median"}

    lines = []
    summary_parts = []
    for m in metrics:
        col = m.get("col", "")
        agg = m.get("agg", "sum")
        label = m.get("label") or col

        if col not in df.columns:
            lines.append(f"- **{label}**: column not found")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"- **{label}**: not a numeric column")
            continue

        agg = agg if agg in agg_map else "sum"
        series = df[col].dropna()
        if agg == "sum":
            val = float(series.sum())
        elif agg == "mean":
            val = float(series.mean())
        elif agg == "count":
            val = float(len(series))
        elif agg == "min":
            val = float(series.min())
        elif agg == "max":
            val = float(series.max())
        elif agg == "median":
            val = float(series.median())
        else:
            val = float(series.sum())

        agg_name = agg_map[agg]
        lines.append(f"- **{agg_name} {label}**: **{_fmt(val)}** (from {len(series):,} records)")
        summary_parts.append(f"{agg_name} {label} = {_fmt(val)}")

    if not lines:
        return _error("None of the specified columns could be computed.")

    answer = "**Multi-Metric Summary:**\n" + "\n".join(lines)
    return _result(answer, summary=" | ".join(summary_parts), needs_llm=False)


def fn_get_gauge_vs_target(
    df: pd.DataFrame,
    value_col: str,
    target: float,
    agg: str = "sum",
) -> Dict:
    """
    Gauge chart comparing actual aggregated value vs a user-supplied target.
    Use for: 'sales vs target', 'how close are we to our goal', '% of target achieved'.
    """
    if value_col not in df.columns:
        return _error(f"Column '{value_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _error(f"'{value_col}' is not numeric.")

    agg = agg if agg in ["sum", "mean", "count", "max"] else "sum"
    actual = float(getattr(df[value_col].dropna(), agg)())
    target = float(target)
    pct = round(actual / target * 100, 1) if target else 0
    status = "above" if actual >= target else "below"
    gap = abs(actual - target)

    answer = (
        f"**{value_col} vs Target**\n"
        f"- Actual ({agg}): **{_fmt(actual)}**\n"
        f"- Target: **{_fmt(target)}**\n"
        f"- Achievement: **{pct}%** — {status} target by **{_fmt(gap)}**"
    )

    from services.visualizer import DataVisualizer
    chart_json = DataVisualizer.build_gauge(
        actual, target, label=value_col,
        title=f"{value_col}: {_fmt(actual)} vs Target {_fmt(target)}"
    )
    return _result(answer, summary=answer, chart_json=chart_json, needs_llm=False)
