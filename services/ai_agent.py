"""
DataAIAgent — Orchestrator for the 2-Pass Analysis Pipeline
============================================================

Primary path  — Pattern A (Function Calling):
  Pass 1: Schema-only prompt + tool declarations -> Gemini returns function_call { name, args }
  Local:  Python dispatches to pre-written analysis function (full DataFrame, zero API cost)
  Output: Template answer used directly — NO 2nd LLM call for 80%+ of queries
          Only calls LLM for Pass 2 when result.needs_llm = True

Fallback path — Pattern B (Code Generation):
  Triggered when Gemini doesn't pick any declared function
  Gemini writes a pandas snippet; code_executor runs it in a sandboxed environment
  Every fallback question is logged to logs/fallback_queries.jsonl

Token budget per question:
  Pattern A (typical) : Pass 1 ~500-800 tokens | Pass 2 = 0 (template used)
  Pattern A (complex) : Pass 1 ~500-800 tokens | Pass 2 ~300-500 tokens
  Pattern B (fallback): Pass 1 ~800 tokens     | Code-gen call ~400-600 tokens

Gemini NEVER sees raw data rows. Schema only.
"""

import os
import json
import functools
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from services import analysis_functions as af
from services.code_executor import execute_sandboxed, log_fallback_query
from services.visualizer import DataVisualizer
from services.query_logger import (
    log_query_start,
    log_local_route,
    log_gemini_pass1,
    log_pattern_b_code,
    log_api_failure,
    log_heuristic_fallback,
)

# ---------------------------------------------------------------------------
# Tool Declaration Groups
# Declarations are sent to Gemini in Pass 1. Only relevant groups are included
# based on schema detection, keeping token cost low.
# ---------------------------------------------------------------------------

_BASE_TOOLS = [
    {
        "name": "get_missing_values",
        "description": "Count missing/null values per column. Use for: missing data, null check, completeness.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_duplicates",
        "description": "Count duplicate rows in the dataset.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_data_overview",
        "description": "High-level summary: shape, column types, missing %, duplicate count.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_entities_above_threshold",
        "description": "Find categories/entities (e.g. countries, products, customers) whose total or average metric exceeds or meets a threshold. Use for: 'which countries generated more than $5M', 'products with sales over 100k', 'customers who spent more than $1000'.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {"type": "string", "description": "Categorical column (exact name)"},
                "value_col": {"type": "string", "description": "Numeric column to aggregate (exact name)"},
                "threshold": {"type": "number", "description": "Numeric threshold value (e.g. 5000000 for $5M)"},
                "agg":       {"type": "string", "enum": ["sum", "mean", "count"], "description": "Aggregation function (default: sum)"},
                "op":        {"type": "string", "enum": [">", ">=", "<", "<="], "description": "Comparison operator (default: >)"}
            },
            "required": ["group_col", "value_col", "threshold"]
        }
    },
    {
        "name": "get_top_n",
        "description": (
            "Top or Bottom N items by aggregated value. "
            "Use for: 'which X has highest/lowest Y', 'best/worst performers', 'rank by metric'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {"type": "string", "description": "Categorical column to group by (exact name)"},
                "value_col": {"type": "string", "description": "Numeric column to aggregate (exact name)"},
                "n":         {"type": "integer", "description": "Number of results (default: 10)"},
                "agg":       {"type": "string", "enum": ["sum","mean","count","min","max"], "description": "Aggregation function"},
                "ascending": {"type": "boolean", "description": "true = bottom N, false = top N (default: false)"}
            },
            "required": ["group_col", "value_col", "agg"]
        }
    },
    {
        "name": "get_dual_metric_ranking",
        "description": (
            "Compare entities on two different metrics/aggregations simultaneously (e.g. highest total profit but lowest average profit per order). "
            "Use for: 'highest X but lowest Y', 'compare entity across two metrics', 'best in sales but worst in margin'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_col":  {"type": "string", "description": "Categorical entity column (e.g. Country, Product)"},
                "metric1":    {"type": "string", "description": "First numeric column (exact name)"},
                "agg1":       {"type": "string", "enum": ["sum","mean","count","min","max"], "description": "First aggregation function"},
                "ascending1": {"type": "boolean", "description": "true = lowest, false = highest for metric 1"},
                "metric2":    {"type": "string", "description": "Second numeric column (exact name)"},
                "agg2":       {"type": "string", "enum": ["sum","mean","count","min","max"], "description": "Second aggregation function"},
                "ascending2": {"type": "boolean", "description": "true = lowest, false = highest for metric 2"},
                "top_n":      {"type": "integer", "description": "Max entities to show in breakdown"}
            },
            "required": ["group_col", "metric1", "metric2"]
        }
    },
    {
        "name": "get_category_breakdown",
        "description": "Full ranked breakdown of a metric across all categories. Use for: 'breakdown by X', 'all categories'.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {"type": "string", "description": "Categorical column"},
                "value_col": {"type": "string", "description": "Numeric column"},
                "agg":       {"type": "string", "enum": ["sum","mean","count","min","max"]}
            },
            "required": ["group_col", "value_col", "agg"]
        }
    },
    {
        "name": "get_distribution",
        "description": "Value frequency counts. Use for: 'how many of each', 'most common', 'distribution'.",
        "parameters": {
            "type": "object",
            "properties": {
                "col":   {"type": "string", "description": "Column to count"},
                "top_n": {"type": "integer", "description": "Max values to show (default: 15)"}
            },
            "required": ["col"]
        }
    },
    {
        "name": "get_filtered_summary",
        "description": "Filter rows by column value then aggregate. Use for: 'revenue in East', 'orders from category X'.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_col":   {"type": "string", "description": "Column to filter on"},
                "filter_value": {"type": "string", "description": "Value to match (case-insensitive)"},
                "target_col":   {"type": "string", "description": "Numeric column to aggregate (optional)"},
                "agg":          {"type": "string", "enum": ["sum","mean","count","min","max"]}
            },
            "required": ["filter_col", "filter_value"]
        }
    },
    {
        "name": "get_describe",
        "description": "Descriptive statistics (mean, std, min, max) for numeric columns. Use for: 'statistics', 'average', 'range'.",
        "parameters": {
            "type": "object",
            "properties": {
                "cols": {"type": "array", "items": {"type": "string"}, "description": "Numeric columns (empty = all)"}
            }
        }
    },
    {
        "name": "get_correlation",
        "description": "Pearson correlation between numeric columns. Use for: 'correlation', 'relationship between X and Y'.",
        "parameters": {
            "type": "object",
            "properties": {
                "cols": {"type": "array", "items": {"type": "string"}, "description": "Columns to correlate (empty = all numeric)"}
            }
        }
    },
    {
        "name": "get_outliers",
        "description": "Detect outliers in a numeric column using IQR method. Use for: 'anomalies', 'extreme values'.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "Numeric column to check for outliers"}
            },
            "required": ["col"]
        }
    },
    {
        "name": "compare_segments",
        "description": "Compare two specific segment values on a metric. Use for: 'East vs West', 'A vs B'.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_col": {"type": "string", "description": "Column containing the segments"},
                "val1":        {"type": "string", "description": "First segment value"},
                "val2":        {"type": "string", "description": "Second segment value"},
                "metric_col":  {"type": "string", "description": "Numeric column to compare"},
                "agg":         {"type": "string", "enum": ["sum","mean","count","min","max"]}
            },
            "required": ["segment_col", "val1", "val2", "metric_col", "agg"]
        }
    },
    {
        "name": "get_percentile_breakdown",
        "description": "Percentile breakdown (10th–99th) for a numeric column. Use for: 'quartiles', 'percentile', 'distribution shape'.",
        "parameters": {
            "type": "object",
            "properties": {
                "col": {"type": "string", "description": "Numeric column to analyze"}
            },
            "required": ["col"]
        }
    },
    {
        "name": "get_customer_metrics",
        "description": "Customer revenue breakdown: mean/median revenue per customer, total customer count, spend range, top customers. Use for: 'revenue per customer', 'how much revenue does each customer generate', 'customer spend'.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_col": {"type": "string", "description": "Customer identifier or name column"},
                "value_col":    {"type": "string", "description": "Sales or revenue column"}
            },
            "required": ["customer_col", "value_col"]
        }
    },
    {
        "name": "search_items",
        "description": "Keyword search across text columns. Use for: 'find product X', 'rows containing Y'.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword":     {"type": "string", "description": "Search keyword"},
                "search_cols": {"type": "array", "items": {"type": "string"}, "description": "Columns to search (empty = all text columns)"}
            },
            "required": ["keyword"]
        }
    },
]

_TIME_TOOLS = [
    {
        "name": "get_date_bounds",
        "description": (
            "Find the earliest date, latest date, start year, end year, or overall date range of a date column. "
            "Use for: 'when was the first order', 'earliest date/year', 'when did orders begin', "
            "'latest order date', 'what year did this happen', 'date range of the data'. "
            "Do NOT use get_time_series when the user is only asking for the first date, latest date, or start year."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_col": {"type": "string", "description": "Date or time column (exact name)"},
                "target":   {
                    "type": "string",
                    "enum": ["first", "latest", "both"],
                    "description": "'first' for earliest/first order/start date or year, 'latest' for last/end date, 'both' for full date coverage"
                }
            },
            "required": ["date_col"]
        }
    },
    {
        "name": "get_time_series",
        "description": (
            "Aggregate a value over time periods. Use for: 'trend', 'monthly revenue', 'daily orders', 'over time'. "
            "When the user specifies a date range or specific years (e.g. 'between 2022 and 2023', 'last year', 'in 2023'), "
            "set from_date and to_date accordingly. Use a 4-digit year string (e.g. '2022') when only a year is given."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string", "description": "Date column (exact name)"},
                "value_col": {"type": "string", "description": "Numeric column to aggregate"},
                "freq":      {"type": "string", "enum": ["D","W","ME","QE","YE"], "description": "D=daily,W=weekly,ME=monthly,QE=quarterly,YE=yearly"},
                "agg":       {"type": "string", "enum": ["sum","mean","count"]},
                "from_date": {"type": "string", "description": "Optional start date or year (e.g. '2022' or '2022-01-01'). Set when user specifies a start year/date."},
                "to_date":   {"type": "string", "description": "Optional end date or year (e.g. '2023' or '2023-12-31'). Set when user specifies an end year/date."},
                "n_periods": {"type": "integer", "description": "Optional: limit to last N time periods (e.g. n_periods=12 for last 12 months)."}
            },
            "required": ["date_col", "value_col", "freq", "agg"]
        }
    },
    {
        "name": "get_date_range_summary",
        "description": "Filter by date range then aggregate. Use for: 'Q1 revenue', 'between Jan and Mar', 'this month'.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string"},
                "from_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "to_date":   {"type": "string", "description": "End date YYYY-MM-DD"},
                "value_col": {"type": "string", "description": "Numeric column (optional)"},
                "agg":       {"type": "string", "enum": ["sum","mean","count","min","max"]}
            },
            "required": ["date_col", "from_date", "to_date"]
        }
    },
    {
        "name": "get_growth_rate",
        "description": (
            "Period-over-period growth rate. Use for: 'MoM growth', 'YoY change', 'is X growing?'. "
            "When the user specifies a date range or specific years, set from_date and to_date accordingly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string"},
                "value_col": {"type": "string"},
                "freq":      {"type": "string", "enum": ["D","W","ME","QE","YE"]},
                "from_date": {"type": "string", "description": "Optional start date or year (e.g. '2022')."},
                "to_date":   {"type": "string", "description": "Optional end date or year (e.g. '2023')."},
                "n_periods": {"type": "integer", "description": "Optional: limit to last N periods."}
            },
            "required": ["date_col", "value_col", "freq"]
        }
    },
    {
        "name": "get_monthly_pattern",
        "description": "Average by calendar month (seasonal pattern). Use for: 'peak month', 'seasonality', 'busiest month'.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string"},
                "value_col": {"type": "string"}
            },
            "required": ["date_col", "value_col"]
        }
    },
]

_FINANCIAL_TOOLS = [
    {
        "name": "get_profit_margin",
        "description": "Profit margin % = (revenue - cost) / revenue. Use for: 'margin', 'profitability', 'most profitable X'.",
        "parameters": {
            "type": "object",
            "properties": {
                "cost_col":    {"type": "string", "description": "Cost price column"},
                "revenue_col": {"type": "string", "description": "Selling price / revenue column"},
                "group_col":   {"type": "string", "description": "Optional: group by this column"}
            },
            "required": ["cost_col", "revenue_col"]
        }
    },
]

_INVENTORY_TOOLS = [
    {
        "name": "get_low_stock",
        "description": "Find items at or below a stock threshold. Use for: 'low stock', 'reorder alert', 'stock below X'.",
        "parameters": {
            "type": "object",
            "properties": {
                "qty_col":   {"type": "string", "description": "Quantity/stock column"},
                "threshold": {"type": "number", "description": "Alert below this units (default: 10)"},
                "name_col":  {"type": "string", "description": "Product name column for labels (optional)"}
            },
            "required": ["qty_col"]
        }
    },
    {
        "name": "get_inventory_value",
        "description": "Total inventory value = quantity x cost. Use for: 'stock worth', 'inventory capital', 'value by category'.",
        "parameters": {
            "type": "object",
            "properties": {
                "qty_col":  {"type": "string", "description": "Quantity/stock column"},
                "cost_col": {"type": "string", "description": "Cost price per unit column"},
                "group_col": {"type": "string", "description": "Optional grouping column"}
            },
            "required": ["qty_col", "cost_col"]
        }
    },
]

_ADVANCED_TOOLS = [
    {
        "name": "get_pareto",
        "description": "80/20 Pareto analysis — bar chart of sorted values + cumulative % line. Use for: 'which items generate 80% of revenue', 'pareto chart', '80/20 analysis'.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {"type": "string", "description": "Categorical column to group by"},
                "value_col": {"type": "string", "description": "Numeric column to aggregate"},
                "n":         {"type": "integer", "description": "Max categories to display (default: 20)"}
            },
            "required": ["group_col", "value_col"]
        }
    },
    {
        "name": "get_treemap",
        "description": "Hierarchical treemap chart showing nested category contributions to a total. Use for: 'treemap', 'hierarchical breakdown', 'nested categories'.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_cols": {"type": "array", "items": {"type": "string"}, "description": "1 or 2 categorical columns for hierarchy"},
                "value_col":  {"type": "string", "description": "Numeric column to size rectangles by"}
            },
            "required": ["group_cols", "value_col"]
        }
    },
    {
        "name": "get_waterfall",
        "description": "Waterfall / bridge chart showing positive/negative category contributions to a total. Use for: 'waterfall chart', 'revenue bridge', 'what drives profit/loss'.",
        "parameters": {
            "type": "object",
            "properties": {
                "label_col": {"type": "string", "description": "Categorical column for step labels"},
                "value_col": {"type": "string", "description": "Numeric column for step values"}
            },
            "required": ["label_col", "value_col"]
        }
    },
    {
        "name": "get_funnel",
        "description": "Funnel chart for pipeline/conversion stage analysis. Use for: 'conversion funnel', 'stage drop-off', 'pipeline chart'.",
        "parameters": {
            "type": "object",
            "properties": {
                "stage_col": {"type": "string", "description": "Column containing pipeline stage names"}
            },
            "required": ["stage_col"]
        }
    },
    {
        "name": "get_heatmap_cross",
        "description": "2D Cross-tabulation heatmap of two categorical dimensions weighted by a metric. Use for: 'heatmap', 'crosstab', 'matrix of X by Y'.",
        "parameters": {
            "type": "object",
            "properties": {
                "row_col":   {"type": "string", "description": "Categorical column for Y-axis (rows)"},
                "col_col":   {"type": "string", "description": "Categorical column for X-axis (columns)"},
                "value_col": {"type": "string", "description": "Numeric column to aggregate"},
                "agg":       {"type": "string", "enum": ["sum", "mean", "count"]}
            },
            "required": ["row_col", "col_col", "value_col"]
        }
    },
    {
        "name": "get_rolling_average",
        "description": "Rolling / moving average trend line. Use for: 'rolling 30-day sales', '7-day moving average', 'smoothed trend'.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":  {"type": "string", "description": "Date column"},
                "value_col": {"type": "string", "description": "Numeric column"},
                "window":    {"type": "integer", "description": "Rolling window size in days (e.g. 7 or 30)"}
            },
            "required": ["date_col", "value_col"]
        }
    },
    {
        "name": "get_bubble_data",
        "description": "Bubble chart comparing 3 numeric variables (X vs Y sized by Z). Use for: 'bubble chart', '3-variable chart'.",
        "parameters": {
            "type": "object",
            "properties": {
                "x_col":    {"type": "string", "description": "Numeric column for X-axis"},
                "y_col":    {"type": "string", "description": "Numeric column for Y-axis"},
                "size_col": {"type": "string", "description": "Numeric column for bubble size"},
                "cat_col":  {"type": "string", "description": "Optional categorical column for color"}
            },
            "required": ["x_col", "y_col", "size_col"]
        }
    },
    {
        "name": "get_grouped_comparison",
        "description": "Grouped or stacked bar chart comparing a metric across two categorical dimensions. Use for: 'grouped bar', 'stacked bar', 'compare X by Y'.",
        "parameters": {
            "type": "object",
            "properties": {
                "x_col":     {"type": "string", "description": "Primary categorical column (X-axis)"},
                "group_col": {"type": "string", "description": "Secondary categorical column (group/stack)"},
                "value_col": {"type": "string", "description": "Numeric column to aggregate"}
            },
            "required": ["x_col", "group_col", "value_col"]
        }
    },
    {
        "name": "get_cohort_retention",
        "description": "Monthly customer retention cohort heatmap. Use for: 'cohort retention', 'repeat customer rate by cohort'.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_col":     {"type": "string", "description": "Date column"},
                "customer_col": {"type": "string", "description": "Customer ID / unique identifier column"}
            },
            "required": ["date_col", "customer_col"]
        }
    },
    {
        "name": "get_gauge_vs_target",
        "description": "Gauge chart comparing an aggregated metric against a target. Use for: 'sales vs target', 'goal achievement %'.",
        "parameters": {
            "type": "object",
            "properties": {
                "value_col": {"type": "string", "description": "Numeric column"},
                "target":    {"type": "number", "description": "Target numeric goal"},
                "agg":       {"type": "string", "enum": ["sum", "mean", "count"]}
            },
            "required": ["value_col", "target"]
        }
    },
    {
        "name": "get_aggregate",
        "description": "Simple scalar aggregation (single number answer, no chart). Use for: 'total net sales', 'grand total revenue', 'average order value', 'how many orders in total'.",
        "parameters": {
            "type": "object",
            "properties": {
                "col":   {"type": "string", "description": "Numeric column to aggregate"},
                "agg":   {"type": "string", "enum": ["sum", "mean", "count", "min", "max", "median"]},
                "label": {"type": "string", "description": "Display label"}
            },
            "required": ["col", "agg"]
        }
    },
]

# ---------------------------------------------------------------------------
# Function name -> analysis_functions dispatcher
# ---------------------------------------------------------------------------

_FUNCTION_MAP = {
    "get_missing_values":      lambda df, **kw: af.fn_get_missing_values(df),
    "get_duplicates":          lambda df, **kw: af.fn_get_duplicates(df),
    "get_data_overview":       lambda df, **kw: af.fn_get_data_overview(df),
    "get_top_n":               lambda df, **kw: af.fn_get_top_n(df, **kw),
    "get_category_breakdown":  lambda df, **kw: af.fn_get_category_breakdown(df, **kw),
    "get_distribution":        lambda df, **kw: af.fn_get_distribution(df, **kw),
    "get_filtered_summary":    lambda df, **kw: af.fn_get_filtered_summary(df, **kw),
    "get_describe":            lambda df, **kw: af.fn_get_describe(df, **kw),
    "get_correlation":         lambda df, **kw: af.fn_get_correlation(df, **kw),
    "get_outliers":            lambda df, **kw: af.fn_get_outliers(df, **kw),
    "compare_segments":        lambda df, **kw: af.fn_compare_segments(df, **kw),
    "get_percentile_breakdown":lambda df, **kw: af.fn_get_percentile_breakdown(df, **kw),
    "get_single_percentile":   lambda df, **kw: af.fn_get_single_percentile(df, **kw),
    "get_pct_above_threshold": lambda df, **kw: af.fn_get_pct_above_threshold(df, **kw),
    "get_entities_above_threshold": lambda df, **kw: af.fn_get_entities_above_threshold(df, **kw),
    "get_dual_metric_ranking": lambda df, **kw: af.fn_dual_metric_ranking(df, **kw),
    "get_median":              lambda df, **kw: af.fn_get_median(df, **kw),
    "search_items":            lambda df, **kw: af.fn_search_items(df, **kw),
    # Scalar aggregation (no chart)
    "get_aggregate":           lambda df, **kw: af.fn_get_aggregate(df, **kw),
    "get_multi_aggregate":     lambda df, **kw: af.fn_get_multi_aggregate(df, **kw),
    "count_unique":            lambda df, **kw: af.fn_count_unique(df, **kw),
    # Advanced charts
    "get_pareto":              lambda df, **kw: af.fn_get_pareto(df, **kw),
    "get_heatmap_cross":       lambda df, **kw: af.fn_get_heatmap_cross(df, **kw),
    "get_treemap":             lambda df, **kw: af.fn_get_treemap(df, **kw),
    "get_funnel":              lambda df, **kw: af.fn_get_funnel(df, **kw),
    "get_waterfall":           lambda df, **kw: af.fn_get_waterfall(df, **kw),
    "get_rolling_average":     lambda df, **kw: af.fn_get_rolling_average(df, **kw),
    "get_bubble_data":         lambda df, **kw: af.fn_get_bubble_data(df, **kw),
    "get_grouped_comparison":  lambda df, **kw: af.fn_get_grouped_comparison(df, **kw),
    "get_cohort_retention":    lambda df, **kw: af.fn_get_cohort_retention(df, **kw),
    "get_gauge_vs_target":     lambda df, **kw: af.fn_get_gauge_vs_target(df, **kw),
    # Time-based
    "get_date_bounds":         lambda df, **kw: af.fn_get_date_bounds(df, **kw),
    "get_time_series":         lambda df, **kw: af.fn_get_time_series(df, **kw),
    "get_date_range_summary":  lambda df, **kw: af.fn_get_date_range_summary(df, **kw),
    "get_growth_rate":         lambda df, **kw: af.fn_get_growth_rate(df, **kw),
    "get_monthly_pattern":     lambda df, **kw: af.fn_get_monthly_pattern(df, **kw),
    # Financial / Customer
    "get_profit_margin":       lambda df, **kw: af.fn_get_profit_margin(df, **kw),
    "get_customer_metrics":    lambda df, **kw: af.fn_get_customer_metrics(df, **kw),
    # Inventory
    "get_low_stock":           lambda df, **kw: af.fn_get_low_stock(df, **kw),
    "get_inventory_value":     lambda df, **kw: af.fn_get_inventory_value(df, **kw),
}


# ---------------------------------------------------------------------------
# Multilingual helpers
# ---------------------------------------------------------------------------

def _is_indic_script(text: str) -> bool:
    """
    Returns True if the text contains characters from:
      - Devanagari (Hindi): U+0900–U+097F
      - Bengali script:     U+0980–U+09FF
    This is used to decide whether to force Pass 2 so the answer
    is returned in the same language and script as the user's question.
    """
    return any('\u0900' <= ch <= '\u097F' or '\u0980' <= ch <= '\u09FF' for ch in text)


# ---------------------------------------------------------------------------
# Module-level LRU answer cache
# Persists across requests within a Flask worker process.
# Key: (df.shape, tuple(columns), normalized_question) → answer dict
# Capped at 256 entries to prevent unbounded memory growth.
# ---------------------------------------------------------------------------
class _BoundedDict(dict):
    """A plain dict that evicts the oldest entry when capacity is exceeded."""
    def __init__(self, maxsize: int = 256):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        if len(self) >= self._maxsize:
            # Evict the oldest key (insertion order in Python 3.7+)
            oldest = next(iter(self))
            del self[oldest]
        super().__setitem__(key, value)

_ANSWER_CACHE: _BoundedDict = _BoundedDict(maxsize=256)


# ===========================================================================
# Main Agent Class
# ===========================================================================

class DataAIAgent:
    """
    Orchestrates the Pattern A / Pattern B hybrid pipeline.

    Pattern A (primary):
      1. Build compact schema string (cached from upload time)
      2. Gemini function calling: schema + question -> { name, args }
      3. Python executes the named function on the full DataFrame
      4. Return template answer directly (no 2nd LLM call)
      5. Only call LLM for Pass 2 when result.needs_llm = True

    Pattern B (fallback — logged):
      1. Gemini generates a pandas code snippet
      2. Sandboxed executor runs it (read-only df, whitelisted builtins, 8s timeout)
      3. Question logged to logs/fallback_queries.jsonl
    """

    def __init__(self, df: pd.DataFrame, cached_schema: Optional[str] = None):
        """
        Args:
            df:            Full DataFrame (loaded from upload).
            cached_schema: Pre-built schema string computed once at upload time.
                           Avoids recomputing on every chat request.
        """
        self.df = df
        self._schema_cache = cached_schema
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.has_genai = False
        self.model = None
        self._genai = None
        self._model_with_tools = None   # Cached tools-equipped model (rebuilt only when tool list changes)
        self._cached_tools_key = None   # Hash of the tool set used to build _model_with_tools

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        # Pre-compute column role groups once at upload — used by the local intent router
        self._col_roles = self._build_col_roles()

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self._genai = genai
                self.has_genai = True
            except Exception:
                self.has_genai = False

    # ─────────────────────────────────────────────────────────────────────────
    # Column Role Heuristics — Built Once Per Dataset
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Column Role Heuristics — Built Once Per Dataset
    # ─────────────────────────────────────────────────────────────────────────

    def _build_col_roles(self) -> Dict[str, Any]:
        """
        Classify columns into semantic roles using name-keyword matching.
        Called once at __init__ time; results reused by the local intent router.
        Filters out ID / Key / Zipcode / Status / Risk columns from numeric metrics.
        """
        import re
        all_num   = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = [c for c in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[c]) and not pd.api.types.is_datetime64_any_dtype(self.df[c])]
        date_cols = [c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])]
        # Also try string columns that look like dates and can be parsed as timestamps
        for c in cat_cols:
            if c not in date_cols and any(k in c.lower() for k in ["date", "time", "day", "year", "month", "period"]):
                try:
                    sample = self.df[c].dropna().head(10)
                    if not sample.empty:
                        parsed = pd.to_datetime(sample, errors="coerce")
                        if parsed.notna().sum() >= len(sample) * 0.5:
                            date_cols.append(c)
                except Exception:
                    pass

        # Filter out ID / Code / Key / Zip / Phone / Lat / Lon / Unnamed / Status / Risk from metric pools
        id_keywords = ["id", "code", "zip", "key", "num", "number", "ssn", "phone", "lat", "lon", "latitude", "longitude", "unnamed", "status", "risk"]
        metric_num_cols = [
            c for c in all_num
            if not any(re.search(rf"\b{kw}\b", c.lower().replace("_", " ").replace("-", " ")) or kw in c.lower() for kw in id_keywords)
        ]
        num_cols = metric_num_cols if metric_num_cols else all_num

        def _pick_best(cols, exact_candidates, partial_keywords):
            for cand in exact_candidates:
                for c in cols:
                    if c.lower().replace("_", " ").strip() == cand.lower().strip() or c.lower().strip() == cand.lower().strip():
                        return c
            for cand in exact_candidates:
                for c in cols:
                    if cand.lower() in c.lower().replace("_", " "):
                        return c
            for kw in partial_keywords:
                for c in cols:
                    if re.search(rf"\b{kw}\b", c.lower().replace("_", " ").replace("-", " ")):
                        return c
            return None

        return {
            "num":             num_cols,
            "all_num":         all_num,
            "cat":             cat_cols,
            "date":            date_cols,
            # Identifiers
            "customer_id":     _pick_best(self.df.columns, ["customer id", "customer_id", "customerid", "client id", "buyer id"], ["customer_id", "customerid"]),
            "product_id":      _pick_best(self.df.columns, ["product card id", "product id", "product_id", "item id", "sku"], ["product_id", "productcardid"]),
            "order_id":        _pick_best(self.df.columns, ["order id", "order_id", "orderid"], ["order_id", "orderid"]),
            # Entities
            "customer":        _pick_best(cat_cols, ["customer name", "customer_name", "customer full name", "customer fname", "customer"], ["customer", "client", "buyer"]),
            "product":         _pick_best(cat_cols, ["product name", "product_name", "product title", "product"], ["product", "item", "sku"]),
            "category":        _pick_best(cat_cols, ["category name", "category_name", "product category", "category"], ["category", "dept", "department"]),
            "segment":         _pick_best(cat_cols, ["customer segment", "customer_segment", "segment", "client segment"], ["segment"]),
            "shipping_mode":   _pick_best(cat_cols, ["shipping mode", "shipping_mode", "ship mode", "delivery mode"], ["shipping mode", "ship mode"]),
            "order_status":    _pick_best(cat_cols, ["order status", "order_status"], ["order status"]),
            "delivery_status": _pick_best(cat_cols, ["delivery status", "delivery_status"], ["delivery status"]),
            "department":      _pick_best(cat_cols, ["department name", "department_name", "department"], ["department", "dept"]),
            "region":          _pick_best(cat_cols, ["order region", "customer region", "region"], ["region", "territory", "zone"]),
            "country":         _pick_best(cat_cols, ["order country", "customer country", "country"], ["country", "nation"]),
            "state":           _pick_best(cat_cols, ["order state", "customer state", "state"], ["state", "province"]),
            "city":            _pick_best(cat_cols, ["order city", "customer city", "city"], ["city", "town"]),
            "market":          _pick_best(cat_cols, ["market"], ["market"]),
            "date_col":        _pick_best(date_cols, ["order date", "order_date", "shipping date", "date"], ["order", "date", "transaction", "created", "time"]),
            # Metrics
            "sales":           _pick_best(num_cols, ["sales", "total sales", "revenue", "order item total", "item total"], ["sale", "revenue", "turnover"]),
            "profit":          _pick_best(num_cols, ["order profit per order", "order profit", "profit", "net profit", "total profit"], ["profit", "benefit"]),
            "profit_ratio":    _pick_best(num_cols, ["order item profit ratio", "profit ratio", "profit margin", "margin ratio", "profit_margin", "profit %", "margin %"], ["ratio", "margin"]),
            "quantity":        _pick_best(num_cols, ["order item quantity", "quantity", "units", "qty"], ["quantity", "qty", "units"]),
            "discount":        _pick_best(num_cols, ["order item discount", "discount", "order item discount rate"], ["discount", "rebate"]),
            "cost":            _pick_best(num_cols, ["product price", "cost", "expense", "spend", "cogs"], ["cost", "expense"]),
            "delivery":        _pick_best(num_cols, ["days for shipping (real)", "delays for shipping", "days for shipment (scheduled)"], ["shipping", "delivery", "days"]),
            "payment":         _pick_best(cat_cols, ["payment method", "payment type", "payment_method", "payment_type", "payment mode", "payment", "type"], ["payment", "pay_mode", "pay_type", "pay"]),
            # Roles for advanced charts
            "stage":           _pick_best(cat_cols, ["order status", "delivery status", "stage", "step", "status"], ["stage", "status", "phase"]),
            "hour":            _pick_best(num_cols + cat_cols, ["order hour", "hour", "hh"], ["hour"]),
            "rating":          _pick_best(num_cols, ["rating", "score", "satisfaction"], ["rating", "score"]),
            "target":          _pick_best(num_cols, ["target", "goal", "budget"], ["target", "goal"]),
            "second_num":      num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None),
            "second_cat":      cat_cols[1] if len(cat_cols) > 1 else (cat_cols[0] if cat_cols else None),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Compound Query Detector
    # ─────────────────────────────────────────────────────────────────────────

    def _is_compound_question(self, question: str) -> bool:
        """
        Identifies multi-clause, contrasting, dual-extreme, comparative, or feature-engineering queries
        that require multi-metric aggregation, custom column calculations, or sandboxed code execution.
        """
        import re
        q = question.lower().strip()

        # 1. Direct contrast conjunctions and compound connectors
        if re.search(r"\b(but|yet|however|whereas|while having|while also|and also|along with having|having the (highest|lowest|best|worst)|with the (highest|lowest|best|worst)|and the (highest|lowest|best|worst)|and lowest|and highest|but lowest|but highest)\b", q):
            return True

        # 2. Dual contrasting extremes (highest vs lowest)
        has_high = bool(re.search(r"\b(highest|top|maximum|most|greatest|best|leading|winner)\b", q))
        has_low  = bool(re.search(r"\b(lowest|bottom|minimum|least|smallest|worst|poorest)\b", q))
        if has_high and has_low:
            return True

        # 3. Dual condition filters & mathematical operators
        if re.search(r"\b(where|with|having|for)\b.+(\band\b|\bor\b|\bbut\b).+(>=|<=|>|<|==|!=|\babove\b|\bbelow\b|\bmore than\b|\bless than\b|\bgreater than\b)", q):
            return True

        # 4. A vs B comparisons (versus / vs / compare X versus Y)
        if re.search(r"\b(versus|vs\.?|compare\b.+\bversus\b|comparing\b.+\band\b.+\b(for each|by|across)\b)\b", q):
            return True

        # 5. Feature Engineering: Column math, differences, lead time, delays, duration, custom ratios
        if re.search(r"\b(subtracting|subtract|difference between|delay by|delay between|duration between|lead time|leadtime|time between|days between|hours between|gap between|ratio of\b.+\bto\b|divided by|product of|multiplying)\b", q):
            return True

        # 6. Feature Engineering: Segmentation, bucketing, classification, scoring, cohort analysis
        if re.search(r"\b(classify|categorize|segment|bucket|cohort|rfm|tier|rank based on|based on\b.+\band\b.+\b(profit|sales|margin|spend|frequency|delay|ratio))\b", q):
            return True

        # 7. Multi-metric composite ranking (e.g. based on sales and profit ratio)
        if re.search(r"\b(based on|considering|combining)\b.+\band\b.+\b(ratio|margin|score|performance|profit|sales|growth)\b", q):
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Local Intent Router — Zero Gemini API Calls
    # ─────────────────────────────────────────────────────────────────────────

    def _local_intent_router(self, question: str) -> Optional[tuple]:
        """
        Pattern-match common questions locally and return (fn_name, fn_args) tuples.
        Returns None if the question doesn't match — caller then falls through to Gemini Pass 1.
        """
        import re
        q = question.lower().strip()
        roles = self._col_roles

        def _stem(w):
            w = w.lower().strip()
            if w.endswith("ies") and len(w) > 4:
                return w[:-3] + "y"
            if w.endswith("delivery") or w.endswith("deliveries") or w.endswith("delivered") or w.endswith("delivering"):
                return "deliver"
            if w.endswith("es") and len(w) > 4 and not w.endswith("ses"):
                return w[:-2]
            if w.endswith("ed") and len(w) > 4:
                return w[:-2]
            if w.endswith("ing") and len(w) > 5:
                return w[:-3]
            if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
                return w[:-1]
            return w

        q_words = re.findall(r"\b\w+\b", q)
        q_stemmed = {_stem(w) for w in q_words}
        entity_stopwords = {"product", "item", "order", "card", "customer", "user", "line", "detail", "entry", "record", "per"}

        def _find_col_in_q(col_list, fallback=None, is_num=False):
            """Find the first column whose name appears in the question, prioritizing semantic metrics when is_num=True."""
            if not col_list:
                return fallback

            if is_num:
                if any(w in q_stemmed for w in ["revenue", "sales", "sale", "income", "earning", "turnover"]):
                    target = next((c for c in col_list if any(k in c.lower() for k in ["sale", "revenue", "total"])), None)
                    if target: return target
                if any(w in q_stemmed for w in ["profit", "margin", "net", "gain", "benefit"]):
                    target = next((c for c in col_list if any(k in c.lower() for k in ["profit", "margin", "benefit"])), None)
                    if target: return target
                if any(w in q_stemmed for w in ["quantity", "qty", "volume", "units", "items"]):
                    target = next((c for c in col_list if any(k in c.lower() for k in ["qty", "quantity", "units", "count"])), None)
                    if target: return target
                if any(w in q_stemmed for w in ["discount", "discounts", "rebate"]):
                    target = next((c for c in col_list if "discount" in c.lower()), None)
                    if target: return target
                if any(w in q_stemmed for w in ["price", "pricing"]):
                    target = next((c for c in col_list if "price" in c.lower()), None)
                    if target: return target
                if any(w in q_stemmed for w in ["cost", "expense", "cogs"]):
                    target = next((c for c in col_list if any(k in c.lower() for k in ["cost", "expense"])), None)
                    if target: return target

            for col in col_list:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                words = [_stem(w) for w in re.split(r"[\s_\-]+", c_clean) if len(w) > 2 and (not is_num or w not in entity_stopwords)]
                if any(w in q_stemmed or any(w in qw or qw in w for qw in q_stemmed if len(qw) > 3) for w in words):
                    return col
            return fallback

        def _extract_target_group_col(text):
            t_match = re.search(r'\b(?:which|what|who|name the|tell me the|top|best|worst|lowest)\s+([a-z\s]+?)(?:\s+(?:generated|generates|produced|produces|has|have|had|with|by|in|for|among|from|is|was|leads|ranked|ranks|got)\b|\?|$)', text)
            target_phrase = t_match.group(1).strip() if t_match else ''

            def _map_phrase(phrase):
                if re.search(r'\b(payment\s*method|payment\s*type|payment\s*mode|payment\s*option|payment\s*channel|payment|pay\s*method|pay\s*type)\b', phrase):
                    return roles.get('payment') or next((c for c in self.df.columns if "payment" in c.lower() or "pay" in c.lower()), None)
                if re.search(r'\b(shipping\s*mode|ship\s*mode|delivery\s*mode|shipping\s*type)\b', phrase):
                    return roles.get('shipping_mode') or 'Shipping Mode'
                if re.search(r'\b(customer\s*segment|client\s*segment|segment)\b', phrase):
                    return roles.get('segment') or 'Customer Segment'
                if re.search(r'\b(product\s*name|product|item\s*name|item|sku|goods)\b', phrase):
                    return roles.get('product') or 'Product Name'
                if re.search(r'\b(category\s*name|category|product\s*category)\b', phrase):
                    return roles.get('category') or 'Category Name'
                if re.search(r'\b(department\s*name|department|dept)\b', phrase):
                    return roles.get('department') or 'Department Name'
                if re.search(r'\b(order\s*status)\b', phrase):
                    return roles.get('order_status') or 'Order Status'
                if re.search(r'\b(delivery\s*status)\b', phrase):
                    return roles.get('delivery_status') or 'Delivery Status'
                if re.search(r'\b(customer|buyer|client|shopper)\b', phrase):
                    return roles.get('customer') or 'Customer Fname'
                if re.search(r'\b(region|territory|zone)\b', phrase):
                    return roles.get('region') or 'Order Region'
                if re.search(r'\b(country|nation)\b', phrase):
                    return roles.get('country') or 'Order Country'
                if re.search(r'\b(state|province)\b', phrase):
                    return roles.get('state') or 'Order State'
                if re.search(r'\b(city|town)\b', phrase):
                    return roles.get('city') or 'Order City'
                if re.search(r'\b(market)\b', phrase):
                    return roles.get('market') or 'Market'
                return None

            mapped = _map_phrase(target_phrase)
            if mapped and mapped in self.df.columns:
                return mapped
            mapped_q = _map_phrase(text)
            if mapped_q and mapped_q in self.df.columns:
                return mapped_q
            return _find_col_in_q(roles["cat"]) or roles.get("product") or roles.get("category") or roles.get("region")

        def _extract_metric_col(text):
            if re.search(r'\b(profit\s*ratio|profit\s*margin|margin\s*ratio|margin|profitability)\b', text):
                return roles.get('profit_ratio') or roles.get('profit') or _find_col_in_q(roles["num"], is_num=True)
            if re.search(r'\b(profit|net profit|earnings|gain)\b', text):
                return roles.get('profit') or _find_col_in_q(roles["num"], is_num=True)
            if re.search(r'\b(order value|aov|sales|revenue|income|turnover|net sales|spending|spend|amount|order total)\b', text):
                return roles.get('sales') or _find_col_in_q(roles["num"], is_num=True)
            if re.search(r'\b(quantity|qty|volume|units)\b', text):
                return roles.get('quantity') or _find_col_in_q(roles["num"], is_num=True)
            if re.search(r'\b(discount|rebate)\b', text):
                return roles.get('discount') or _find_col_in_q(roles["num"], is_num=True)
            if re.search(r'\b(delivery time|shipping days|lead time|shipping duration|days for shipping|delay)\b', text):
                return roles.get('delivery') or _find_col_in_q(roles["num"], is_num=True)
            return _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)

        def _parse_amount(text):
            """Parse amounts with multipliers like $5 million, 5M, 500k, 5,000,000, zero, nil, negative values."""
            clean_text = text.replace(",", "")
            if re.search(r"\b(zero|nil|null|none)\b", clean_text, re.IGNORECASE):
                return 0.0
            m = re.search(r'[\$€£₹]?\s*(-?\d+(?:\.\d+)?)\s*(k|thousand|m|million|b|billion|lakh|crore)?\b', clean_text, re.IGNORECASE)
            if not m:
                return None
            val = float(m.group(1))
            unit = (m.group(2) or '').lower()
            if unit in ['k', 'thousand']:
                val *= 1_000
            elif unit in ['m', 'million']:
                val *= 1_000_000
            elif unit in ['b', 'billion']:
                val *= 1_000_000_000
            elif unit == 'lakh':
                val *= 100_000
            elif unit == 'crore':
                val *= 10_000_000
            return val

        def _extract_filter(text, exclude_col=None):
            filter_stopwords = {"payment", "method", "type", "mode", "status", "order", "product", "item", "category", "customer", "segment", "region", "country", "city", "state", "sales", "profit", "discount", "price"}
            for col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    if col != exclude_col:
                        unique_vals = self.df[col].dropna().unique()
                        if len(unique_vals) <= 100:
                            for v in unique_vals:
                                v_str = str(v).strip().lower()
                                if len(v_str) > 2 and v_str not in filter_stopwords and re.search(rf'\b{re.escape(v_str)}\b', text):
                                    return col, str(v).strip()
            return None, None

        # ── 00. COMPOUND / DUAL-METRIC QUERIES ──────────────────────────────
        if self._is_compound_question(question):
            # When AI key is available, return None so Pattern B code generation can handle it fully
            if self.has_genai:
                return None

            # In zero-API / fallback mode: attempt dual-metric ranking
            grp_col = _extract_target_group_col(q)
            if grp_col:
                parts = re.split(r"\b(but|while having|while also|and also|along with having|whereas|yet|and)\b", q, maxsplit=1)
                part1 = parts[0] if parts else q
                part2 = parts[-1] if len(parts) > 1 else q

                has_low_first = bool(re.search(r"\b(lowest|bottom|minimum|least|worst)\b", part1))
                has_low_second = bool(re.search(r"\b(lowest|bottom|minimum|least|worst)\b", part2))

                asc1 = True if has_low_first else False
                asc2 = True if has_low_second else False

                m1 = _extract_metric_col(part1) or roles.get("profit") or roles.get("sales")
                m2 = _extract_metric_col(part2) or (roles.get("sales") if m1 != roles.get("sales") else roles.get("profit")) or m1

                agg1 = "mean" if re.search(r"\b(average|avg|mean|per order|per item|aov)\b", part1) else "sum"
                agg2 = "mean" if re.search(r"\b(average|avg|mean|per order|per item|aov)\b", part2) else "sum"

                if m1 and m2 and m1 in self.df.columns and m2 in self.df.columns:
                    return ("get_dual_metric_ranking", {
                        "group_col": grp_col,
                        "metric1": m1,
                        "agg1": agg1,
                        "ascending1": asc1,
                        "metric2": m2,
                        "agg2": agg2,
                        "ascending2": asc2,
                    })
            return None

        # ── 0a. GROUPED ENTITY THRESHOLD QUERIES ─────────────────────────────
        # "Which countries have generated more than $5 million in sales?",
        # "products with sales above 100k", "which customers spent more than $500",
        # "Which products have a profit ratio below zero?", "products with negative profit", "loss making products"
        is_threshold_q = bool(
            re.search(r"\b(which|what|who|list|show|find|name)\b.*\b(more than|greater than|above|over|exceed|exceeding|at least|higher than|less than|below|under|negative|loss|unprofitable)\b", q) and
            re.search(r"\b(countries|country|products?|items?|customers?|segments?|markets?|regions?|cities|city|states?|departments?|brands?|categories|category|buyers?|clients?)\b", q)
        )
        if is_threshold_q or re.search(r"\b(profit\s*ratio|margin|profit)\b.*\b(below|under|<|less than|negative|zero)\b", q) or re.search(r"\b(below zero|less than 0|under 0|negative)\b", q):
            thresh = _parse_amount(q)
            if thresh is None and re.search(r"\b(negative|loss|unprofitable|below zero|less than 0|under 0)\b", q):
                thresh = 0.0

            if thresh is not None:
                grp_col = _extract_target_group_col(q)
                val_col = _extract_metric_col(q)
                if grp_col and val_col:
                    agg_type = "mean" if re.search(r"\b(average|avg|mean|per order|per item|aov|ratio|margin)\b", q) else "sum"
                    op = "<" if re.search(r"\b(less than|below|under|negative|loss|unprofitable|<|<=)\b", q) else ">"
                    return ("get_entities_above_threshold", {
                        "group_col": grp_col,
                        "value_col": val_col,
                        "threshold": thresh,
                        "agg": agg_type,
                        "op": op
                    })

        # ── 5. TOP N & RANKING queries ─────────────────────────────────────────
        # "Which market has the highest total profit?" -> n = 1 (single winner, direct text, no chart)
        # "Which customer segment has the highest average order value?" -> n = 1
        # "Top 5 products by sales" -> n = 5 (rankings + chart)
        if re.search(r"\b(top|best|highest|most|leading|greatest|maximum|winner|leads|peak)\b", q) and \
                not re.search(r"\b(recent|latest|newest|last order|last date|last transaction|first order|earliest|oldest|when|date)\b", q) and \
                not re.search(r"\b(what is the (maximum|minimum|highest|lowest|max|min)|what is our (maximum|minimum)|what('s| is) the (max|min))\b", q):
            n_match = re.search(r"\b(?:top|best|highest|most)\s+(\d+)\b", q)
            if not n_match:
                n_match = re.search(r"\b(\d+)\s+(?:top|best|highest|products|items|categories|customers|regions|brands|performers)\b", q)
            if n_match:
                n = int(n_match.group(1 if len(n_match.groups()) == 1 else 2))
            else:
                is_singular = (
                    bool(re.search(r"\b(which|who|what is the|what's the|name the|tell me the|find the)\s+(?:single\s+)?(product|item|category|region|customer|buyer|client|city|state|country|market|segment|department|branch|store|brand|supplier|channel|one)\b", q)) or
                    bool(re.search(r"\b(top|best|highest|most|leading)\s+(product|item|category|region|customer|buyer|client|city|state|country|market|segment|department|branch|store|brand|supplier|channel)\b", q)) or
                    bool(re.search(r"\b(which|who|what)\b.*\b(highest|most|best|top|leading)\b", q))
                ) and not re.search(r"\b(products|items|categories|regions|customers|buyers|clients|cities|states|countries|markets|segments|departments|branches|stores|brands|suppliers|channels|ranking|rankings|list|all|table|chart|compare|comparison|gap)\b", q)
                n = 1 if is_singular else 10

            group_col = _extract_target_group_col(q)
            value_col = _extract_metric_col(q)
            filter_col, filter_val = _extract_filter(q, exclude_col=group_col)
            agg_type = "mean" if re.search(r"\b(average|avg|mean|per order|per customer|per item|aov)\b", q) else "sum"
            if group_col and value_col:
                return ("get_top_n", {
                    "group_col": group_col,
                    "value_col": value_col,
                    "n": n,
                    "agg": agg_type,
                    "ascending": False,
                    "filter_col": filter_col,
                    "filter_val": filter_val,
                })

        # ── 6. BOTTOM / WORST / LOWEST queries ──────────────────────────────
        if re.search(r"\b(bottom|worst|lowest|least|minimum|poorest|weakest)\b", q):
            n_match = re.search(r"\b(?:bottom|worst|lowest)\s+(\d+)\b", q)
            if not n_match:
                n_match = re.search(r"\b(\d+)\s+(?:bottom|worst|lowest|least|products|items|categories|customers|regions|brands|performers)\b", q)
            if n_match:
                n = int(n_match.group(1 if len(n_match.groups()) == 1 else 2))
            else:
                is_singular = (
                    bool(re.search(r"\b(which|who|what is the|what's the|name the|tell me the|find the)\s+(?:single\s+)?(product|item|category|region|customer|buyer|client|city|state|country|market|segment|department|branch|store|brand|supplier|channel|one)\b", q)) or
                    bool(re.search(r"\b(bottom|worst|lowest|least)\s+(product|item|category|region|customer|buyer|client|city|state|country|market|segment|department|branch|store|brand|supplier|channel)\b", q)) or
                    bool(re.search(r"\b(which|who|what)\b.*\b(lowest|least|worst|bottom)\b", q))
                ) and not re.search(r"\b(products|items|categories|regions|customers|buyers|clients|cities|states|countries|markets|segments|departments|branches|stores|brands|suppliers|channels|ranking|rankings|list|all|table|chart|compare|comparison|gap)\b", q)
                n = 1 if is_singular else 10

            group_col = _extract_target_group_col(q)
            value_col = _extract_metric_col(q)
            filter_col, filter_val = _extract_filter(q, exclude_col=group_col)
            agg_type = "mean" if re.search(r"\b(average|avg|mean|per order|per customer|per item|aov)\b", q) else "sum"
            if group_col and value_col:
                return ("get_top_n", {
                    "group_col": group_col,
                    "value_col": value_col,
                    "n": n,
                    "agg": agg_type,
                    "ascending": True,
                    "filter_col": filter_col,
                    "filter_val": filter_val,
                })

        # ── 00. STATUS FLAGS / DELIVERY PERFORMANCE / FILTERED COUNTS ──────
        if {"late", "deliver"}.issubset(q_stemmed) or re.search(r"\b(late delivery|delivered late|delayed orders|orders delayed|delays in shipping|at risk of late)\b", q):
            group_entity_in_q = re.search(r"\b(regions?|countr(?:y|ies)|cit(?:y|ies)|states?|areas?|territor(?:y|ies)|products?|categor(?:y|ies)|segments?|customers?|brands?|departments?|zones?)\b", q)
            if group_entity_in_q:
                late_risk_col = next((c for c in self.df.columns if "late" in c.lower() and "risk" in c.lower()), None)
                group_col = _find_col_in_q(roles["cat"], roles.get("region") or roles.get("category") or roles.get("product"))
                if late_risk_col and group_col:
                    try:
                        late_df = self.df[self.df[late_risk_col].astype(str) == "1"]
                        grp = late_df.groupby(group_col).size().reset_index(name="Late Delivery Count")
                        grp = grp.sort_values("Late Delivery Count", ascending=False)
                        return ("get_top_n", {"group_col": group_col, "value_col": late_risk_col,
                                               "n": 10, "agg": "sum", "ascending": False})
                    except Exception:
                        pass

            late_risk_col = next((c for c in self.df.columns if "late" in c.lower() and "risk" in c.lower()), None)
            if late_risk_col:
                return ("get_filtered_summary", {
                    "filter_col": late_risk_col,
                    "filter_value": "1",
                    "target_col": None,
                    "agg": "count"
                })
            deliv_col = next((c for c in self.df.columns if any(k in c.lower() for k in ["delivery", "shipping", "status"]) and not pd.api.types.is_numeric_dtype(self.df[c])), None)
            if deliv_col:
                unique_vals = self.df[deliv_col].dropna().unique()
                late_val = next((v for v in unique_vals if "late" in str(v).lower() or "delay" in str(v).lower()), None)
                if late_val:
                    return ("get_filtered_summary", {
                        "filter_col": deliv_col,
                        "filter_value": str(late_val),
                        "target_col": None,
                        "agg": "count"
                    })

        # General Filtered Count & Status
        if re.search(r"\b(how many|count of|number of|percentage of|total orders.*where|total.*with|are there any|how many.*are|how many.*in|how many.*from)\b", q):
            for col in self.df.columns:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                words_in_col = [w for w in c_clean.split() if len(w) > 2]
                if len(words_in_col) >= 2 and all(w in q for w in words_in_col):
                    unique_vals = set(self.df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, "0", "1", True, False, "True", "False", "yes", "no", "Yes", "No"}):
                        pos_val = 1 if 1 in unique_vals else (True if True in unique_vals else ("1" if "1" in unique_vals else ("yes" if "yes" in unique_vals else list(unique_vals)[0])))
                        target_col = _find_col_in_q(roles["num"], None, is_num=True) if re.search(r"\b(sum|revenue|sales|amount|value)\b", q) and not re.search(r"\b(how many|count)\b", q) else None
                        return ("get_filtered_summary", {
                            "filter_col": col,
                            "filter_value": str(pos_val),
                            "target_col": target_col,
                            "agg": "sum" if target_col else "count"
                        })

            for col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    vals = self.df[col].dropna().unique()
                    if len(vals) <= 500:
                        for val in vals:
                            val_str = str(val).lower().strip()
                            val_words = {_stem(w) for w in re.findall(r"\b\w+\b", val_str) if len(w) > 2}
                            if val_words and val_words.issubset(q_stemmed):
                                target_col = _find_col_in_q(roles["num"], None, is_num=True) if re.search(r"\b(sum|revenue|sales|amount|value)\b", q) and not re.search(r"\b(how many|count)\b", q) else None
                                return ("get_filtered_summary", {
                                    "filter_col": col,
                                    "filter_value": str(val),
                                    "target_col": target_col,
                                    "agg": "sum" if target_col else "count"
                                })

        # ── 0g. CUSTOMER REVENUE & SPEND METRICS ────────────────────────────
        if re.search(r"\b(each customer|per customer|revenue per buyer|spend per buyer|customer generate|customer spend|revenue by customer|customer revenue)\b", q):
            cust_col = roles.get("customer") or next((c for c in self.df.columns if any(k in c.lower() for k in ["customer", "client", "buyer"])), None)
            sales_col = roles.get("sales") or (roles["num"][0] if roles["num"] else None)
            if cust_col and sales_col:
                return ("get_customer_metrics", {"customer_col": cust_col, "value_col": sales_col})

        # ── COUNT UNIQUE / DISTINCT entity ──────────────────────────────────
        _count_unique_trigger = re.search(
            r"\b(how many|total number of|number of|count of|how much|total unique|unique count)\b", q
        )
        if _count_unique_trigger:
            _entity_map = [
                (["customer", "customers", "buyer", "buyers", "client", "clients", "shopper", "shoppers", "user", "users"], ["customer_id", "customer"]),
                (["product", "products", "item", "items", "sku", "skus", "goods"],                                            ["product_id", "product"]),
                (["order", "orders", "transaction", "transactions", "purchase", "purchases"],                                  ["order_id"]),
                (["segment", "segments"],                                                                                      ["segment"]),
                (["shipping mode", "ship mode", "delivery mode"],                                                              ["shipping_mode"]),
                (["category", "categories", "categor", "type", "types", "class", "classes"],                                   ["category"]),
                (["department", "departments", "dept", "depts"],                                                               ["department"]),
                (["country", "countries", "nation", "nations"],                                                                ["country"]),
                (["city", "cities", "town", "towns"],                                                                          ["city"]),
                (["state", "states", "province", "provinces"],                                                                ["state"]),
                (["region", "regions", "area", "areas", "territory", "territories", "zone", "zones"],                         ["region"]),
                (["market", "markets"],                                                                                        ["market"]),
                (["status", "statuses"],                                                                                       ["order_status", "delivery_status"]),
            ]
            _unique_col = None
            _unique_label = None
            for keywords, role_hints in _entity_map:
                if any(kw in q for kw in keywords) or any(_stem(kw) in q_stemmed for kw in keywords):
                    for rh in role_hints:
                        cand = roles.get(rh)
                        if cand and cand in self.df.columns:
                            _unique_col = cand
                            break
                    if not _unique_col:
                        for c in roles["cat"]:
                            c_lower = c.lower().replace("_", " ").replace("-", " ")
                            if any(kw in c_lower for kw in keywords) or any(_stem(kw) in c_lower for kw in keywords):
                                _unique_col = c
                                break
                    if _unique_col:
                        kw0 = keywords[0]
                        _unique_label = (kw0[:-1] + "ies") if kw0.endswith("y") else (kw0 + "s")
                        break

            if _unique_col:
                return ("count_unique", {"col": _unique_col, "label": _unique_label or _unique_col})

            if re.search(r"\b(orders?|rows?|records?|entry|entries|transactions?|purchases?)\b", q):
                return ("get_aggregate", {"col": roles["num"][0] if roles["num"] else list(self.df.select_dtypes('number').columns)[0],
                                          "agg": "count", "label": "Orders"})

        # ── 0c. GROWTH RATE / YoY / MoM / QoQ ────────────────────────────────
        if re.search(r"\b(yoy|mom|qoq|year over year|year-over-year|month over month|month-over-month|quarter over quarter|quarter-over-quarter|growth rate|percentage growth|growth percentage|sales growth|revenue growth|growth %|is .* growing|period.*growth|growth compared|compared with.*(?:month|year|quarter|period|previous)|compared to.*(?:month|year|quarter|period|previous)|growth from.*(?:month|year|quarter|period|previous)|growth over.*(?:month|year|quarter|period|previous)|percentage change.*(?:month|year|quarter|period|previous))\b", q) or \
           (re.search(r"\b(growth|percentage growth|pct growth|growth rate)\b", q) and re.search(r"\b(month|monthly|year|yearly|annual|quarter|quarterly|week|weekly|period|previous)\b", q)):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if dc and value_col:
                if re.search(r"\b(yoy|year over year|year-over-year|annual growth|yearly growth|annual|yearly)\b", q):
                    freq = "YE"
                elif re.search(r"\b(qoq|quarter over quarter|quarter-over-quarter|quarterly growth|quarterly)\b", q):
                    freq = "QE"
                elif re.search(r"\b(weekly growth|wow|week over week|weekly)\b", q):
                    freq = "W"
                elif re.search(r"\b(daily growth|day over day|daily)\b", q):
                    freq = "D"
                else:
                    freq = "ME"
                return ("get_growth_rate", {"date_col": dc, "value_col": value_col, "freq": freq})

        # ── 0b. TIME-SERIES TREND ────────────────────────────────────────────
        # "daily sales over time", "how has sales changed", "performance over time", "monthly revenue trend"
        if re.search(r"\b(over time|changed over|trend|daily sales|monthly sales|monthly revenue|per day|per month|per year|time series|how has.*changed|how did.*change|performance.*over|sales.*time|revenue.*time|growth over|day by day|week by week|month by month)\b", q) and not re.search(r"\b(yoy|mom|qoq|growth rate|growth %|profit margin|seasonality)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if dc and value_col:
                if re.search(r"\b(daily|day by day|per day|each day)\b", q):
                    freq = "D"
                elif re.search(r"\b(weekly|per week|each week|week by week)\b", q):
                    freq = "W"
                elif re.search(r"\b(yearly|annual|per year|year by year)\b", q):
                    freq = "YE"
                else:
                    freq = "ME"
                return ("get_time_series", {"date_col": dc, "value_col": value_col, "freq": freq})

        # ── 0d. SEASONALITY / MONTHLY PATTERN ────────────────────────────────
        if re.search(r"\b(seasonality|seasonal|monthly pattern|busiest month|best month|which month|peak sales month|slowest month)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if dc and value_col:
                return ("get_monthly_pattern", {"date_col": dc, "value_col": value_col})

        # ── Bug 1 fix: MULTI-METRIC (e.g., "total sales and profit margin") ──
        if re.search(r"\b(total|avg|average|sum|mean|max|min|overall)\b", q) and \
                re.search(r"\b(and|&|,)\b", q) and \
                not re.search(r"\b(chart|graph|plot|over time|by region|by product|by category|breakdown|trend|top|highest|lowest|best|worst|growth|percentage growth|growth rate|growth %|previous month|last month|prior month|per month|monthly|year over|month over|quarter over)\b", q):
            agg_type = "mean" if re.search(r"\b(average|avg|mean)\b", q) else "sum"
            candidate_metrics = []
            for col in roles["num"]:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                col_words = [w for w in c_clean.split() if len(w) > 2]
                if any(w in q for w in col_words):
                    candidate_metrics.append({"col": col, "agg": agg_type, "label": col})
            if len(candidate_metrics) >= 2:
                return ("get_multi_aggregate", {"metrics": candidate_metrics[:5]})

        # ── 0. SCALAR AGGREGATION (total / sum / average / count / max / min) ──
        # "total net sales", "what is total revenue", "average order value",
        # "how many orders", "grand total profit", "what is the maximum profit"
        if re.search(r"\b(total|sum of|grand total|overall|what is (?:our|the) total|how many|average order value|average order|avg order|mean (?:sales|profit|revenue|order|value)|max(?:imum)?|min(?:imum)?|what is the (?:max|min|maximum|minimum|highest value|lowest value))\b", q) and \
                not re.search(r"\b(which|who|what product|what market|what segment|what country|what customer|what region|what category|chart|graph|plot|treemap|heatmap|waterfall|pareto|funnel|over time|by month|by year|trend|by region|by product|by category|across|distributed|distribution|breakdown|share|contribute|contribution|proportions|split|top |inventory|stock|each customer|per customer|growth|percentage growth|growth rate|growth %|previous month|last month|prior month|per month|monthly|year over|month over|quarter over)\b", q):
            if re.search(r"\b(sales|revenue|net sales|total sales)\b", q) and roles.get("sales"):
                col = roles["sales"]
            elif re.search(r"\b(profit|earning|margin)\b", q) and roles.get("profit"):
                col = roles["profit"]
            else:
                col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if col:
                if re.search(r"\b(average|avg|mean|per order|per customer)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "mean", "label": col})
                elif re.search(r"\b(count|how many|number of)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "count", "label": col})
                elif re.search(r"\b(min|minimum|smallest|lowest value)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "min", "label": col})
                elif re.search(r"\b(max|maximum|largest|highest value)\b", q) and not re.search(r"\b(top|best|leading|by |across|per )\b", q):
                    return ("get_aggregate", {"col": col, "agg": "max", "label": col})
                else:
                    return ("get_aggregate", {"col": col, "agg": "sum", "label": col})

        # ── 0e. PROFIT MARGIN ────────────────────────────────────────────────
        if re.search(r"\b(profit margin|margin %|gross margin|operating margin|margin by)\b", q):
            cost_col = roles.get("cost") or (roles["num"][1] if len(roles["num"]) > 1 else None)
            rev_col = roles.get("sales") or roles.get("revenue") or (roles["num"][0] if roles["num"] else None)
            group_col = _find_col_in_q(roles["cat"], None)
            if cost_col and rev_col and cost_col != rev_col:
                return ("get_profit_margin", {"cost_col": cost_col, "revenue_col": rev_col, "group_col": group_col})

        # ── 0f. INVENTORY VALUE & LOW STOCK ──────────────────────────────────
        if re.search(r"\b(inventory value|stock value|value of stock|capital.*stock)\b", q):
            qty_col = roles.get("quantity") or _find_col_in_q(roles["num"], None, is_num=True)
            cost_col = roles.get("cost") or (roles["num"][1] if len(roles["num"]) > 1 else None)
            if qty_col and cost_col:
                return ("get_inventory_value", {"qty_col": qty_col, "cost_col": cost_col})

        if re.search(r"\b(low stock|out of stock|reorder|stock alert|running out)\b", q):
            qty_col = roles.get("quantity") or _find_col_in_q(roles["num"], None, is_num=True)
            item_col = roles.get("product") or (roles["cat"][0] if roles["cat"] else None)
            threshold = _extract_number(q) or 10
            if qty_col:
                return ("get_low_stock", {"qty_col": qty_col, "name_col": item_col, "threshold": int(threshold)})

        # ── 1. PERCENTILE queries ───────────────────────────────────────────
        pct_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*percentile", q)
        if pct_match:
            p = int(pct_match.group(1))
            col = _find_col_in_q(roles["num"], is_num=True) or roles.get("sales")
            if col:
                return ("get_single_percentile", {"col": col, "percentile": p})

        # ── 2. MEDIAN queries ───────────────────────────────────────────────
        if re.search(r"\b(median|middle value|50th percentile)\b", q):
            col = (_find_col_in_q(roles["num"], roles.get("delivery"), is_num=True)
                   if "delivery" in q or "time" in q
                   else _find_col_in_q(roles["num"], roles.get("sales"), is_num=True))
            if col:
                return ("get_median", {"col": col})

        # ── 3. PERCENTAGE ABOVE THRESHOLD queries ───────────────────────────
        # Only when explicitly asking for percentage or proportion of records/orders
        if re.search(r"\b(what percentage|what percent|what %|pct of|percentage of|proportion of|share of (?:orders|records|transactions|rows|items|customers))\b", q) and \
                re.search(r"\b(more than|above|exceed|over|greater than|higher than|less than|below|under)\b", q):
            num = _parse_amount(q) or _extract_number(re.sub(r"[₹$€£¥,]", "", q))
            if num is not None and num > 0:
                col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
                if col:
                    return ("get_pct_above_threshold", {"col": col, "threshold": num})

        # ── 4. OUTLIER queries ──────────────────────────────────────────────
        if re.search(r"\b(outlier\w*|anomal\w*|unusual|spike\w*)\b", q) or \
                ("extreme" in q and re.search(r"\b(sales|revenue|order|value|quantit|number|data)\b", q)):
            col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if col:
                return ("get_outliers", {"col": col})

        # ── Bug 4 fix: UNIQUE / LIST / ENUMERATE categorical values ─────────
        # "What are the unique delivery statuses?", "list all product categories",
        # "how many distinct order statuses are there", "enumerate regions"
        if re.search(r"\b(unique|distinct|list of|list all|enumerate|all possible|what are the|show all|how many different)\b", q):
            col = _find_col_in_q(roles["cat"])
            if col is None:
                # try matching any categorical column name keyword in question
                for c in roles["cat"]:
                    c_clean = c.lower().replace("_", " ").replace("-", " ")
                    if any(part in q for part in c_clean.split() if len(part) > 3):
                        col = c
                        break
            if col:
                return ("get_distribution", {"col": col})

        # ── 7. DISTRIBUTION queries ──────────────────────────────────────────
        if re.search(r"\b(distribution|frequency|histogram|spread|how many unique|count of each)\b", q):
            col = _find_col_in_q(roles["cat"] + roles["num"])
            if col is None:
                all_cols = roles["cat"] + roles["num"]
                for c in all_cols:
                    if any(part in q for part in re.split(r"[\s_\-]+", c.lower()) if len(part) > 3):
                        col = c
                        break
            col = col or (roles["cat"][0] if roles["cat"] else (roles["num"][0] if roles["num"] else None))
            if col:
                return ("get_distribution", {"col": col})

        # ── 8a. TREEMAP ──────────────────────────────────────────────────────
        if re.search(r"\b(treemap|tree map|hierarchical|nested.*chart)\b", q):
            group_col = _find_col_in_q(roles["cat"], roles.get("category") or roles.get("product"))
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            second_cat = roles.get("second_cat")
            if group_col and value_col:
                group_cols = [group_col, second_cat] if second_cat and second_cat != group_col else [group_col]
                return ("get_treemap", {"group_cols": group_cols, "value_col": value_col})

        # ── 8b. WATERFALL ────────────────────────────────────────────────────
        if re.search(r"\b(waterfall|bridge chart|revenue bridge|cost bridge)\b", q):
            group_col = _find_col_in_q(roles["cat"], roles.get("category") or roles.get("product"))
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if group_col and value_col:
                return ("get_waterfall", {"label_col": group_col, "value_col": value_col})

        # ── 8c. HEATMAP cross-tab ────────────────────────────────────────────
        if re.search(r"\b(heatmap|heat map|cross.?tab|crosstab)\b", q):
            row_col = _find_col_in_q(roles["cat"], roles.get("product") or roles.get("category"))
            col_col = (_find_col_in_q(roles["cat"][1:], roles.get("region") or roles.get("second_cat"))
                       if len(roles["cat"]) > 1 else None)
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if row_col and col_col and row_col != col_col:
                return ("get_heatmap_cross", {"row_col": row_col, "col_col": col_col,
                                              "value_col": value_col, "agg": "sum"})

        # ── 8d. GROUPED / STACKED BAR ────────────────────────────────────────
        if re.search(r"\b(grouped bar|stacked bar|side by side|group.*comparison)\b", q):
            x_col = _find_col_in_q(roles["cat"], roles.get("category") or roles.get("product"))
            group_col = roles.get("second_cat") or roles.get("region") or roles.get("category")
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if x_col and group_col and value_col and x_col != group_col:
                return ("get_grouped_comparison", {"x_col": x_col, "group_col": group_col,
                                                    "value_col": value_col})

        # ── 8. CATEGORY BREAKDOWN / CONTRIBUTION queries ────────────────────
        # "How is our revenue distributed across product categories, and which categories contribute the largest share of total sales?"
        if re.search(r"\b(breakdown|distributed across|distribution across|distributed by|by region|by product|by category|by segment|by country|by city|by area|by type|per region|per product|per category|across categories|across products|across regions|across segments|share of|contribute|contribution|proportions|split by|split across)\b", q):
            group_col = _find_col_in_q(roles["cat"], roles.get("category") or roles.get("product") or roles.get("region"))
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if group_col and value_col:
                return ("get_category_breakdown", {"group_col": group_col, "value_col": value_col, "agg": "sum"})

        # ── 9. CORRELATION & CAUSE-AND-EFFECT / IMPACT queries ──────────────
        # "Does discount appear related to quantity sold?", "Does offering higher discounts appear to increase the quantity of products sold?", "correlation between sales and quantity"
        if re.search(r"\b(correlat\w*|relationship\w*|related|relate\w*|depend\w*|associat\w*|impact\w* of|effect\w* of|influence\w* of|does .* (?:increase|decrease|affect|lead|impact|drive|relate)|do .* (?:increase|decrease|affect|lead|impact|drive|relate)|higher .* (?:increase|decrease|lead|drive)|lower .* (?:increase|decrease|lead|drive))\b", q):
            word_to_col = {}
            for col in roles["num"]:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                metric_words = [_stem(w) for w in re.split(r"[\s_\-]+", c_clean) if len(w) > 2 and w not in entity_stopwords]
                for w in metric_words:
                    pos = q.find(w)
                    if pos != -1:
                        if pos not in word_to_col or len(col) < len(word_to_col[pos]):
                            word_to_col[pos] = col
            sorted_positions = sorted(word_to_col.keys())
            found_cols = [word_to_col[p] for p in sorted_positions]

            if len(found_cols) < 2:
                for semantic_kws, role_key in [
                    (["profit", "margin", "benefit", "gain"], "profit"),
                    (["discount", "rebate"], "discount"),
                    (["cost", "expense", "spend", "cogs"], "cost"),
                    (["quantity", "qty", "volume", "units", "count"], "quantity"),
                    (["sales", "revenue", "income", "amount", "price"], "sales"),
                    (["delivery", "ship", "days"], "delivery"),
                ]:
                    if any(kw in q for kw in semantic_kws):
                        candidate = roles.get(role_key)
                        if candidate and candidate not in found_cols:
                            found_cols.append(candidate)
                            if len(found_cols) >= 2:
                                break

            if len(found_cols) >= 2:
                return ("get_correlation", {"cols": found_cols[:2]})
            elif len(found_cols) == 1:
                other_col = roles.get("sales") if found_cols[0] != roles.get("sales") else (roles.get("quantity") or roles["num"][0])
                if other_col and other_col != found_cols[0]:
                    return ("get_correlation", {"cols": [found_cols[0], other_col]})
            return ("get_correlation", {})

        # ── 10. MISSING VALUES ───────────────────────────────────────────────
        if re.search(r"\b(missing|null|nan|empty cell|incomplete|not filled)\b", q):
            return ("get_missing_values", {})

        # ── 11. DUPLICATES ───────────────────────────────────────────────────
        if re.search(r"\b(duplicat\w*|repeated row|same row twice)\b", q):
            return ("get_duplicates", {})

        # ── 12. OVERVIEW / SCHEMA ────────────────────────────────────────────
        if re.search(r"\b(overview|describe the data|tell me about|what columns|about the data|summarize the data|data info|dataset summary)\b", q):
            return ("get_data_overview", {})

        # ── 13. STATISTICS / DESCRIBE ────────────────────────────────────────
        if re.search(r"\b(statist\w*|summary stat\w*|mean and|average and|min and max|std dev|standard deviation|describe)\b", q):
            return ("get_describe", {})

        # ── 14. SEARCH ───────────────────────────────────────────────────────
        srch = re.search(r"\b(find|search for|search|look for|containing|show.*rows.*with|rows.*where.*named)\s+[\"']?([a-zA-Z0-9 _\-]+)[\"']?", q)
        if srch:
            keyword = srch.group(2).strip()
            if len(keyword) > 1:
                return ("search_items", {"keyword": keyword})

        # ── 15. DATE BOUNDS ──────────────────────────────────────────────────
        if re.search(r"\b(first order|earliest|oldest|when did|start year|begin|first date|first transaction)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            if dc:
                return ("get_date_bounds", {"date_col": dc, "target": "first"})

        if re.search(r"\b(latest|last order|most recent|end date|last date|last transaction|newest)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            if dc:
                return ("get_date_bounds", {"date_col": dc, "target": "latest"})

        if re.search(r"\b(date range|date span|time span|time range|covers|from.*to.*date)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            if dc:
                return ("get_date_bounds", {"date_col": dc, "target": "both"})

        # ── 16. PARETO / 80-20 analysis ─────────────────────────────────────
        if re.search(r"\b(pareto|80.?20|80\s*percent|80%|generate.*80|80.*revenue)\b", q):
            group_col = _find_col_in_q(roles["cat"], roles.get("product") or roles.get("category"))
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if group_col and value_col:
                return ("get_pareto", {"group_col": group_col, "value_col": value_col})

        # ── 17. FUNNEL / CONVERSION ──────────────────────────────────────────
        if re.search(r"\b(funnel|conversion.*funnel|drop.?off|where.*drop|pipeline.*chart)\b", q):
            stage_col = _find_col_in_q(roles["cat"], roles.get("stage") or roles.get("category"))
            if stage_col:
                return ("get_funnel", {"stage_col": stage_col})

        # ── 18. ROLLING AVERAGE / MOVING AVERAGE ────────────────────────────
        if re.search(r"\b(rolling|moving average|moving avg|smoothed|7.?day|30.?day|14.?day)\b", q):
            window_match = re.search(r"\b(\d+).?day", q)
            window = int(window_match.group(1)) if window_match else 30
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if dc and value_col:
                return ("get_rolling_average", {"date_col": dc, "value_col": value_col, "window": window})

        # ── 19. BUBBLE CHART ─────────────────────────────────────────────────
        if re.search(r"\b(bubble chart|bubble plot|size.*by|sized by|three variable|3.?variable)\b", q):
            x_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            y_col = roles.get("second_num") or roles.get("profit") or x_col
            size_col = roles.get("quantity") or roles.get("second_num") or x_col
            cat_col = _find_col_in_q(roles["cat"], roles.get("category"))
            if x_col and y_col and size_col and x_col != y_col:
                return ("get_bubble_data", {"x_col": x_col, "y_col": y_col,
                                            "size_col": size_col, "cat_col": cat_col})

        # ── 20. COHORT / RETENTION ───────────────────────────────────────────
        if re.search(r"\b(cohort|retention.*month|repeat.*customer|returning.*cohort)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            customer_col = roles.get("customer")
            if dc and customer_col:
                return ("get_cohort_retention", {"date_col": dc, "customer_col": customer_col})

        # ── 21. GAUGE / KPI VS TARGET ────────────────────────────────────────
        if re.search(r"\b(vs target|vs budget|vs goal|% of target|against target|how close.*target|target.*achievement)\b", q):
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            target_col = roles.get("target")
            target_num = _extract_number(re.sub(r"[₹$€£¥,]", "", q))
            if value_col:
                if target_num and target_num > 100:
                    return ("get_gauge_vs_target", {"value_col": value_col, "target": target_num})
                elif target_col and target_col in self.df.columns:
                    target_val = float(self.df[target_col].dropna().iloc[0]) if not self.df[target_col].dropna().empty else None
                    if target_val:
                        return ("get_gauge_vs_target", {"value_col": value_col, "target": target_val})

        # ── 22. SCATTER PLOT ─────────────────────────────────────────────────
        if re.search(r"\b(scatter plot|scatter chart|plot.*vs|scatterplot|scatter)\b", q):
            word_to_col = {}
            for col in roles["num"]:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                metric_words = [_stem(w) for w in re.split(r"[\s_\-]+", c_clean) if len(w) > 2 and w not in entity_stopwords]
                for w in metric_words:
                    pos = q.find(w)
                    if pos != -1:
                        if pos not in word_to_col or len(col) < len(word_to_col[pos]):
                            word_to_col[pos] = col
            sorted_positions = sorted(word_to_col.keys())
            found_cols = [word_to_col[p] for p in sorted_positions]
            if len(found_cols) < 2:
                for semantic_kws, role_key in [
                    (["profit", "margin", "benefit", "gain"], "profit"),
                    (["discount", "rebate"], "discount"),
                    (["cost", "expense", "spend", "cogs"], "cost"),
                    (["quantity", "qty", "volume", "units", "count"], "quantity"),
                    (["sales", "revenue", "income", "amount", "price"], "sales"),
                    (["delivery", "ship", "days"], "delivery"),
                ]:
                    if any(kw in q for kw in semantic_kws):
                        candidate = roles.get(role_key)
                        if candidate and candidate not in found_cols:
                            found_cols.append(candidate)
                            if len(found_cols) >= 2:
                                break

            if len(found_cols) >= 2:
                return ("get_correlation", {"cols": found_cols[:2]})
            x_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            y_col = roles.get("second_num") or roles.get("profit") or (roles["num"][1] if len(roles["num"]) > 1 else None)
            if x_col and y_col and x_col != y_col:
                return ("get_correlation", {"cols": [x_col, y_col]})

        # ── 23. DONUT CHART ──────────────────────────────────────────────────
        if re.search(r"\b(donut|doughnut|donut chart|ring chart)\b", q):
            group_col = _find_col_in_q(roles["cat"], roles.get("category") or roles.get("product"))
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if group_col and value_col:
                return ("get_category_breakdown", {"group_col": group_col,
                                                    "value_col": value_col, "agg": "sum"})

        # ── 24. RADAR / SPIDER ───────────────────────────────────────────────
        if re.search(r"\b(radar chart|spider chart|radar plot|multi.?metric|multiple metric)\b", q):
            return ("get_describe", {})

        # ── No local match — fall through to Gemini Pass 1 ──────────────────
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Schema — Compact, Cached, < 300 Tokens
    # ─────────────────────────────────────────────────────────────────────────

    def _get_schema_context(self) -> str:
        """Returns cached schema (built once at upload, never rebuilt per-question)."""
        if not self._schema_cache:
            self._schema_cache = self._build_schema_context()
        return self._schema_cache

    def _build_schema_context(self) -> str:
        """
        Compact column-level schema — no sample rows, no describe().
        Typical size: ~150-300 tokens for a 20-column dataset.
        """
        rows, cols = self.df.shape
        lines = [f"Dataset: {rows:,} rows x {cols} columns"]
        for col in self.df.columns:
            s = self.df[col]
            dtype = str(s.dtype)
            nunique = int(s.nunique())
            if pd.api.types.is_numeric_dtype(s):
                mn, mx = s.min(), s.max()
                lines.append(f"  {col} | {dtype} | range: {mn:.2f}-{mx:.2f} | {nunique} unique")
            elif pd.api.types.is_datetime64_any_dtype(s):
                mn = s.dropna().min()
                mx = s.dropna().max()
                lines.append(f"  {col} | datetime | {str(mn)[:10]}-{str(mx)[:10]} | {nunique} unique")
            else:
                sample = s.dropna().unique()[:4]
                sample_str = ", ".join(str(v) for v in sample)
                lines.append(f"  {col} | {dtype} | {nunique} unique (e.g. {sample_str})")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic Tool Selection — Only Send Relevant Declarations
    # ─────────────────────────────────────────────────────────────────────────

    def _get_relevant_tools(self) -> List[Dict]:
        """
        Selects which tool groups to include based on column names.
        Keeps Pass 1 token cost minimal by not sending irrelevant declarations.
        """
        tools = list(_BASE_TOOLS)
        cols_str = " ".join(self.df.columns).lower()

        if any(kw in cols_str for kw in ["date", "time", "year", "month", "day", "period"]):
            tools.extend(_TIME_TOOLS)

        if any(kw in cols_str for kw in ["price", "cost", "revenue", "sales", "amount", "profit", "margin", "earning"]):
            tools.extend(_FINANCIAL_TOOLS)

        if any(kw in cols_str for kw in ["stock", "quantity", "qty", "inventory", "units", "level"]):
            tools.extend(_INVENTORY_TOOLS)

        # Advanced tools (Pareto, Treemap, Waterfall, Funnel, Heatmap, Bubble, etc.)
        tools.extend(_ADVANCED_TOOLS)

        return tools

    # ─────────────────────────────────────────────────────────────────────────
    # Pass 1 — Gemini Function Calling
    # ─────────────────────────────────────────────────────────────────────────

    def _pass1_get_function_call(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Sends schema + question + relevant tool declarations to Gemini.
        Reuses a cached GenerativeModel instance when the tool set hasn't changed.

        Returns:
          {"name": str, "args": dict}   → Gemini selected a function
          None                           → Gemini responded but selected no function (genuine no-match)
          {"api_error": True, "quota": bool}  → API call failed (quota/auth/network)
        """
        schema = self._get_schema_context()
        tools = self._get_relevant_tools()

        # Reuse the same model+tools instance unless the tool list changed
        tools_key = len(tools)
        if self._model_with_tools is None or self._cached_tools_key != tools_key:
            self._model_with_tools = self._genai.GenerativeModel(
                self.model_name,
                tools=[{"function_declarations": tools}],
            )
            self._cached_tools_key = tools_key

        prompt = (
            f"Dataset schema:\n{schema}\n\n"
            f"User question: {question}\n\n"
            "Instructions:\n"
            "1. The question may be written in Bengali (বাংলা), Hindi (हिन्दी / Hinglish), or English.\n"
            "2. Understand the semantic meaning regardless of language and map it to the exact English column names listed in the schema above.\n"
            "   Examples of cross-language mappings (use schema columns, not these literals):\n"
            "   - 'বিক্রি' / 'বিক্রয়' / 'बिक्री' / 'sales'  →  Sales or Revenue column\n"
            "   - 'লাভ' / 'মুনাফা' / 'मुनाफा' / 'लाभ' / 'profit' →  Profit or Margin column\n"
            "   - 'তারিখ' / 'দিন' / 'दिनांक' / 'तारीख' / 'date'  →  Date or OrderDate column\n"
            "   - 'পণ্য' / 'প্রোডাক্ট' / 'उत्पाद' / 'product'    →  Product or Item column\n"
            "   - 'অঞ্চল' / 'এলাকা' / 'क्षेत्र' / 'region'       →  Region or Area column\n"
            "   - 'গ্রাহক' / 'কাস্টমার' / 'ग्राहक' / 'customer'  →  Customer column\n"
            "   - 'পরিমাণ' / 'সংখ্যা' / 'मात्रा' / 'quantity'    →  Quantity or Qty column\n"
            "3. Select the single most appropriate function and supply the exact column names from the schema."
        )
        try:
            response = self._model_with_tools.generate_content(prompt)
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", None):
                        return {"name": fc.name, "args": dict(fc.args)}
            # API responded successfully but selected no function → genuine no-match
            return None
        except Exception as e:
            err_str = str(e)
            is_quota = (
                "429" in err_str
                or "quota" in err_str.lower()
                or "resourceexhausted" in err_str.lower()
                or "rate" in err_str.lower()
            )
            import logging
            logging.getLogger(__name__).warning(
                "Pass 1 API error (quota=%s): %s", is_quota, err_str[:200]
            )
            return {"api_error": True, "quota": is_quota}

    # ─────────────────────────────────────────────────────────────────────────
    # Local Execution — Pattern A (Zero API Cost)
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_function(self, fn_name: str, fn_args: Dict) -> Dict:
        """
        Dispatches to the pre-written analysis function.
        Handles wrong/extra args from Gemini gracefully via TypeError recovery.
        """
        fn = _FUNCTION_MAP.get(fn_name)
        if fn is None:
            return {"answer": f"Unknown function: {fn_name}", "needs_llm": False}
        try:
            return fn(self.df, **fn_args)
        except TypeError:
            # Gemini passed extra/wrong params — try with only valid ones
            import inspect
            try:
                valid = set(inspect.signature(fn).parameters.keys()) - {"df"}
                cleaned = {k: v for k, v in fn_args.items() if k in valid}
                return fn(self.df, **cleaned)
            except Exception as e:
                return {"answer": f"Execution error: {str(e)}", "needs_llm": False}
        except Exception as e:
            return {"answer": f"Error running {fn_name}: {str(e)}", "needs_llm": False}

    # ─────────────────────────────────────────────────────────────────────────
    # Pass 2 — LLM Narrative (Only When needs_llm = True)
    # ─────────────────────────────────────────────────────────────────────────

    def _pass2_generate_answer(self, question: str, fn_name: str, fn_args: Dict,
                                result_summary: str) -> str:
        """
        Wraps a complex result with a professional, language-matched business narrative.
        Always responds in the same language and script as the user's question.
        Supports Bengali (বাংলা), Hindi (हिन्दी / Hinglish), and English.
        Token cost: question + summary (capped) + instruction ~ 300-500 tokens.
        """
        prompt = (
            "You are a concise, expert business data analyst.\n"
            "Rules:\n"
            "1. Respond in the EXACT SAME LANGUAGE and SCRIPT as the user's question:\n"
            "   - If the question is in Bengali (বাংলা) → respond entirely in natural, fluent Bengali.\n"
            "   - If the question is in Hindi (हिन्दी) or Hinglish → respond in natural Hindi or Hinglish.\n"
            "   - If the question is in English → respond in English.\n"
            "2. Answer in 1-2 sentences maximum. Lead directly with the key metric or finding.\n"
            "3. Bold key numbers, entity names, and dates using **markdown** formatting.\n"
            "4. Do NOT use introductory filler phrases like 'Based on the data', 'The analysis shows', 'আপনার প্রশ্নের উত্তরে', or 'डेटा के अनुसार'.\n\n"
            f"User Question: {question}\n"
            f"Executed Analysis: {fn_name}({json.dumps(fn_args, default=str)})\n"
            f"Data Result:\n{result_summary[:800]}"
        )
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except Exception:
            return result_summary  # Graceful degradation

    # ─────────────────────────────────────────────────────────────────────────
    # Pattern B — Code Generation Fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _pattern_b_fallback(self, question: str) -> Dict[str, Any]:
        """
        Fallback: ask Gemini to write a pandas snippet, execute it in sandbox, log it.
        Used when Pattern A function calling returns no match or for compound multi-condition queries.
        Supports multilingual questions — code generation is always in English/Python,
        but the final answer narrative is returned in the user's language.
        If the API is unavailable or quota is exceeded, falls back smoothly to local heuristics.
        """
        if not self.has_genai:
            return self._local_heuristic_fallback(question)

        is_indic = _is_indic_script(question)
        schema = self._get_schema_context()
        code_prompt = (
            f"Dataset schema:\n{schema}\n\n"
            "The user may have asked in Bengali, Hindi, or English. Understand the semantic meaning "
            f"and write an accurate Python/pandas code snippet to answer: {question}\n\n"
            "Rules:\n"
            "- The DataFrame is already loaded as `df`\n"
            "- Only use `df`, `pd`, `np`, `math` — no imports allowed\n"
            "- Store the final answer in a variable named `result` (can be a string, formatted summary, number, dict, or DataFrame/Series)\n"
            "- For compound questions (e.g. highest X but lowest Y), compute both aggregations and clearly state if an entity satisfies both or compare the top performers\n"
            "- For threshold questions (e.g. profit ratio below 0), filter matching items and summarize the count and key findings\n"
            "- Keep it under 20 lines\n"
            "- Output ONLY executable Python code, no explanation, no markdown fences"
        )

        generated_code = ""
        try:
            resp = self.model.generate_content(code_prompt)
            generated_code = resp.text.strip().removeprefix("```python").removesuffix("```").strip()
        except Exception as e:
            err_str = str(e)
            is_quota = "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower()
            log_api_failure(self.model_name, f"Code generation failed: {err_str}", is_quota=is_quota)
            log_fallback_query(question, "", False, error=f"Code generation failed: {err_str}")
            return self._local_heuristic_fallback(question, is_quota_error=is_quota)

        exec_result = execute_sandboxed(self.df, generated_code)
        success = exec_result.get("error") is None
        log_pattern_b_code(
            self.model_name,
            generated_code,
            success=success,
            result=exec_result.get("result"),
            error=exec_result.get("error"),
        )
        log_fallback_query(
            question=question,
            generated_code=generated_code,
            success=success,
            result_preview=exec_result.get("result"),
            error=exec_result.get("error"),
        )

        if exec_result.get("result"):
            # Pass 2: wrap raw code result in a language-matched business answer
            answer = self._pass2_generate_answer(question, "pattern_b_code", {}, exec_result["result"])
            chart_json = None
            res_df = exec_result.get("result_df")
            if res_df is not None and isinstance(res_df, pd.DataFrame) and len(res_df) > 1 and len(res_df.columns) >= 2:
                try:
                    chart_json = DataVisualizer.create_chart(
                        res_df,
                        "bar",
                        res_df.columns[0],
                        res_df.columns[1],
                        f"Analysis: {question[:40]}",
                    )
                except Exception:
                    chart_json = None
            return {"answer": answer, "chart_recommended": chart_json is not None, "chart": chart_json, "_used_pattern_b": True}

        return self._local_heuristic_fallback(question, is_quota_error=False)

    def _local_heuristic_fallback(self, question: str, is_quota_error: bool = False) -> Dict[str, Any]:
        """
        Zero-API resilient fallback using dataset schema and intelligent heuristic analysis.
        Ensures the user ALWAYS gets a real, meaningful data answer even if API quotas are exceeded.
        """
        import re
        q = question.lower().strip()
        roles = self._col_roles
        quota_notice = "\n\n*(Note: AI service is currently at capacity. Analysis computed via local Pandas engine.)*" if is_quota_error else ""

        # 1. Try to route through local intent router
        match = self._local_intent_router(question)
        if match:
            fn_name, fn_args = match
            fn_res = self._execute_function(fn_name, fn_args)
            chart_json = self._build_chart(fn_res, question)
            return {
                "answer": fn_res["answer"] + quota_notice,
                "chart_recommended": chart_json is not None,
                "chart": chart_json,
                "_fn_name": fn_name,
                "_fn_args": fn_args,
                "_routed_locally": True,
            }

        # 2. Extract mentioned columns
        num_cols = roles["num"]
        cat_cols = roles["cat"]
        date_cols = roles["date"]

        # Find best numeric and categorical column
        val_col = None
        for col in num_cols:
            c_clean = col.lower().replace("_", " ").replace("-", " ")
            if any(w in q for w in c_clean.split() if len(w) > 2):
                val_col = col
                break
        val_col = val_col or roles.get("sales") or (num_cols[0] if num_cols else None)

        grp_col = None
        for col in cat_cols:
            c_clean = col.lower().replace("_", " ").replace("-", " ")
            if any(w in q for w in c_clean.split() if len(w) > 2):
                grp_col = col
                break
        grp_col = grp_col or roles.get("category") or roles.get("product") or (cat_cols[0] if cat_cols else None)

        # Check compound question in local fallback
        if self._is_compound_question(question) and grp_col:
            m1 = val_col or (num_cols[0] if num_cols else None)
            m2 = roles.get("profit_ratio") or (num_cols[1] if len(num_cols) > 1 else m1)
            if m1 and m2:
                res = af.fn_dual_metric_ranking(self.df, grp_col, m1, "sum", False, m2, "mean", True)
                chart_json = self._build_chart(res, question)
                return {
                    "answer": res["answer"] + quota_notice,
                    "chart_recommended": bool(chart_json),
                    "chart": chart_json,
                    "_fn_name": "fn_dual_metric_ranking",
                }

        # 3. Check for specific intent keywords
        if any(kw in q for kw in ["trend", "time", "date", "month", "year", "over time", "daily", "monthly", "yearly"]) and (date_cols or roles.get("date_col")) and val_col:
            dc = roles.get("date_col") or date_cols[0]
            res = af.fn_get_time_series(self.df, dc, val_col, "ME")
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        if any(kw in q for kw in ["correlat", "relation", "impact", "affect", "increase", "depend"]) and len(num_cols) >= 2:
            res = af.fn_get_correlation(self.df, num_cols[:5])
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        if any(kw in q for kw in ["distribut", "spread", "count of", "frequency", "unique"]) and grp_col:
            res = af.fn_get_distribution(self.df, grp_col)
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        if any(kw in q for kw in ["outlier", "anomaly", "extreme", "unusual"]) and val_col:
            res = af.fn_get_outliers(self.df, val_col)
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        if any(kw in q for kw in ["top", "best", "highest", "most", "largest", "rank", "leader"]) and grp_col and val_col:
            res = af.fn_get_top_n(self.df, grp_col, val_col, 10, "sum", False)
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        if any(kw in q for kw in ["bottom", "worst", "lowest", "least", "smallest"]) and grp_col and val_col:
            res = af.fn_get_top_n(self.df, grp_col, val_col, 10, "sum", True)
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        if any(kw in q for kw in ["total", "sum", "average", "avg", "mean", "min", "max", "overall", "what is"]) and val_col:
            agg_type = "mean" if any(k in q for k in ["avg", "average", "mean"]) else ("max" if "max" in q else ("min" if "min" in q else "sum"))
            res = af.fn_get_aggregate(self.df, val_col, agg_type, val_col)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": False}

        # 4. Default: If categorical and numeric columns exist, provide top category breakdown
        if grp_col and val_col:
            res = af.fn_get_top_n(self.df, grp_col, val_col, 10, "sum", False)
            chart_json = self._build_chart(res, question)
            return {"answer": res["answer"] + quota_notice, "chart_recommended": bool(chart_json), "chart": chart_json}

        # 5. General statistical overview
        res = af.fn_get_data_overview(self.df)
        return {"answer": res["answer"] + quota_notice, "chart_recommended": False}


    # ─────────────────────────────────────────────────────────────────────────
    # Chart Building (Intent-Aware)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_chart(self, fn_result: Dict, question: Optional[str] = None) -> Optional[Dict]:
        """
        Builds Plotly chart from function result metadata with Intent-Aware Charting.

        Priority:
          1. chart_json  — pre-built by analysis function (Pareto, Treemap, Waterfall, Funnel, Heatmap, etc.)
          2. chart_type  — delegated to DataVisualizer.create_chart() with optional chart_kwargs

        Intent-Aware Logic:
          - For ranking, top-N, category breakdowns, and distributions (chart_type == "bar"),
            bar charts are suppressed unless the question explicitly requests visualization
            (e.g., 'chart', 'plot', 'graph', 'visualize', 'bar', 'draw', 'display chart', etc.).
          - This keeps chat answers fast, clean, and uncluttered for standard text-based data queries.
        """
        import re

        # Priority 1: pre-built chart JSON (complex charts: Pareto, Treemap, Waterfall, etc.)
        if fn_result.get("chart_json"):
            cj = fn_result["chart_json"]
            if isinstance(cj, dict) and "error" not in cj:
                return cj

        # Priority 2: simple chart via create_chart()
        chart_type = fn_result.get("chart_type")
        if not chart_type:
            return None

        result_df = fn_result.get("result_df")
        if result_df is None or result_df.empty or len(result_df) <= 1:
            return None

        # Intent-Aware check for standard bar charts (top-N, breakdowns, distributions)
        # Exception: time-series, monthly patterns, growth, and rolling averages are
        # inherently visual and should always render a chart regardless of wording.
        if chart_type == "bar" and question:
            q_lower = question.lower()
            # Always show charts for temporal/trend questions
            temporal_keywords = r"\b(month|monthly|trend|over time|by year|by quarter|seasonal|pattern|growth|rolling|moving average|weekly|daily|annual|yearly|time series|period|forecast)\b"
            if not re.search(temporal_keywords, q_lower):
                visual_keywords = r"\b(chart|plot|graph|visual|visualize|visualization|visuals|bar|bars|pie|draw|display chart|show chart|show plot|show graph|plot of|graph of|chart of)\b"
                if not re.search(visual_keywords, q_lower):
                    return None

        try:
            return DataVisualizer.create_chart(
                result_df,
                chart_type,
                fn_result.get("x_col"),
                fn_result.get("y_col"),
                fn_result.get("chart_title"),
                **fn_result.get("chart_kwargs", {}),
            )
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Processes a user question through the full pipeline:

        1. Zero-API fast-paths for trivial questions
        2. Pass 1: Gemini function calling (schema + question -> function name + args)
        3. Local execution: Python runs the function on the full DataFrame
        4. Template answer used directly (no Pass 2 for needs_llm=False)
        5. Pass 2: LLM narrative only if needs_llm=True
        6. Pattern B: sandboxed code generation if no function matched
        """
        log_query_start(question)
        q_lower = question.lower()

        # ── Zero-API fast-paths (common questions, no function call needed) ──
        # These cover the most frequent query patterns and short-circuit before any API call.

        if any(kw in q_lower for kw in ["missing", "null", "nan", "empty cell", "incomplete"]):
            fn_result = af.fn_get_missing_values(self.df)
            log_local_route("get_missing_values", {}, answer=fn_result["answer"])
            return {"answer": fn_result["answer"], "chart_recommended": False}

        if any(kw in q_lower for kw in ["duplicate", "duplicated", "repeated row"]):
            fn_result = af.fn_get_duplicates(self.df)
            log_local_route("get_duplicates", {}, answer=fn_result["answer"])
            return {"answer": fn_result["answer"], "chart_recommended": False}

        if any(kw in q_lower for kw in ["overview", "describe the dataset", "tell me about the data", "what columns", "about the data", "summarize the data"]):
            fn_result = af.fn_get_data_overview(self.df)
            log_local_route("get_data_overview", {}, answer=fn_result["answer"])
            return {"answer": fn_result["answer"], "chart_recommended": False}

        if any(kw in q_lower for kw in ["statistic", "describe", "summary stat", "mean and", "average and", "min and max"]):
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            fn_result = af.fn_get_describe(self.df, cols=num_cols[:6])
            log_local_route("get_describe", {"cols": num_cols[:6]}, answer=fn_result["answer"])
            return {"answer": fn_result["answer"], "chart_recommended": False}

        # ── Compound queries: if API key is active, route directly to Pattern B code generation ──
        if self._is_compound_question(question) and self.has_genai:
            return self._ask_with_cache(question)

        # ── No API key / or API key present: try local router first ──
        # The local router covers 15+ common intent patterns entirely with Pandas.
        # It runs here regardless of whether has_genai is True or False.
        local_match = self._local_intent_router(question)
        if local_match:
            fn_name, fn_args = local_match
            fn_result = self._execute_function(fn_name, fn_args)
            chart_json = self._build_chart(fn_result, question)
            log_local_route(fn_name, fn_args, answer=fn_result["answer"], chart=chart_json is not None)
            return {
                "answer": fn_result["answer"],
                "chart_recommended": chart_json is not None,
                "chart": chart_json,
                "_fn_name": fn_name,
                "_fn_args": fn_args,
                "_routed_locally": True,
            }

        if not self.has_genai:
            return self._local_heuristic_fallback(question)

        # ── LRU answer cache: skip API for repeated identical questions ──
        return self._ask_with_cache(question)

    def _ask_with_cache(self, question: str) -> Dict[str, Any]:
        """Inner ask logic with per-dataset LRU cache keyed on (df shape, columns, question)."""
        cache_key = (self.df.shape, tuple(self.df.columns), question.strip().lower())
        cached = _ANSWER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        result = self._ask_uncached(question)
        # Cache only successful answers (not errors)
        if result.get("answer") and not result["answer"].startswith("⚠️"):
            # Don't cache chart JSON (large) — strip it before caching
            _ANSWER_CACHE[cache_key] = {k: v for k, v in result.items() if k != "chart"}
        return result

    def _ask_uncached(self, question: str) -> Dict[str, Any]:
        """Core pipeline: local router → Pass 1 function calling → local execution → optional Pass 2."""
        q_lower = question.lower()

        # ── Detect if question is in an Indic script (Bengali or Hindi) ──
        # When True, we always invoke Pass 2 so the answer is returned in
        # the same language/script as the user's question.
        is_indic = _is_indic_script(question)

        # ── Compound queries with active AI key: route straight to Pattern B code generation ──
        if self._is_compound_question(question) and self.has_genai:
            return self._pattern_b_fallback(question)

        # ── Local Intent Router: Zero-API fast match ──────────────────────────
        # For common question patterns (top N, percentile, outliers, median,
        # threshold, correlation, etc.) we map directly to the analysis function
        # without ANY Gemini API call. Saves ~500-800 tokens and ~1-2 seconds per query.
        local_match = self._local_intent_router(question)
        if local_match:
            fn_name, fn_args = local_match
            fn_result = self._execute_function(fn_name, fn_args)
            # For Indic-language questions: still use Pass 2 to translate the answer
            if is_indic and self.has_genai:
                answer = self._pass2_generate_answer(
                    question, fn_name, fn_args,
                    fn_result.get("summary", fn_result["answer"])
                )
            else:
                answer = fn_result["answer"]
            chart_json = self._build_chart(fn_result, question)
            log_local_route(fn_name, fn_args, answer=answer, chart=chart_json is not None)
            return {
                "answer": answer,
                "chart_recommended": chart_json is not None,
                "chart": chart_json,
                "_fn_name": fn_name,
                "_fn_args": fn_args,
                "_routed_locally": True,
            }

        # ── Pass 1: Gemini Function Calling (only reached for unrecognized questions) ──
        fn_call = self._pass1_get_function_call(question)

        # Bug 6 fix: handle typed sentinel from _pass1_get_function_call
        if isinstance(fn_call, dict) and fn_call.get("api_error"):
            log_api_failure(self.model_name, "Pass 1 API error", is_quota=fn_call.get("quota", False))
            return self._local_heuristic_fallback(question, is_quota_error=fn_call.get("quota", False))

        if fn_call is None:
            # Gemini responded but selected no function → try Pattern B code generation
            return self._pattern_b_fallback(question)

        fn_name = fn_call["name"]
        fn_args = fn_call["args"]

        # ── Local Execution: full DataFrame, zero API cost ──
        fn_result = self._execute_function(fn_name, fn_args)

        # ── Decide: template answer or LLM narrative (Pass 2) ──
        # Force Pass 2 for Indic-script questions so the answer is
        # returned in Bengali or Hindi, not English-only templates.
        if (fn_result.get("needs_llm") or is_indic) and self.has_genai:
            answer = self._pass2_generate_answer(
                question, fn_name, fn_args,
                fn_result.get("summary", fn_result["answer"])
            )
        else:
            # Template answer — no 2nd LLM call
            answer = fn_result["answer"]

        # ── Build chart if the function returned chart metadata ──
        chart_json = self._build_chart(fn_result, question)
        log_gemini_pass1(self.model_name, fn_name, fn_args, answer=answer, chart=chart_json is not None)

        return {
            "answer": answer,
            "chart_recommended": chart_json is not None,
            "chart": chart_json,
            "_fn_name": fn_name,
            "_fn_args": fn_args,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Suggestions & Insights — Schema-Driven, Zero API Cost
    # ─────────────────────────────────────────────────────────────────────────

    def generate_suggested_questions(self) -> List[Dict[str, str]]:
        """Generates 4 context-aware suggested questions based on dataset schema."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object","category"]).columns.tolist()
        date_cols = self.df.select_dtypes(include=["datetime64"]).columns.tolist()
        cols_lower = " ".join(self.df.columns).lower()

        suggestions = []

        # Top performer
        if cat_cols and num_cols:
            suggestions.append({
                "category": "Top Performer",
                "question": f"Which {cat_cols[0]} has the highest total {num_cols[0]}?"
            })

        # Financial — if price/cost columns present
        if any(kw in cols_lower for kw in ["price","cost","revenue","sales"]):
            cost_hint = next((c for c in self.df.columns if "cost" in c.lower()), None)
            rev_hint = next((c for c in self.df.columns if any(k in c.lower() for k in ["revenue","sales","price"])), None)
            if cost_hint and rev_hint:
                suggestions.append({
                    "category": "Profitability",
                    "question": f"What is the profit margin for each {cat_cols[0]}?" if cat_cols else
                                f"What is the overall profit margin?"
                })

        # Time trend
        if date_cols and num_cols:
            suggestions.append({
                "category": "Trend Analysis",
                "question": f"Show the monthly trend of {num_cols[0]} over time."
            })

        # Stock alert
        if any(kw in cols_lower for kw in ["stock","qty","quantity","inventory"]):
            qty_col = next((c for c in self.df.columns if any(k in c.lower() for k in ["stock","qty","quantity"])), None)
            if qty_col:
                suggestions.append({
                    "category": "Inventory Alert",
                    "question": f"Which products have low stock (below 10 units)?"
                })

        # Fallback suggestions
        if len(suggestions) < 4:
            suggestions.append({"category": "Data Quality", "question": "Are there any missing values?"})
        if len(suggestions) < 4 and len(num_cols) >= 2:
            suggestions.append({
                "category": "Correlation",
                "question": f"What is the correlation between {num_cols[0]} and {num_cols[1]}?"
            })
        if len(suggestions) < 4 and cat_cols:
            suggestions.append({
                "category": "Distribution",
                "question": f"Show the distribution of {cat_cols[0]}."
            })

        return suggestions[:4]

    def generate_automated_insights(self) -> List[Dict[str, Any]]:
        """Generates insight cards using local Pandas analysis — zero API calls."""
        insights = []
        total_rows, total_cols = self.df.shape
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=["object","category"]).columns.tolist()

        # 1. Scale
        insights.append({
            "title": "Dataset Volume",
            "explanation": f"Loaded {total_rows:,} rows across {total_cols} columns.",
            "metric": f"{total_rows:,} Records",
            "level": "Good",
        })

        # 2. Completeness
        missing_count = int(self.df.isna().sum().sum())
        missing_pct = round(missing_count / self.df.size * 100, 1) if self.df.size > 0 else 0
        if missing_pct > 0:
            insights.append({
                "title": "Data Completeness Flag",
                "explanation": f"{missing_count:,} missing cells ({missing_pct}%). Consider imputation.",
                "metric": f"{missing_pct}% Missing",
                "level": "Warning",
            })
        else:
            insights.append({
                "title": "Data Completeness",
                "explanation": "No missing values detected.",
                "metric": "100% Complete",
                "level": "Good",
            })

        # 3. Top performer
        if cat_cols and num_cols:
            try:
                grouped = (
                    self.df.groupby(cat_cols[0], as_index=False)[num_cols[0]]
                    .sum().sort_values(num_cols[0], ascending=False)
                )
                if not grouped.empty:
                    top_name = str(grouped.iloc[0][cat_cols[0]])
                    top_val = float(grouped.iloc[0][num_cols[0]])
                    total_val = float(grouped[num_cols[0]].sum())
                    share = round(top_val / total_val * 100, 1) if total_val > 0 else 0
                    insights.append({
                        "title": f"Top Performer: {top_name}",
                        "explanation": f"'{top_name}' accounts for {share}% of total {num_cols[0]}.",
                        "metric": f"{share}% Share",
                        "level": "Good",
                    })
            except Exception:
                pass

        # 4. Strongest correlation
        if len(num_cols) >= 2:
            try:
                corr = self.df[num_cols].corr().abs().copy()
                np.fill_diagonal(corr.values, 0)
                if not corr.isna().all().all():
                    pair = corr.unstack().idxmax()
                    val = float(self.df[num_cols].corr().loc[pair[0], pair[1]])
                    if abs(val) > 0.4:
                        insights.append({
                            "title": "Strong Correlation",
                            "explanation": f"r = {val:.2f} between '{pair[0]}' and '{pair[1]}'.",
                            "metric": f"r = {val:.2f}",
                            "level": "Good",
                        })
            except Exception:
                pass

        # 5. Duplicate warning
        dups = int(self.df.duplicated().sum())
        if dups > 0:
            pct = round(dups / len(self.df) * 100, 1)
            insights.append({
                "title": "Duplicate Rows Detected",
                "explanation": f"{dups:,} duplicate rows ({pct}%). Remove before aggregating.",
                "metric": f"{dups:,} Duplicates",
                "level": "Warning",
            })

        return insights