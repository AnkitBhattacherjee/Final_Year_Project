import sys
sys.path.insert(0, '.')
import re
import pandas as pd

df = pd.read_csv('uploads/sales_data_sample-selected-columns.csv')
from services.ai_agent import DataAIAgent

agent = DataAIAgent(df)
roles = agent._col_roles

def test_fixed_router(q, roles, df):
    q = q.lower().strip()
    
    # helper
    def _find_col_in_q(col_list, fallback=None, is_num=False):
        if not col_list: return fallback
        if is_num:
            if any(w in q for w in ["revenue", "sales", "sale", "income", "earning", "turnover"]):
                target = next((c for c in col_list if any(k in c.lower() for k in ["sale", "revenue", "total"])), None)
                if target: return target
        for col in col_list:
            if col.lower() in q:
                return col
        return fallback

    # ── GROWTH RATE / YoY / MoM / QoQ ────────────────────────────────
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

    # ── TIME-SERIES TREND ────────────────────────────────────────────
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

    # ── SCALAR AGGREGATION (total / sum / average / count / max / min) ──
    if re.search(r"\b(total|sum of|grand total|overall|what is (?:our|the) total|how many|average order value|average order|avg order|mean (?:sales|profit|revenue|order|value)|max(?:imum)?|min(?:imum)?|what is the (?:max|min|maximum|minimum|highest value|lowest value))\b", q) and \
            not re.search(r"\b(which|who|what product|what market|what segment|what country|what customer|what region|what category|chart|graph|plot|treemap|heatmap|waterfall|pareto|funnel|over time|by month|by year|trend|by region|by product|by category|across|distributed|distribution|breakdown|share|contribute|contribution|proportions|split|top |inventory|stock|each customer|per customer|growth|percentage growth|previous month|last month|prior month|per month|monthly|year over|month over|quarter over)\b", q):
        col = roles.get("sales") or _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
        if col:
            return ("get_aggregate", {"col": col, "agg": "sum", "label": col})

    return None

questions = [
    "Calculate monthly total sales and the percentage growth compared with the previous month.",
    "What is the monthly sales growth?",
    "Show me month over month growth in revenue",
    "Calculate total sales",
    "Daily sales over time",
    "Compare sales growth with previous month",
    "What is the percentage growth compared with the previous month",
]

for q in questions:
    res = test_fixed_router(q, roles, df)
    print(f"\nQ: {q}\n-> {res}")
