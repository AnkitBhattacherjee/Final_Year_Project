import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import re

def test_enhanced_router():
    df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)
    
    # Let's write the prototype router
    all_num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in cat_cols:
        if c not in date_cols and any(k in c.lower() for k in ["date", "time", "day", "year", "month", "period"]):
            date_cols.append(c)

    id_keywords = ["id", "code", "zip", "key", "num", "number", "ssn", "phone", "lat", "lon", "latitude", "longitude", "unnamed", "status", "risk"]
    metric_num_cols = [
        c for c in all_num
        if not any(re.search(rf"\b{kw}\b", c.lower().replace("_", " ").replace("-", " ")) or kw in c.lower() for kw in id_keywords)
    ]
    num_cols = metric_num_cols if metric_num_cols else all_num

    def _pick(cols, *keywords):
        for kw in keywords:
            for c in cols:
                if kw in c.lower():
                    return c
        return cols[0] if cols else None

    roles = {
        "num": num_cols,
        "all_num": all_num,
        "cat": cat_cols,
        "date": date_cols,
        "sales": _pick(num_cols, "sale", "revenue", "total", "amount", "price", "value"),
        "quantity": _pick(num_cols, "qty", "quantity", "units", "count", "volume"),
        "discount": _pick(num_cols, "discount", "rebate"),
        "cost": _pick(num_cols, "cost", "expense", "spend", "cogs"),
        "profit": _pick(num_cols, "profit", "margin", "net"),
        "delivery": _pick(num_cols, "delivery", "ship", "lead", "days", "duration"),
        "product": _pick(cat_cols, "product", "item", "sku", "name", "good"),
        "category": _pick(cat_cols, "category", "segment", "type", "class", "line", "group"),
        "region": _pick(cat_cols, "region", "country", "state", "city", "territory", "area"),
        "customer": _pick(cat_cols, "customer", "client", "buyer", "account"),
        "date_col": _pick(date_cols, "order", "date", "transaction", "created"),
        "second_num": num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None),
        "second_cat": cat_cols[1] if len(cat_cols) > 1 else (cat_cols[0] if cat_cols else None),
    }

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

    def _find_col_in_q(col_list, fallback=None, is_num=False):
        if not col_list:
            return fallback
        q_words = [_stem(w) for w in re.findall(r"\b\w+\b", q)]
        
        # Check semantic metric synonyms first if is_num=True
        if is_num:
            if any(w in q_words for w in ["revenue", "sales", "sale", "income", "earning", "turnover"]):
                target = _pick(col_list, "sale", "revenue", "total")
                if target: return target
            if any(w in q_words for w in ["profit", "margin", "net", "gain"]):
                target = _pick(col_list, "profit", "margin")
                if target: return target
            if any(w in q_words for w in ["quantity", "qty", "volume", "units", "items"]):
                target = _pick(col_list, "qty", "quantity", "units", "count")
                if target: return target
            if any(w in q_words for w in ["discount", "discounts", "rebate"]):
                target = _pick(col_list, "discount")
                if target: return target
            if any(w in q_words for w in ["price", "pricing"]):
                target = _pick(col_list, "price")
                if target: return target
            if any(w in q_words for w in ["cost", "expense", "cogs"]):
                target = _pick(col_list, "cost", "expense")
                if target: return target

        for col in col_list:
            words = [_stem(w) for w in re.split(r"[\s_\-]+", col.lower()) if len(w) > 2]
            if any(w in q_words or any(w in qw or qw in w for qw in q_words if len(qw) > 3) for w in words):
                return col
        return fallback

    def _extract_number(text):
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        return float(m.group(1)) if m else None

    q = ""
    def route(question):
        nonlocal q
        q = question.lower().strip()
        q_words = re.findall(r"\b\w+\b", q)
        q_stemmed = {_stem(w) for w in q_words}

        # ── 00. STATUS FLAGS / DELIVERY PERFORMANCE / FILTERED COUNTS ──────
        if {"late", "deliver"}.issubset(q_stemmed) or re.search(r"\b(late delivery|delivered late|delayed orders|orders delayed|delays in shipping|at risk of late)\b", q):
            late_risk_col = next((c for c in df.columns if "late" in c.lower() and "risk" in c.lower()), None)
            if late_risk_col:
                return ("get_filtered_summary", {
                    "filter_col": late_risk_col,
                    "filter_value": "1",
                    "target_col": None,
                    "agg": "count"
                })
            deliv_col = next((c for c in df.columns if any(k in c.lower() for k in ["delivery", "shipping", "status"]) and not pd.api.types.is_numeric_dtype(df[c])), None)
            if deliv_col:
                unique_vals = df[deliv_col].dropna().unique()
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
            for col in df.columns:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                words_in_col = [w for w in c_clean.split() if len(w) > 2]
                if len(words_in_col) >= 2 and all(w in q for w in words_in_col):
                    unique_vals = set(df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, "0", "1", True, False, "True", "False", "yes", "no", "Yes", "No"}):
                        pos_val = 1 if 1 in unique_vals else (True if True in unique_vals else ("1" if "1" in unique_vals else ("yes" if "yes" in unique_vals else list(unique_vals)[0])))
                        target_col = _find_col_in_q(roles["num"], None, is_num=True) if re.search(r"\b(sum|revenue|sales|amount|value)\b", q) and not re.search(r"\b(how many|count)\b", q) else None
                        return ("get_filtered_summary", {
                            "filter_col": col,
                            "filter_value": str(pos_val),
                            "target_col": target_col,
                            "agg": "sum" if target_col else "count"
                        })

            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    vals = df[col].dropna().unique()
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
            cust_col = roles.get("customer") or next((c for c in df.columns if "customer" in c.lower() or "client" in c.lower() or "buyer" in c.lower()), None)
            sales_col = roles.get("sales") or (roles["num"][0] if roles["num"] else None)
            if cust_col and sales_col:
                return ("get_customer_metrics", {"customer_col": cust_col, "value_col": sales_col})

        # ── 0. SCALAR AGGREGATION ───────────────────────────────────────────
        if re.search(r"\b(total|sum of|grand total|overall|what is (?:our|the) total|how many|average order|avg order|mean (?:sales|profit|revenue|order|value))\b", q) and \
                not re.search(r"\b(chart|graph|plot|treemap|heatmap|waterfall|pareto|funnel|over time|by month|by year|trend|by region|by product|by category|across|distributed|distribution|breakdown|share|contribute|contribution|proportions|split|top |highest|lowest|best|worst|inventory|stock|margin|each customer|per customer)\b", q):
            col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if col:
                if re.search(r"\b(average|avg|mean|per order|per customer)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "mean", "label": col})
                elif re.search(r"\b(count|how many|number of)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "count", "label": col})
                elif re.search(r"\b(min|minimum|lowest|smallest)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "min", "label": col})
                elif re.search(r"\b(max|maximum|highest|largest)\b", q) and not re.search(r"\b(top|best|leading)\b", q):
                    return ("get_aggregate", {"col": col, "agg": "max", "label": col})
                else:
                    return ("get_aggregate", {"col": col, "agg": "sum", "label": col})

        # ── 0b. TIME-SERIES TREND ───────────────────────────────────────────
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

        # ── 0c. GROWTH RATE / YoY / MoM / QoQ ───────────────────────────────
        if re.search(r"\b(yoy|mom|qoq|year over year|year-over-year|month over month|month-over-month|quarter over quarter|quarter-over-quarter|growth rate|sales growth|revenue growth|growth %|is .* growing|period.*growth)\b", q):
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

        # ── 0d. SEASONALITY ──────────────────────────────────────────────────
        if re.search(r"\b(seasonality|seasonal|monthly pattern|busiest month|best month|which month|peak sales month|slowest month)\b", q):
            dc = roles.get("date_col") or (roles["date"][0] if roles["date"] else None)
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if dc and value_col:
                return ("get_monthly_pattern", {"date_col": dc, "value_col": value_col})

        # ── 0e. PROFIT MARGIN ────────────────────────────────────────────────
        if re.search(r"\b(profit margin|margin %|gross margin|operating margin|margin by)\b", q):
            cost_col = roles.get("cost") or (roles["num"][1] if len(roles["num"]) > 1 else None)
            rev_col = roles.get("sales") or roles.get("revenue") or (roles["num"][0] if roles["num"] else None)
            group_col = _find_col_in_q(roles["cat"], None)
            if cost_col and rev_col and cost_col != rev_col:
                return ("get_profit_margin", {"cost_col": cost_col, "revenue_col": rev_col, "group_col": group_col})

        # ── 5. TOP N queries ────────────────────────────────────────────────
        if re.search(r"\b(top|best|highest|most|leading|greatest|maximum)\b", q):
            n_match = re.search(r"\b(?:top|best|highest|most)\s+(\d+)\b", q)
            if not n_match:
                n_match = re.search(r"\b(\d+)\s+(?:top|best|highest|products|items|categories|customers|regions|brands|performers)\b", q)
            n = int(n_match.group(1 if len(n_match.groups()) == 1 else 2)) if n_match else 10
            group_col = (
                _find_col_in_q(roles["cat"]) or
                roles.get("product") or roles.get("category") or roles.get("region")
            )
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if group_col and value_col:
                return ("get_top_n", {"group_col": group_col, "value_col": value_col,
                                      "n": n, "agg": "sum", "ascending": False})

        # ── 8. CATEGORY BREAKDOWN / CONTRIBUTION queries ────────────────────
        if re.search(r"\b(breakdown|distributed across|distribution across|distributed by|by region|by product|by category|by segment|by country|by city|by area|by type|per region|per product|per category|across categories|across products|across regions|across segments|share of|contribute|contribution|proportions|split by|split across)\b", q):
            group_col = _find_col_in_q(roles["cat"], roles.get("category") or roles.get("product") or roles.get("region"))
            value_col = _find_col_in_q(roles["num"], roles.get("sales"), is_num=True)
            if group_col and value_col:
                return ("get_category_breakdown", {"group_col": group_col, "value_col": value_col, "agg": "sum"})

        # ── 9. CORRELATION & CAUSE-AND-EFFECT / IMPACT queries ──────────────
        if re.search(r"\b(correlat|relationship|related|relate|depend|associat|impact of|effect of|influence of|does .* (?:increase|decrease|affect|lead|impact|drive)|do .* (?:increase|decrease|affect|lead|impact|drive)|higher .* (?:increase|decrease|lead|drive)|lower .* (?:increase|decrease|lead|drive))\b", q):
            found_cols = []
            for col in roles["num"]:
                c_clean = col.lower().replace("_", " ").replace("-", " ")
                c_words = [_stem(w) for w in c_clean.split() if len(w) > 2]
                if any(w in q_stemmed for w in c_words):
                    if col not in found_cols:
                        found_cols.append(col)
            if len(found_cols) >= 2:
                return ("get_correlation", {"cols": found_cols[:2]})
            elif len(found_cols) == 1:
                other_col = roles.get("sales") if found_cols[0] != roles.get("sales") else (roles.get("quantity") or roles["num"][0])
                if other_col and other_col != found_cols[0]:
                    return ("get_correlation", {"cols": [found_cols[0], other_col]})
            return ("get_correlation", {})

        return None

    questions = [
        "What percentage of orders were delivered late?",
        "how many orders are at risk of late delivery",
        "Does offering higher discounts appear to increase the quantity of products sold?",
        "how much revenue does each customer generate?",
        "What is our total net sales or, total profit?",
        "Which 10 products generate the most revenue, and how large is the gap between the top-performing and lower-performing products?",
        "How is our revenue distributed across product categories, and which categories contribute the largest share of total sales?",
        "what is the profit margin by category",
        "sales growth YoY",
        "seasonality in sales"
    ]

    print("Testing Enhanced Router Prototype:")
    for question in questions:
        res = route(question)
        print(f"\nQ: {question}\n-> {res}")

test_enhanced_router()
