import sys
sys.path.insert(0, '.')
import re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from services.ai_agent import DataAIAgent

def is_compound_or_feature_engineering(q: str) -> bool:
    q = q.lower().strip()
    
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

    # 4. A vs B comparisons (versus / vs / compared to)
    if re.search(r"\b(versus|vs\.?|compare\b.+\bversus\b|comparing\b.+\band\b.+\b(for each|by|across)\b)\b", q):
        return True

    # 5. Feature Engineering: Column math, differences, lead time, delays, duration, custom ratios
    if re.search(r"\b(subtracting|subtract|difference between|delay by|delay between|duration between|lead time|leadtime|time between|days between|hours between|gap between|ratio of\b.+\bto\b|divided by|product of|multiplying)\b", q):
        return True

    # 6. Feature Engineering: Segmentation, bucketing, classification, scoring
    if re.search(r"\b(classify|categorize|segment|bucket|cohort|rfm|tier|rank based on|based on\b.+\band\b.+\b(profit|sales|margin|spend|frequency|delay))\b", q):
        return True

    # 7. Multi-metric composite ranking (e.g. based on sales and profit ratio)
    if re.search(r"\b(based on|considering|combining)\b.+\band\b.+\b(ratio|margin|score|performance|profit|sales|growth)\b", q):
        return True

    return False

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=500)
agent = DataAIAgent(df)

complex_questions = [
    "Calculate the average shipping delay by comparing 'Days for shipping (real)' and 'Days for shipment (scheduled)' for each 'Shipping Mode'.",
    "Compare the average Order Profit Per Order and Sales for high discount orders (discount rate >= 0.15) versus low discount orders (discount rate < 0.15).",
    "Identify the top 5 most valuable customer cities based on total sales and profit ratio.",
    "Classify customers into high, medium, and low spenders and calculate the average profit per group.",
    "Calculate the lead time by subtracting order date from delivery date and find which product has the longest lead time.",
]

for q in complex_questions:
    detected = is_compound_or_feature_engineering(q)
    print(f"\nQ: {q}\n-> Detected for Pattern B (Feature Engineering/Dynamic Code): {detected}")
    if detected:
        # Test Gemini code generation
        res = agent._pattern_b_fallback(q)
        print("Generated Code / Answer:\n", res.get("answer"))
