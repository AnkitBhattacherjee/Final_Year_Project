import sys
sys.path.insert(0, '.')
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)
agent = DataAIAgent(df)

complex_questions = [
    # Feature Engineering 1: Delivery Delay in days (real days - scheduled days)
    "Calculate the average shipping delay by comparing 'Days for shipping (real)' and 'Days for shipment (scheduled)' for each 'Shipping Mode'.",
    
    # Feature Engineering 2: High vs Low Discount Comparison
    "Compare the average Order Profit Per Order and Sales for high discount orders (discount rate >= 0.15) versus low discount orders (discount rate < 0.15).",

    # Feature Engineering 3: RFM / Customer grouping
    "Identify the top 5 most valuable customer cities based on total sales and profit ratio."
]

for q in complex_questions:
    print(f"\n{'#'*80}\nQUESTION: {q}\n{'#'*80}")
    res = agent.ask(q)
    print("FINAL ANSWER:\n", res.get("answer"))
    print("CHART GENERATED:", "Yes" if res.get("chart") else "No")
