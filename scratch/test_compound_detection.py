import sys
sys.path.insert(0, '.')
import re
import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=100)
agent = DataAIAgent(df)

questions = [
    "Calculate the average shipping delay by comparing 'Days for shipping (real)' and 'Days for shipment (scheduled)' for each 'Shipping Mode'.",
    "Compare the average Order Profit Per Order and Sales for high discount orders (discount rate >= 0.15) versus low discount orders (discount rate < 0.15).",
    "Identify the top 5 most valuable customer cities based on total sales and profit ratio.",
    "Classify customers into high, medium, and low spenders and calculate the average profit per group.",
    "Calculate the lead time by subtracting order date from delivery date and find which product has the longest lead time.",
]

for q in questions:
    is_compound = agent._is_compound_question(q)
    print(f"\nQ: {q}\n_is_compound_question: {is_compound}")
