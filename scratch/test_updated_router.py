import sys
sys.path.insert(0, '.')
import pandas as pd
import re

questions = [
    "Calculate monthly total sales and the percentage growth compared with the previous month.",
    "What is the monthly sales growth?",
    "Show me month over month growth in revenue",
    "Calculate total sales",
    "What is the total sales and profit margin",
    "Daily sales over time",
    "Compare sales growth with previous month",
    "What is the percentage growth compared with the previous month",
    "Total revenue by region",
]

df = pd.read_csv('uploads/sales_data_sample-selected-columns.csv')
from services.ai_agent import DataAIAgent

agent = DataAIAgent(df)
roles = agent._col_roles

print("Roles:", roles)

for q in questions:
    matched = agent._local_intent_router(q)
    print(f"\nQ: {q}\nMatched: {matched}")
