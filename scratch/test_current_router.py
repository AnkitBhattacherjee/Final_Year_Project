import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import re
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)
agent = DataAIAgent(df)

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

print("Testing DataAIAgent router on current code:")
for q in questions:
    routed = agent._local_intent_router(q)
    print(f"\nQ: {q}")
    print(f"-> Routed: {routed}")
