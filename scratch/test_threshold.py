import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.DataFrame({
    'Order Id': [101, 102, 103, 104, 105, 106],
    'Customer Id': [1, 2, 3, 4, 5, 6],
    'Customer Fname': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank'],
    'Customer Segment': ['Consumer', 'Corporate', 'Home Office', 'Consumer', 'Corporate', 'Home Office'],
    'Shipping Mode': ['Standard Class', 'Second Class', 'First Class', 'Same Day', 'Standard Class', 'Same Day'],
    'Market': ['LATAM', 'Europe', 'Pacific Asia', 'USCA', 'Africa', 'Europe'],
    'Order Country': ['United States', 'France', 'Germany', 'China', 'India', 'France'],
    'Sales': [1000000.0, 6000000.0, 5500000.0, 200000.0, 300000.0, 4000000.0],
    'Order Profit Per Order': [200.0, 1500.0, 1200.0, 50.0, 80.0, 2500.0],
    'Product Name': ['Shoe', 'Bike', 'Shirt', 'Watch', 'Glove', 'Helmet']
})

agent = DataAIAgent(df)
questions = [
    "Which countries have generated more than $5 million in sales?",
    "Which market has the highest total profit?",
    "Which customer segment has the highest average order value?",
    "How many unique customers do I have?",
    "Which customer segment generates the most sales?",
    "Which product generated the highest profit among orders from the Consumer segment?",
    "Which product generated the highest profit among orders?",
    "Which product name generated the highest profit among orders?",
    "Which shipping mode generates the highest average profit per order?",
    "What is the average sales?",
    "Top 5 products by sales"
]

for q in questions:
    res = agent.ask(q)
    print(f"=== Q: {q} ===")
    print("Answer:")
    print(res.get("answer"))
    print("-" * 50)

