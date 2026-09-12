import sys
sys.path.insert(0, '.')
import pandas as pd
from services.ai_agent import DataAIAgent

# Load Cleaned_data.csv sample
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

print("=" * 70)
print("TESTING ALL QUESTIONS THROUGH DATA AI AGENT PIPELINE")
print("=" * 70)

for i, q in enumerate(questions, 1):
    print(f"\n[{i}] QUESTION: {q}")
    res = agent.ask(q)
    print(f"    Routed locally: {res.get('_routed_locally', False)}")
    print(f"    Function: {res.get('_fn_name')}")
    print(f"    Chart recommended: {res.get('chart_recommended')}")
    print("    ANSWER:")
    for line in res['answer'].split('\n'):
        print(f"      {line}")
