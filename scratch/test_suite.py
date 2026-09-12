import sys, pandas as pd, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,'.')
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=5000)
agent = DataAIAgent(df)

qs = [
    "How many unique customers do I have?",
    "Which customer segment generates the most sales?",
    "Which product generated the highest profit among orders from the Consumer segment?",
    "Which product generated the highest profit among orders?",
    "Which product name generated the highest profit among orders?",
    "Which shipping mode generates the highest average profit per order?",
    "What is the average sales?",
    "How many orders are there?",
    "How many different cities do we ship to?",
    "Which product has the lowest sales?",
    "Top 5 products by sales",
]

for q in qs:
    r = agent.ask(q)
    fn = r.get('_fn_name', 'N/A')
    args = r.get('_fn_args', {})
    ans = r['answer']
    print(f"=== Q: {q} ===")
    print(f"Function: {fn} | Args: {args}")
    print(f"Answer:\n{ans}\n")
