import sys
sys.path.insert(0, '.')
import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)
agent = DataAIAgent(df)

questions = [
    "Does discount appear related to quantity sold?",
    "Does offering higher discounts appear to increase the quantity of products sold?",
    "scatter plot of sales vs product price",
    "correlation between sales and profit"
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    res = agent.ask(q)
    print("Fn:", res.get('_fn_name'), res.get('_fn_args'))
    print("Chart:", bool(res.get('chart')), "| Title:", res.get('chart', {}).get('layout', {}).get('title', {}).get('text') if res.get('chart') else None)
    print("Answer summary:", res['answer'].split('\n')[0])
