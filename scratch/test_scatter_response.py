import sys
sys.path.insert(0, '.')
import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)
agent = DataAIAgent(df)

q = "Does discount appear related to quantity sold?"
print(f"Testing question: '{q}'")
res = agent.ask(q)

print("\n--- RESULT ---")
print("Routed locally:", res.get('_routed_locally'))
print("Function:", res.get('_fn_name'))
print("Function args:", res.get('_fn_args'))
print("Chart recommended:", res.get('chart_recommended'))
print("Chart generated:", bool(res.get('chart')))
if res.get('chart'):
    print("Chart layout title:", res['chart'].get('layout', {}).get('title', {}).get('text'))
    print("Chart traces count:", len(res['chart'].get('data', [])))
print("\nAnswer:\n", res['answer'])
