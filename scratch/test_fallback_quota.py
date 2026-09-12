import sys
sys.path.insert(0, '.')
import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=1000)
agent = DataAIAgent(df)

# Test simulated quota error fallback
fallback_q = "Explain the distribution of payment types and why customers use them"
print("\nTesting fallback on unrouted question with simulated quota error:")
res = agent._local_heuristic_fallback(fallback_q, is_quota_error=True)
print("Answer:\n", res['answer'])
print("Chart recommended:", res.get('chart_recommended'))
