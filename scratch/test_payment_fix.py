import sys
sys.path.insert(0, '.')
import re
import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)

# Let's inspect Payment type values and average profit per payment type in Cleaned_data.csv
print("\n--- Actual Pandas Ground Truth on Cleaned_data.csv ---")
actual = df.groupby('Payment type')['Order Profit Per Order'].mean().sort_values(ascending=False)
print(actual)

print("\n--- Test on Cleaned_data.csv with agent ---")
agent = DataAIAgent(df)
q = "Which payment method has the highest average order profit?"
res = agent.ask(q)
print("Answer:\n", res.get("answer"))
