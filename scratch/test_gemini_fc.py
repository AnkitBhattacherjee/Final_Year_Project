import sys
sys.path.insert(0, '.')
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_MODEL"] = "gemini-3.6-flash"

from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/sales_data_sample-selected-columns.csv')
agent = DataAIAgent(df)

q1 = "What is the correlation between quantity ordered and price each?"
res1 = agent.ask(q1)
print("\n--- Question 1 ---")
print("Answer:", res1.get("answer"))

q2 = "Calculate monthly total sales and the percentage growth compared with the previous month."
res2 = agent.ask(q2)
print("\n--- Question 2 ---")
print("Answer:", res2.get("answer"))
