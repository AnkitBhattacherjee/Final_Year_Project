import sys
sys.path.insert(0, '.')
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from services.ai_agent import DataAIAgent

df = pd.read_csv('uploads/sales_data_sample-selected-columns.csv')
agent = DataAIAgent(df)

questions = [
    # 1. Local intent router
    "Calculate monthly total sales and the percentage growth compared with the previous month.",
    # 2. Fast-path
    "Are there any missing values in this dataset?",
    # 3. Gemini function calling or local
    "What is the correlation between QUANTITYORDERED and SALES?",
    # 4. Complex compound query (Pattern B dynamic code generation)
    "Which orders have quantity above 40 and sales above 5000 with status Shipped?",
]

for q in questions:
    res = agent.ask(q)
