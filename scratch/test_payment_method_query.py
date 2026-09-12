import sys
sys.path.insert(0, '.')
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from services.ai_agent import DataAIAgent

q = "Which payment method has the highest average order profit?"

datasets = [
    'uploads/restaurant_sales_data.csv',
    'uploads/Cleaned_data.csv',
]

for d in datasets:
    if os.path.exists(d):
        print(f"\n================ Testing on {d} ================")
        df = pd.read_csv(d, nrows=1000)
        print("Columns:", list(df.columns))
        agent = DataAIAgent(df)
        print("Roles:", agent._col_roles)
        
        # Check router match
        route = agent._local_intent_router(q)
        print(f"Local router match: {route}")
        
        res = agent.ask(q)
        print("Answer:\n", res.get("answer"))
