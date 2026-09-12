import sys
sys.path.insert(0, '.')
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from dotenv import load_dotenv
from services.ai_agent import DataAIAgent

load_dotenv()

# Check with uploads/sales_data_sample-selected-columns.csv or restaurant_sales_data.csv
files = [
    'uploads/sales_data_sample-selected-columns.csv',
    'uploads/restaurant_sales_data.csv',
    'uploads/Cleaned_data.csv'
]

question = "Calculate monthly total sales and the percentage growth compared with the previous month."

for f in files:
    if os.path.exists(f):
        print(f"\n================ Testing on {f} ================")
        try:
            df = pd.read_csv(f, nrows=1000)
            print("Columns:", list(df.columns))
            agent = DataAIAgent(df)
            
            # Check local router
            local_route = agent._local_intent_router(question)
            print(f"Local router match: {local_route}")
            
            res = agent.ask(question)
            print("Agent Answer:\n", res.get("answer"))
            print("Chart:", "Yes" if res.get("chart") else "No")
        except Exception as e:
            print(f"Error on {f}:", e)
