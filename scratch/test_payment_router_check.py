import sys
sys.path.insert(0, '.')
import re
import pandas as pd

df_clean = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)
df_rest = pd.read_csv('uploads/restaurant_sales_data.csv')

# Test proposed fix in services/ai_agent.py
from services.ai_agent import DataAIAgent

# Let's verify _build_col_roles and _extract_target_group_col with the fix
agent_clean = DataAIAgent(df_clean)
agent_rest = DataAIAgent(df_rest)

q = "Which payment method has the highest average order profit?"
print("Cleaned_data router:", agent_clean._local_intent_router(q))
print("Restaurant data router:", agent_rest._local_intent_router(q))
