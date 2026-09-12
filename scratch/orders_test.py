import sys, pandas as pd, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,'.')
from services.ai_agent import DataAIAgent
df = pd.read_csv('uploads/Cleaned_data.csv', nrows=5000)
agent = DataAIAgent(df)

qs = [
    'How many orders are there?',
    'How many unique customers do I have?',
    'How many records are in the dataset?',
    'How many transactions do we have?',
    'How many products are there?',
    'How many different cities do we ship to?',
]
for q in qs:
    r = agent.ask(q)
    fn = r.get('_fn_name','N/A')
    args = r.get('_fn_args', {})
    ans = r['answer']
    print(f'Q: {q}')
    print(f'   Fn={fn} Args={args}')
    print(f'   Ans: {ans[:130]}')
    print()
