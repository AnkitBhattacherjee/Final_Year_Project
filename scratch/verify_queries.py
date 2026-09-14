import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np

from services.ai_agent import DataAIAgent
from services.data_analyzer import DataAnalyzer
from services.visualizer import DataVisualizer

df = pd.DataFrame({
    'Order_ID': ['O1', 'O1', 'O2', 'O3', 'O3', 'O4', 'O5', 'O6'],
    'Customer_Segment': ['Consumer', 'Consumer', 'Corporate', 'Home Office', 'Home Office', 'Consumer', 'Corporate', 'Home Office'],
    'Category': ['Technology', 'Grocery', 'Technology', 'Office Supplies', 'Grocery', 'Furniture', 'Grocery', 'Office Supplies'],
    'Payment_Method': ['Credit Card', 'Debit Card', 'Credit Card', 'Debit Card', 'UPI', 'UPI', 'Debit Card', 'Credit Card'],
    'Sales': [1000.0, 50.0, 5000.0, 100.0, 150.0, 800.0, 200.0, 300.0],
    'Profit': [200.0, 2.0, 1500.0, 10.0, 5.0, 80.0, 8.0, 60.0],
})

agent = DataAIAgent(df)

questions = [
    'Which customer segment has the highest average order value?',
    'Which customer segment has the lowest average order value?',
    'Which category has the highest total sales but a below-average profit margin?',
    'which payment method has lowest profit?'
]

for q in questions:
    res = agent.ask(q)
    print(f"Q: {q}")
    print(f"A: {res.get('answer')}")
    rdf = res.get('result_df')
    rows = len(rdf) if rdf is not None else 0
    print(f"X: {res.get('x_col')}, Y: {res.get('y_col')}, result_df rows: {rows}")
    chart_json = DataVisualizer.create_chart(rdf, res.get('chart_type') or 'bar', res.get('x_col'), res.get('y_col')) if rdf is not None else None
    print(f"Chart generated successfully: {chart_json is not None and 'error' not in chart_json}")
    print("-" * 50)
