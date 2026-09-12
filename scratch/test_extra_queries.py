import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from services.ai_agent import DataAIAgent

df = pd.DataFrame({
    "Order Country": ["USA", "USA", "USA", "Germany", "Germany", "UK", "UK"],
    "Order Profit Per Order": [100.0, 150.0, 50.0, 500.0, 400.0, 10.0, 20.0],
    "Order Item Profit Ratio": [-0.10, 0.25, -0.05, 0.35, 0.40, -0.02, 0.01],
    "Product Name": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
    "Sales": [1000, 2000, 500, 1500, 1200, 300, 400],
})

agent = DataAIAgent(df)

queries = [
    "Which products have a profit ratio below zero?",
    "Which products have negative profit?",
    "Which products have profit margin less than 0?",
    "Which country has the highest total profit but the lowest average profit per order?",
    "What country has the lowest profit and highest sales?"
]

for q in queries:
    res = agent.ask(q)
    print(f"\n--- Question: {q} ---")
    print("Answer:\n", res["answer"])
    assert res.get("answer") is not None
    assert not res["answer"].startswith("⚠️")

print("\nALL EXTRA QUERY TESTS PASSED SUCCESSFULLY! [PASS]")
