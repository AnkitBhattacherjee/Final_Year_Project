import pandas as pd
import numpy as np
import re

# Load sample df
df = pd.read_csv('uploads/Cleaned_data.csv', nrows=1000)

def stem(w):
    w = w.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w

questions = [
    "What percentage of orders were delivered late?",
    "how many orders are at risk of late delivery",
    "Does offering higher discounts appear to increase the quantity of products sold?",
    "how much revenue does each customer generate?",
    "What is our total net sales or, total profit?",
    "Which 10 products generate the most revenue, and how large is the gap between the top-performing and lower-performing products?",
    "How is our revenue distributed across product categories, and which categories contribute the largest share of total sales?",
    "what is the profit margin by category",
    "sales growth YoY",
    "seasonality in sales"
]

print("Testing question parsing...")
for q in questions:
    q_lower = q.lower()
    q_words = set(re.findall(r"\b\w+\b", q_lower))
    q_stemmed = {stem(w) for w in q_words}
    print(f"\nQ: {q}")
    print(f"Stemmed: {q_stemmed}")
