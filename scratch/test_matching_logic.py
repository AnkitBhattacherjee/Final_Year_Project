import pandas as pd
import numpy as np
import re

def _stem(w):
    w = w.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("delivery"):
        return "deliver"
    if w.endswith("deliveries"):
        return "deliver"
    if w.endswith("delivered"):
        return "deliver"
    if w.endswith("delivering"):
        return "deliver"
    if w.endswith("es") and len(w) > 4 and not w.endswith("ses"):
        return w[:-2]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w

q1 = "What percentage of orders were delivered late?"
q2 = "how many orders are at risk of late delivery"
q3 = "Does offering higher discounts appear to increase the quantity of products sold?"
q4 = "how much revenue does each customer generate?"

for q in [q1, q2, q3, q4]:
    words = re.findall(r"\b\w+\b", q.lower())
    stemmed = {_stem(w) for w in words}
    print(f"\nQ: {q}")
    print(f"Stemmed: {stemmed}")
    if {"late", "deliver"}.issubset(stemmed):
        print("-> Matched Late Delivery!")
    if any(k in stemmed for k in ["correlat", "impact", "effect"]) or (
        ("increase" in stemmed or "affect" in stemmed or "lead" in stemmed or "depend" in stemmed or "relat" in stemmed)
        and ("discount" in stemmed or "price" in stemmed or "cost" in stemmed)
    ):
        print("-> Matched Correlation / Impact!")
    if "customer" in stemmed and ("revenue" in stemmed or "spend" in stemmed or "generate" in stemmed or "each" in stemmed):
        print("-> Matched Customer Revenue!")
