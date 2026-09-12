import sys
sys.path.insert(0, '.')
import pandas as pd
import re

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=2000)

q = "Does discount appear related to quantity sold?"
q_lower = q.lower()

# Preserve order of appearance in question
num_cols = ['Sales', 'Product Price', 'Order Item Quantity', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Total', 'Order Profit Per Order', 'Order Item Profit Ratio']
entity_stopwords = {"product", "item", "order", "card", "customer", "user", "line", "detail", "entry", "record", "per"}

col_positions = []
for col in num_cols:
    c_clean = col.lower().replace("_", " ").replace("-", " ")
    metric_words = [w for w in re.split(r"[\s_\-]+", c_clean) if len(w) > 2 and w not in entity_stopwords]
    for w in metric_words:
        pos = q_lower.find(w)
        if pos != -1:
            col_positions.append((pos, col))
            break

col_positions.sort(key=lambda x: x[0])
found_cols = []
for _, c in col_positions:
    if c not in found_cols:
        found_cols.append(c)

print("Preserved order cols:", found_cols)
