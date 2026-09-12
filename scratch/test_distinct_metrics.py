import re

def stem(w):
    w = w.lower().strip()
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and not w.endswith("ses"): return w[:-2]
    if w.endswith("ed") and len(w) > 4: return w[:-2]
    if w.endswith("ing") and len(w) > 5: return w[:-3]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w

q = "Does discount appear related to quantity sold?"
q_lower = q.lower()
q_words = re.findall(r"\b\w+\b", q_lower)
q_stemmed = {stem(w) for w in q_words}

num_cols = ['Sales', 'Product Price', 'Order Item Quantity', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Total', 'Order Profit Per Order', 'Order Item Profit Ratio']
entity_stopwords = {"product", "item", "order", "card", "customer", "user", "line", "detail", "entry", "record", "per"}

# Find distinct metric matches by question position
word_to_col = {}
for col in num_cols:
    c_clean = col.lower().replace("_", " ").replace("-", " ")
    metric_words = [w for w in re.split(r"[\s_\-]+", c_clean) if len(w) > 2 and w not in entity_stopwords]
    for w in metric_words:
        pos = q_lower.find(w)
        if pos != -1:
            # If this position / word not chosen yet, or this column has a shorter/more direct name
            if pos not in word_to_col or len(col) < len(word_to_col[pos][1]):
                word_to_col[pos] = (w, col)

# Sort by position in question
sorted_positions = sorted(word_to_col.keys())
distinct_cols = [word_to_col[p][1] for p in sorted_positions]
print("Distinct cols by word match:", distinct_cols)
