import re

def stem(w):
    w = w.lower().strip()
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and not w.endswith("ses"): return w[:-2]
    if w.endswith("ed") and len(w) > 4: return w[:-2]
    if w.endswith("ing") and len(w) > 5: return w[:-3]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w

q = "Does offering higher discounts appear to increase the quantity of products sold?"
q_words = re.findall(r"\b\w+\b", q.lower())
q_stemmed = {stem(w) for w in q_words}

cols = ['Sales', 'Product Price', 'Order Item Quantity', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Total', 'Order Profit Per Order', 'Order Item Profit Ratio']
entity_stopwords = {"product", "item", "order", "card", "customer", "user", "line", "detail", "entry", "record", "per"}

found = []
for c in cols:
    c_clean = c.lower().replace("_", " ").replace("-", " ")
    # filter out entity words
    metric_words = [stem(w) for w in re.split(r"[\s_\-]+", c_clean) if len(w) > 2 and w not in entity_stopwords]
    if any(w in q_stemmed for w in metric_words):
        found.append(c)

print("Found cols with entity stopword filtering:", found)
