import pandas as pd
import numpy as np
import re

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=100)
all_num = df.select_dtypes(include=[np.number]).columns.tolist()
print("All num:", all_num)

id_keywords = ["id", "code", "zip", "key", "num", "number", "ssn", "phone", "lat", "lon", "latitude", "longitude", "unnamed", "status", "risk"]
metric_num_cols = [
    c for c in all_num
    if not any(re.search(rf"\b{kw}\b", c.lower().replace("_", " ").replace("-", " ")) or kw in c.lower() for kw in id_keywords)
]
print("\nMetric num cols:", metric_num_cols)
