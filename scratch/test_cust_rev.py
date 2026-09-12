import pandas as pd
import numpy as np
import re

df = pd.read_csv('uploads/Cleaned_data.csv', nrows=5000)
print(f"Loaded {len(df)} rows")

# Check Customer revenue calculations
cust_col = 'Customer Id'
sales_col = 'Sales'
if cust_col in df.columns and sales_col in df.columns:
    cust_rev = df.groupby(cust_col)[sales_col].sum()
    print(f"Total customers: {len(cust_rev):,}")
    print(f"Mean revenue per customer: ${cust_rev.mean():,.2f}")
    print(f"Median revenue per customer: ${cust_rev.median():,.2f}")
    print(f"Top 5 customers:\n{cust_rev.nlargest(5)}")
