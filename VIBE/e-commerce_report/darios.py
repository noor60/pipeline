import pandas as pd

data=pd.read_json('./weekly_sales/products-CARROT_EXPRESS-2025-05-01-to-2026-04-30-results.json')

data.head()

data1=pd.read_json('./weekly_sales/products-CARROT_EXPRESS-2025-05-01-0.0pct-results.json')
data.head()