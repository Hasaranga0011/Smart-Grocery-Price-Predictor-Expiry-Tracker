import pandas as pd

def check_expiry(data: pd.DataFrame, today="2025-09-03"):
    today = pd.to_datetime(today)
    expiring_soon = data[data['expiry_date'] - today <= pd.Timedelta(days=3)]
    return expiring_soon[['product_name','expiry_date']]
