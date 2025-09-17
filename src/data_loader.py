import pandas as pd

def load_data(path="data/grocery_data.csv"):
    data = pd.read_csv(path, parse_dates=['date','expiry_date'])
    return data
