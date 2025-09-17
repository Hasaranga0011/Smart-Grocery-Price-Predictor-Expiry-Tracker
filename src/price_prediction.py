import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def predict_price(data: pd.DataFrame):
    X = data[['stock', 'sales']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    data['predicted_price'] = model.predict(X)
    return data[['product_name','price','predicted_price']]
