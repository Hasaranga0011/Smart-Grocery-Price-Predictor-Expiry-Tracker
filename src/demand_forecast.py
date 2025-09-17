import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def forecast_demand(product_data: pd.DataFrame, timesteps=3):
    sales = product_data['sales'].values.reshape(-1,1)

    if len(sales) <= timesteps:
        return None   # Not enough data

    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    X_seq, y_seq = [], []
    for i in range(timesteps, len(sales_scaled)):
        X_seq.append(sales_scaled[i-timesteps:i])
        y_seq.append(sales_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Extra check for empty
    if X_seq.shape[0] == 0:
        return None

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=50, verbose=0)

    predicted_sales = model.predict(X_seq[-1].reshape(1, timesteps, 1))
    predicted_sales = scaler.inverse_transform(predicted_sales)
    return predicted_sales[0][0]
