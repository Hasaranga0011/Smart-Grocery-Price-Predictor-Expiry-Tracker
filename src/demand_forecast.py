import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def forecast_demand(product_data: pd.DataFrame, timesteps=3):
    """
    Forecast future demand for a product using LSTM.
    
    Parameters:
        product_data (pd.DataFrame): Must contain a 'sales' column with historical sales data.
        timesteps (int): Number of previous time steps to use for prediction.
        
    Returns:
        int: Forecasted sales as a non-negative integer.
        None: If not enough data to make a prediction.
    """
    # Extract sales values
    sales = product_data['sales'].values.reshape(-1, 1)

    # Not enough data
    if len(sales) <= timesteps:
        return None

    # Scale data between 0 and 1
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    # Prepare sequences for LSTM
    X_seq, y_seq = [], []
    for i in range(timesteps, len(sales_scaled)):
        X_seq.append(sales_scaled[i-timesteps:i])
        y_seq.append(sales_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    if X_seq.shape[0] == 0:
        return None

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_seq, y_seq, epochs=50, verbose=0)

    # Predict the next value
    predicted_sales = model.predict(X_seq[-1].reshape(1, timesteps, 1))
    predicted_sales = scaler.inverse_transform(predicted_sales)

    # Convert to non-negative integer
    predicted_sales_int = max(0, int(round(predicted_sales[0][0])))

    return predicted_sales_int
