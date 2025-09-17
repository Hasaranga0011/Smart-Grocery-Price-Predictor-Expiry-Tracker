# ======================
# IMPORTS
# ======================
import streamlit as st
from src.data_loader import load_data
from src.price_prediction import predict_price
from src.expiry_tracker import check_expiry
from src.demand_forecast import forecast_demand

# ======================
# APP START
# ======================
st.title("🥕 Smart Grocery Price Predictor & Expiry Tracker")

# Load data
data = load_data()

# Show original data
st.subheader("📊 Grocery Data")
st.dataframe(data)

# Price Prediction
st.subheader("💰 Price Prediction")
price_df = predict_price(data.copy())
st.dataframe(price_df)

# Expiry Tracker
st.subheader("⏳ Products Expiring Soon")
expiring = check_expiry(data)
st.dataframe(expiring)

# Demand Forecast (Dropdown Selection)
st.subheader("📈 Demand Forecast")

# Dropdown for product selection
product_list = data['product_name'].unique().tolist()
selected_product = st.selectbox("Select a product to forecast demand:", product_list)

product_data = data[data['product_name'] == selected_product].sort_values('date')

if not product_data.empty:
    next_day_sales = forecast_demand(product_data)
    if next_day_sales is not None:
        st.success(f"🔮 Predicted next day sales for **{selected_product}**: {next_day_sales:.2f}")
    else:
        st.warning(f"Not enough data to forecast demand for **{selected_product}**.")
else:
    st.error(f"No data available for {selected_product}.")
