import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load
import ta
import os
from pandas.tseries.offsets import BDay

# ─── Load Models ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = load(os.path.join(BASE_DIR, 'models.joblib'))
scaler = load(os.path.join(BASE_DIR, 'scaler.joblib'))

FEATURE_COLS = [
    'Close', 'High', 'Low', 'Open', 'Volume',
    'Volume_log', 'SMA_7', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'Lag_1', 'Lag_2', 'Lag_3',
    'Lag_7', 'Daily_Return', 'Close_to_Open',
    'std_5', 'Price_ROC', 'Relative_Volume'
]

# ─── Feature Engineering ───────────────────
def prepare_features(df):
    df = df.copy()
    df['Volume_log']      = np.log1p(df['Volume'])
    df['SMA_7']           = df['Close'].rolling(7).mean()
    df['EMA_12']          = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26']          = df['Close'].ewm(span=26, adjust=False).mean()
    df['RSI']             = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['MACD']            = df['EMA_12'] - df['EMA_26']
    df['Lag_1']           = df['Close'].shift(1)
    df['Lag_2']           = df['Close'].shift(2)
    df['Lag_3']           = df['Close'].shift(3)
    df['Lag_7']           = df['Close'].shift(7)
    df['Daily_Return']    = df['Close'].pct_change()
    df['Close_to_Open']   = df['Close'] - df['Open']
    df['std_5']           = df['Close'].rolling(5).std()
    df['Price_ROC']       = df['Close'].pct_change(5)
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

# ─── Confidence ────────────────────────────
def get_confidence(day):
    if day <= 7:    return "High 🟢"
    elif day <= 14: return "Medium 🟡"
    else:           return "Low 🔴"

# ─── UI ────────────────────────────────────
st.set_page_config(
    page_title="GAIL Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 GAIL.NS Stock Price Predictor")
st.markdown("---")

# Sidebar
st.sidebar.title("Settings")
horizon = st.sidebar.slider(
    "Prediction Days", 
    min_value=1, 
    max_value=14, 
    value=7
)

# Fetch Data
with st.spinner("Fetching latest data..."):
    raw = yf.download(
        "GAIL.NS",
        period="3mo",
        interval="1d",
        progress=False
    )
    raw.columns = raw.columns.droplevel(1)
    raw.reset_index(inplace=True)
    raw.columns.name = None
    df = prepare_features(raw)

# Metrics
current_price = float(df['Close'].iloc[-1])
prev_price    = float(df['Close'].iloc[-2])
change        = current_price - prev_price
change_pct    = (change / prev_price) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{current_price:.2f}")
col2.metric("Day Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
col3.metric("Prediction Days", f"{horizon} days")

st.markdown("---")

# Predict Button
if st.button("🔮 Predict", use_container_width=True):
    with st.spinner("Predicting..."):

        # Predict
        latest = df[FEATURE_COLS].iloc[-1:]
        scaled = scaler.transform(latest)
        preds  = [models[d].predict(scaled)[0] 
                  for d in range(1, horizon + 1)]

        # Business days only
        last_date  = pd.to_datetime(df['Date'].iloc[-1])
        pred_dates = [
            (last_date + BDay(i+1)).strftime('%Y-%m-%d')
            for i in range(horizon)
        ]

        # Chart
        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=df['Date'].tail(30).astype(str),
            y=df['Close'].tail(30),
            name="Historical",
            line=dict(color='#2196F3', width=2)
        ))

        # Connect historical to predicted
        connect_dates  = [str(df['Date'].iloc[-1])] + pred_dates
        connect_prices = [current_price] + preds

        fig.add_trace(go.Scatter(
            x=connect_dates,
            y=connect_prices,
            name="Predicted",
            line=dict(color='#FF9800', width=2, dash='dash'),
            mode='lines+markers'
        ))

        fig.update_layout(
            title="GAIL.NS Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Day-wise Predictions")
        result_df = pd.DataFrame({
            'Day'            : [f"Day {d+1}" for d in range(horizon)],
            'Date'           : pred_dates,
            'Predicted Price': [f"₹{p:.2f}" for p in preds],
            'Confidence'     : [get_confidence(d+1) 
                               for d in range(horizon)]
        })
        result_df = result_df.set_index('Day')
        st.dataframe(result_df, use_container_width=True)