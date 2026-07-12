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
models       = load(os.path.join(BASE_DIR, 'models.joblib'))
scaler       = load(os.path.join(BASE_DIR, 'scaler.joblib'))
feature_cols = [
    'Close_norm', 'High_norm', 'Low_norm', 'Open_norm',
    'Volume_log', 'SMA_7', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7',
    'Daily_Return', 'Close_to_Open', 'std_5',
    'Price_ROC', 'Relative_Volume'
]
# ─── Feature Engineering ───────────────────
def add_features(df):
    df = df.copy()
    
    # Normalize
    first_price      = df['Close'].iloc[0]
    df['Close_norm'] = df['Close'] / first_price * 100
    df['High_norm']  = df['High']  / first_price * 100
    df['Low_norm']   = df['Low']   / first_price * 100
    df['Open_norm']  = df['Open']  / first_price * 100
    
    # Features
    df['SMA_7']  = df['Close_norm'].rolling(7).mean()
    df['EMA_12'] = df['Close_norm'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close_norm'].ewm(span=26, adjust=False).mean()
    df['RSI']    = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    df['Lag_1']  = df['Close_norm'].shift(1)
    df['Lag_2']  = df['Close_norm'].shift(2)
    df['Lag_3']  = df['Close_norm'].shift(3)
    df['Lag_7']  = df['Close_norm'].shift(7)
    df['Daily_Return']    = df['Close'].pct_change()
    df['Close_to_Open']   = df['Close_norm'] - df['Open_norm']
    df['std_5']           = df['Close_norm'].rolling(5).std()
    df['Price_ROC']       = df['Close_norm'].pct_change(5)
    df['Volume_log']      = np.log1p(df['Volume'])
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

# ─── Prediction Function ───────────────────
def predict(df, horizon):
    # Latest row ke features lo
    X_latest = df[feature_cols].iloc[-1:]
    
    # Scale karo
    X_scaled = scaler.transform(X_latest)
    
    # Current price info
    current_price = float(df['Close'].iloc[-1])
    last_norm     = float(df['Close_norm'].iloc[-1])
    
    predictions = []
    for day in range(1, horizon + 1):
        # Normalized price predict karo
        pred_norm = models[day].predict(X_scaled)[0]
        
        # ─── Denormalization ───────────────
        # Model normalized price predict karta hai
        # Actual price nikalna:
        # first_price      = df['Close'].iloc[0]
        # last_norm        = df['Close_norm'].iloc[-1]
        # predicted_norm   = models[day].predict(X_latest)
        # Actual price:
        pred_actual = (pred_norm / last_norm) * current_price
        # ───────────────────────────────────
        
        predictions.append(pred_actual)
    
    return predictions

# ─── Confidence ────────────────────────────
def get_confidence(day):
    if day <= 3:   return "High 🟢"
    elif day <= 5: return "Medium 🟡"
    else:          return "Low 🔴"

# ─── UI ────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor")
st.markdown("---")

# Sidebar
st.sidebar.title("Settings")

ticker = st.sidebar.text_input(
    "Enter NSE Stock Ticker",
    value="GAIL.NS",
    help="Example: RELIANCE.NS, TCS.NS, INFY.NS"
)

horizon = st.sidebar.slider(
    "Prediction Days",
    min_value=1,
    max_value=7,
    value=7
)

# Fetch Data
with st.spinner(f"Fetching {ticker} data..."):
    try:
        raw = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False
        )
        
        if raw.empty:
            st.error(f"❌ {ticker} not found!")
            st.stop()
            
        raw.columns = raw.columns.droplevel(1)
        raw.reset_index(inplace=True)
        raw.columns.name = None
        df = add_features(raw)
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# Metrics
current_price = float(df['Close'].iloc[-1])
prev_price    = float(df['Close'].iloc[-2])
change        = current_price - prev_price
change_pct    = (change / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stock",         ticker)
col2.metric("Current Price", f"₹{current_price:.2f}")
col3.metric("Day Change",    f"₹{change:.2f}",
                             f"{change_pct:.2f}%")
col4.metric("Prediction Days", f"{horizon} days")

st.markdown("---")

# Predict Button
if st.button("🔮 Predict", use_container_width=True):
    with st.spinner("Predicting..."):
        preds = predict(df, horizon)

    # Business days
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

    # Connect + Predicted
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
        title=f"{ticker} Price Prediction",
        xaxis_title="",
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