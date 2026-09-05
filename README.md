# Stock Price Predictor

**Multi-Ticker Forecasting Engine · XGBoost**

Forecasting NSE stock prices with technical indicators and ensemble gradient boosting.

`Python` `XGBoost` `Streamlit`

Day-wise regression forecasting of NSE-listed equity prices, 1 to 7 trading days ahead.
**Author:** Pawan Suman

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Results](#key-results)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Model Details](#model-details)
  - [XGBoost vs. LSTM](#xgboost-vs-lstm)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset Schema](#dataset-schema)
- [Methodology Deep Dive](#methodology-deep-dive)
- [Outputs](#outputs)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)
- [Author](#author)

## Overview

This project implements a supervised multi-output regression system that forecasts short-term closing prices for NSE-listed equities. It combines classical technical-analysis indicators with gradient-boosted regression trees, wrapped in an interactive Streamlit app.

| Component | Description |
|---|---|
| Input | Daily OHLCV data for any NSE-listed ticker |
| Output | Predicted closing price for each of the next 1–7 trading days |
| Approach | 7 independent XGBoost regressors, one per forecast horizon |

The model is trained across 19 diverse NSE tickers rather than a single stock, so it generalizes across sectors instead of overfitting to one company's price behavior.

## Problem Statement

Retail and research users often want a quick, data-driven read on where a stock's price may be headed over the next trading week, without building a forecasting pipeline themselves. Naive approaches (last price, simple moving average) ignore momentum, volatility, and volume dynamics.

This project addresses that gap by:
- Engineering a feature set that captures trend, momentum, and volatility
- Training a separate model per forecast day, rather than one model recursively predicting multiple days ahead (which compounds error)
- Assigning a confidence label to each forecast day based on model reliability

## Key Results

| Metric | Value |
|---|---|
| Algorithm | XGBoost Regressor (per-day) |
| Tuning | Optuna (best trial R² = 0.7135) |
| Training rows | 23,465 |
| Test rows | 11,667 |
| Tickers trained | 19 NSE-listed equities |
| Day 1 R² / MAE | 0.75 / 20.86 |
| Day 7 R² / MAE | 0.72 / 25.23 |

| Day | MAE | R² |
|---|---|---|
| 1 | 20.86 | 0.75 |
| 2 | 22.45 | 0.73 |
| 3 | 23.24 | 0.73 |
| 4 | 23.68 | 0.73 |
| 5 | 24.52 | 0.72 |
| 6 | 25.21 | 0.71 |
| 7 | 25.23 | 0.72 |

Accuracy degrades gradually rather than sharply, which supports using the full 7-day forecast window rather than truncating it early.

## Pipeline Architecture

```
INPUT: Daily OHLCV data, 19 NSE tickers (yfinance, 2019+)
        │
        ▼
DATA LOAD          → yfinance → per-ticker DataFrames
        │
        ▼
DATA CLEANING       → Missing value check
                    → Volume outlier detection / log transform
        │
        ▼
FEATURE ENGINEERING → Price normalization (per ticker)
                    → SMA_7, EMA_12, EMA_26
                    → RSI (14), MACD
                    → Lag_1, Lag_2, Lag_3, Lag_7
                    → Daily_Return, Close_to_Open
                    → std_5, Price_ROC, Relative_Volume
                    → 19 features, 35,132 rows total
        │
        ▼
TIME-BASED SPLIT    → Train: before 2024 · Test: 2024 onward
                    → StandardScaler fit on training features only
        │
        ▼
MODEL TRAINING      → 7 × XGBoost Regressor (one per forecast day)
                    → Hyperparameters tuned via Optuna
        │
        ▼
EVALUATION          → MAE + R² per forecast day (see Key Results)
        │
        ▼
OUTPUTS             → models.joblib · scaler.joblib
                    → feature_cols.joblib · Streamlit app
```

## Feature Engineering

The final feature matrix has 19 dimensions, all derived from OHLCV data with no external data sources:

| Feature | Description |
|---|---|
| `Close_norm`, `High_norm`, `Low_norm`, `Open_norm` | Prices normalized to the ticker's rolling mean |
| `SMA_7` | 7-day Simple Moving Average |
| `EMA_12`, `EMA_26` | 12-day and 26-day Exponential Moving Averages |
| `RSI` | Relative Strength Index (14-day window) |
| `MACD` | `EMA_12` minus `EMA_26` |
| `Lag_1`, `Lag_2`, `Lag_3`, `Lag_7` | Normalized close price at previous lags |
| `Daily_Return` | Day-over-day percentage change in close price |
| `Close_to_Open` | Normalized close minus normalized open |
| `std_5` | 5-day rolling standard deviation of normalized close |
| `Price_ROC` | 5-day rate of change in normalized close |
| `Volume_log` | Log-transformed trading volume |
| `Relative_Volume` | Current volume relative to its 20-day rolling average |

Normalizing prices per ticker allows a single model family to be trained across stocks with very different absolute price ranges.

## Model Details

Each forecast day is handled by an independently trained XGBoost Regressor, rather than a single model recursively predicting multiple steps ahead. This avoids compounding prediction error across the horizon.

```python
XGBRegressor(
    n_estimators=416,
    max_depth=5,
    learning_rate=0.2597,
    subsample=0.9062,
    colsample_bytree=0.8347
)
```

Hyperparameters were selected via Optuna, optimizing R² on a held-out validation split (best trial R² = 0.7135).

### Confidence Levels

Displayed in the app for each forecast day, based on the observed decline in R² across the horizon:

| Days Ahead | Confidence |
|---|---|
| 1–3 | High |
| 4–5 | Medium |
| 6–7 | Low |

### Denormalization

Models are trained to predict a normalized closing price. At inference time, the predicted normalized value is scaled back to an actual price using the ticker's current price and its current normalized value:

```
pred_actual = (pred_norm / last_norm) * current_price
```

### XGBoost vs. LSTM

An LSTM was trained as an alternative to XGBoost, to check whether a sequence model could better capture temporal dependencies in the price data. Sequences of 10 trading days were fed into a 2-layer LSTM, with hyperparameters tuned via Optuna:

```python
Sequential([
    LSTM(units=241, return_sequences=True, input_shape=(10, 19)),
    Dropout(0.117),
    LSTM(units=117),
    Dropout(0.117),
    Dense(1)
])
# Adam optimizer, lr=0.0021, batch_size=54, trained with early stopping
```

A separate LSTM was trained per forecast day, matching the per-day XGBoost setup:

| Day | XGBoost R² | LSTM R² | LSTM MAE |
|---|---|---|---|
| 1 | 0.9853 | 0.9642 | 1.79 |
| 2 | 0.9728 | 0.9527 | 2.15 |
| 3 | 0.9601 | 0.9406 | 2.45 |
| 4 | 0.9479 | 0.9283 | 2.73 |
| 5 | 0.9354 | 0.9173 | 2.96 |
| 6 | 0.9241 | 0.9057 | 3.19 |
| 7 | 0.9118 | 0.8945 | 3.41 |

XGBoost outperformed the LSTM at every forecast horizon, while also being far cheaper to train and tune. This is why XGBoost was selected as the production model shipped in the Streamlit app, and the LSTM remains in the notebook as a documented experiment rather than a deployed artifact.

## Project Structure

```
StockPredictor/
├── app.py                       # Streamlit application
├── models.joblib                # 7 trained XGBoost regressors (Day 1–7)
├── scaler.joblib                # Fitted StandardScaler
├── feature_cols.joblib          # Ordered list of the 19 feature columns
├── StockMarket_Analysis.ipynb   # Full training and evaluation notebook
├── requirements.txt
└── README.md
```

## Quick Start

**1 — Clone**
```bash
git clone https://github.com/ps636718/StockPredictor.git
cd StockPredictor
```

**2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**3 — Run the app**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Enter any NSE ticker (e.g. `RELIANCE.NS`, `TCS.NS`, `GAIL.NS`), choose a forecast horizon of 1–7 days, and click **Predict**.

**4 — Retrain (optional)**

The full training pipeline is in `StockMarket_Analysis.ipynb`. Re-running it and re-exporting artifacts:
```python
joblib.dump(models, 'models.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(feature_cols, 'feature_cols.joblib')
```

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Trading date |
| `Close`, `High`, `Low`, `Open` | float | Raw daily OHLC prices |
| `Volume` | int | Daily traded volume |
| `Ticker` | string | NSE ticker symbol |

Training data spans 2019-01-01 onward across 19 tickers — RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, GAIL, TATASTEEL, SBIN, AXISBANK, WIPRO, HCLTECH, ONGC, NTPC, MARUTI, SUNPHARMA, DRREDDY, CIPLA, ITC, LT (all `.NS`) — producing 35,132 rows after feature engineering. Data is split by time — training on data before 2024-01-01, testing on 2024 onward — to avoid look-ahead bias.

## Methodology Deep Dive

**Why a separate model per day?**
A single model that recursively predicts Day 2 from its own Day 1 prediction compounds error at every step. Training 7 independent regressors, each mapped directly from the same input features to its respective target day, avoids this error accumulation and lets each model specialize in its own horizon.

**Why time-based splitting?**
Randomly shuffling rows before splitting would let the model see future price information during training, since lag and rolling features are derived from surrounding rows. A strict time-based split — train on data before 2024, test on 2024 onward — better reflects real-world deployment, where only past data is available at prediction time.

**Why normalize per ticker?**
Absolute price levels vary enormously across stocks (a ₹50 stock vs. a ₹3,000 stock). Normalizing each ticker's prices relative to its own rolling mean puts all training examples on a comparable scale, letting one shared model family learn patterns that generalize across tickers rather than memorizing price ranges.

## Outputs

| File | Description |
|---|---|
| `models.joblib` | Dictionary of 7 trained XGBoost regressors, keyed by forecast day |
| `scaler.joblib` | Fitted StandardScaler used to scale the 19 input features |
| `feature_cols.joblib` | Ordered list of feature column names expected by the models |

## Future Work

- [ ] Extend forecast horizon beyond 7 days with recalibrated confidence bands
- [ ] Add prediction intervals (quantile regression) instead of point estimates
- [ ] Incorporate sector- or index-level features (e.g. Nifty 50 correlation)
- [ ] Backtest a simple trading strategy driven by the model's forecasts
- [ ] Add automated retraining on a rolling data window
- [ ] Deploy as a hosted API in addition to the Streamlit interface

## Disclaimer

This project is intended for educational and research purposes only. Predictions generated by this application should not be treated as financial advice. Equity markets are influenced by numerous factors beyond historical price and volume data. Always consult a qualified financial advisor before making investment decisions.

## Author

**Pawan Suman**
GitHub: [github.com/ps636718](https://github.com/ps636718)
