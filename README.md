# StockPredictor
# 📈 Stock Price Predictor

A Machine Learning based web application that predicts **GAIL.NS stock prices** for the next **1-14 trading days** using XGBoost and technical indicators.

---

## 🚀 Live Demo

> Run locally using Streamlit

---

## 📌 Features

- Predicts next 1-14 trading days closing price
- Skips weekends automatically (only trading days)
- Confidence score for each prediction (High / Medium / Low)
- Interactive chart with historical + predicted prices
- Day-wise prediction table
- Real-time data fetching using yfinance

---

## 🧠 How It Works

```
yfinance → Raw Data
    ↓
Data Cleaning (Missing values, Volume check)
    ↓
Feature Engineering (SMA, EMA, RSI, MACD, Lag features)
    ↓
StandardScaler (Normalization)
    ↓
29 XGBoost Models (One per day)
    ↓
Optuna Hyperparameter Tuning
    ↓
Streamlit Web App
```

---

## 📊 Model Performance

| Day Range | R2 Score | Confidence |
|-----------|----------|------------|
| Day 1-7   | 0.94 - 0.73 | 🟢 High |
| Day 8-14  | 0.70 - 0.62 | 🟡 Medium |
| Day 15-29 | 0.60 - 0.31 | 🔴 Low |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML Model | XGBoost |
| Tuning | Optuna |
| Frontend | Streamlit |
| Charts | Plotly |
| Data | yfinance |
| Scaling | Scikit-learn |
| TA Features | ta library |

---

## 📁 Project Structure

```
StockPredictor/
├── app.py              # Main Streamlit app
├── models.joblib       # Trained XGBoost models (29)
├── scaler.joblib       # Fitted StandardScaler
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Features Used

| Feature | Description |
|---------|-------------|
| SMA_7 | 7-day Simple Moving Average |
| EMA_12 | 12-day Exponential Moving Average |
| EMA_26 | 26-day Exponential Moving Average |
| RSI | Relative Strength Index (14-day) |
| MACD | Moving Average Convergence Divergence |
| Lag_1,2,3,7 | Previous day closing prices |
| Daily_Return | Day over day % change |
| Close_to_Open | Close - Open price difference |
| std_5 | 5-day price volatility |
| Price_ROC | 5-day Rate of Change |
| Relative_Volume | Volume vs 20-day average |
| Volume_log | Log transformed volume |

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOURUSERNAME/StockPredictor.git
cd StockPredictor
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
yfinance
pandas
numpy
xgboost
scikit-learn
joblib
plotly
ta
```

---

## 📈 Data

- **Stock:** GAIL.NS (NSE Listed)
- **Source:** Yahoo Finance (yfinance)
- **Training Period:** 2019-01-01 to 2024-12-31
- **Test Period:** 2025-01-01 to 2026-05-27
- **Total Rows:** 1797 (after cleaning)
- **Features:** 20
- **Targets:** 29 (Day 1 to Day 29)

---

## 🤖 Model Details

- **Algorithm:** XGBoost Regressor
- **Models:** 29 (one per prediction day)
- **Tuning:** Optuna (50 trials)
- **Best Parameters:**
```python
{
    'n_estimators'    : 100,
    'max_depth'       : 3,
    'learning_rate'   : 0.075,
    'subsample'       : 0.90,
    'colsample_bytree': 0.91
}
```

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**.
> Stock market predictions are inherently uncertain.
> **Do not use this for actual investment decisions.**
> Always consult a financial advisor before investing.

---

## 👨‍💻 Author

- **Name:** Pawan Suman
- **GitHub:** github.com/ps636718
- **LinkedIn:** www.linkedin.com/in/pawan-suman-b59512330

---

## 📄 License

MIT License — Free to use and modify.
