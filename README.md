# 📈 AlphaTrade AI — Stock Prediction, IPO Intelligence & Financial Research Platform

A comprehensive Full-Stack AI-Powered Financial Intelligence Platform featuring **7-Day Multi-Horizon Stock Forecasting** (XGBoost, LSTM, GRU), **IPO Analysis Engine**, **Market Sentiment Scoring**, and **Financial Report Assistant**.

> **Academic Validation**: 1,026-window walk-forward validation across 19 NSE equities (2019–2024) achieving **70.67% Day-1 directional accuracy** — honestly benchmarked against the naive persistence baseline.

---

## 🌟 Platform Capabilities

### 1. 7-Day Multi-Horizon Stock Forecasting
- **XGBoost 7-Stage Regressors**: 7 independently tuned gradient-boosted models predicting each future day without recursive error compounding.
- **LSTM Neural Network**: 10-day historical sequence memory capturing non-linear market dynamics.
- **GRU Recurrent Network**: Gated Recurrent Unit architecture for efficient sequential pattern recognition.
- **252-Day Rolling Normalization**: Eliminates non-stationarity and price drift across stocks of vastly different magnitudes.
- **Dynamic Confidence Intervals**: 95% uncertainty fan chart bounds widening with horizon (±2σ√t).

### 2. IPO Intelligence Center
- **40+ Real Indian IPOs** analyzed with structured scoring (Tata Technologies, Mankind Pharma, Bajaj Housing Finance, Ola Electric, FirstCry, etc.)
- **AI Subscribe/Avoid Recommendations** based on: subscription demand (30%), GMP trend (20%), P/E vs sector (25%), promoter holding (15%), and financials (10%).
- **Historical IPO Performance Tracking** with listing-day gain/loss analysis.

### 3. Market Sentiment Engine
- **Per-Stock Sentiment Scoring** (-1.0 to +1.0 scale with Very Bearish → Very Bullish classification)
- **News Headline Analysis**, social buzz indicators, and analyst consensus tracking.
- **Market Mood Index** for overall market temperature assessment.

### 4. Financial Report Research Assistant
- **Document Analysis**: Paste financial text (annual reports, earnings calls, news) and get structured AI-powered insights.
- **Pre-Built Research**: Key financial metrics, risk factors, and opportunity analysis for tracked stocks.
- **Revenue, Debt, Cash Flow, and Management Quality** assessment templates.

### 5. Multi-Factor AI Trading Signal Engine
- Synthesizes RSI momentum, MACD divergence, Moving Average trends, and AI forecast trajectories.
- **Ensemble Confluence Score** (-100 to +100) generating: `STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`.
- **Risk Management**: Dynamic stop loss, support/resistance levels, position sizing, risk-reward ratio, and volatility regime classification.

### 6. Modern Trading Terminal Interface
- Dark Mode Glassmorphic UI with neon status indicators and micro-interactions.
- Interactive Chart.js charting (Price, SMA 20/50/200, Bollinger Bands, Volume, RSI, MACD).
- Five navigation tabs: Market Terminal, AI Forecast, Model Benchmarks, IPO Intelligence, Research Assistant.
- Graceful offline fallback (Demo Mode) when live data is unavailable.

---

## 🏛️ System Architecture

```
stock-market/
├── backend/
│   ├── app.py                    # FastAPI REST API (15+ endpoints)
│   ├── main.py                   # Backend ASGI entrypoint for cloud runners
│   ├── data_engine.py            # yfinance data fetching, caching & fallback
│   ├── feature_pipeline.py       # 19 technical features & rolling normalization
│   ├── signals.py                # Enhanced trading signal generator
│   ├── train_models.py           # Offline model training pipeline
│   ├── ipo_engine.py             # IPO analysis with 40+ real IPOs
│   ├── sentiment_engine.py       # Market sentiment scoring engine
│   ├── report_assistant.py       # Financial report analysis assistant (Groq/OpenAI)
│   ├── demo_fixtures.py          # Offline demo data & fallback system
│   ├── requirements.txt          # Backend dependency declarations
│   └── models/
│       ├── xgb_engine.py         # 7-day multi-horizon XGBoost inference
│       ├── lstm_engine.py        # Sequential LSTM neural inference
│       └── saved_weights/        # Serialized models & benchmark JSON
├── frontend/
│   ├── index.html                # 5-tab financial trading dashboard
│   ├── styles.css                # Glassmorphic dark theme design system
│   ├── app.js                    # Interactive Chart.js client & UI controller
│   └── vercel.json               # Vercel deployment & rewrite configuration
├── .env.example                  # Environment variable template with placeholders
├── .gitignore                    # Ensures .env and secrets are never committed
├── main.py                       # Root ASGI entrypoint for cloud hosting
├── render.yaml                   # Render Blueprint web service configuration
├── run.py                        # CLI launcher (--train, --demo, --port)
├── requirements.txt              # Pinned project dependencies
└── README.md                     # Documentation and deployment guide
```

---

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
Python 3.10+ installed.

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY if using the AI Report Assistant
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Locally
```bash
python run.py
```

### Advanced Launch Options
```bash
python run.py --demo          # Force demo mode (no internet required)
python run.py --train         # Re-train models before starting
python run.py --port 9000     # Custom port
python run.py --no-browser    # Don't auto-open browser
```

---

## 🌐 Production Deployment (24/7 Cloud Hosting)

Follow these exact steps to deploy the terminal live on the web so anyone (recruiter, interviewer, client) can access it 24/7 from any device:

### Step 1: Push Codebase to GitHub
1. Initialize git and commit your files (the included `.gitignore` guarantees `.env` and secrets are never committed):
   ```bash
   git init
   git add .
   git commit -m "Prepare StockAI for 24/7 cloud deployment"
   ```
2. Create a new repository on [GitHub](https://github.com/new) and push your code:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```

---

### Step 2: Deploy Backend on Render.com
1. Sign in to [Render.com](https://render.com) using your GitHub account.
2. Click **New +** → **Web Service**.
3. Select your GitHub repository and configure the service:
   - **Name**: `stockai-backend` (or your choice of name)
   - **Region**: Singapore or Oregon
   - **Branch**: `main`
   - **Root Directory**: `backend` (or leave blank if using root `main.py`)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
4. Add the **Environment Variables** in Render Dashboard:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `GROQ_API_KEY` | `gsk_...` | Your API key from [Groq Console](https://console.groq.com/keys) |
   | `FRONTEND_URL` | `https://your-stockai-frontend.vercel.app` | Your deployed Vercel domain (or `*` during first build) |
   | `ALPHATRADE_DEMO_MODE` | `0` | `0` for live market data via yfinance |
   | `PYTHON_VERSION` | `3.11.8` | Recommended Python version |
5. Click **Create Web Service**. Wait 2–3 minutes for build completion.
6. Note down your backend URL (e.g., `https://stockai-backend.onrender.com`).
   - Test it by visiting: `https://stockai-backend.onrender.com/health`
   - You should see: `{"status":"healthy","version":"2.5.0",...}`

---

### Step 3: Deploy Frontend on Vercel
1. Sign in to [Vercel.com](https://vercel.com) using your GitHub account.
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. In the configuration screen:
   - **Project Name**: `stockai-terminal`
   - **Framework Preset**: `Other`
   - **Root Directory**: Click *Edit* and select `frontend`
5. Click **Deploy**. Vercel will deploy your static frontend in seconds.
6. Note down your frontend URL (e.g., `https://stockai-terminal.vercel.app`).

---

### Step 4: Link Frontend to Live Backend & Update CORS
1. In `frontend/app.js`, update line 8 with your live Render backend URL:
   ```javascript
   const API_BASE_URL = "https://stockai-backend.onrender.com";
   ```
2. Commit and push the update to GitHub:
   ```bash
   git add frontend/app.js
   git commit -m "Point API_BASE_URL to live Render backend"
   git push origin main
   ```
   *(Vercel automatically detects the commit and redeploys the frontend in ~15 seconds).*
3. In your **Render Dashboard** → **Environment**, ensure `FRONTEND_URL` matches your exact Vercel URL (e.g., `https://stockai-terminal.vercel.app`). Save changes.

---

### Step 5: How to Verify from a Different Device
1. Open your smartphone, tablet, or another computer (or an Incognito window).
2. Navigate to your live Vercel URL: `https://stockai-terminal.vercel.app`.
3. Verify that:
   - Live ticker cards load with real NSE/NASDAQ quotes.
   - Interactive Chart.js price charts render with technical indicators (SMA, Bollinger).
   - Click **AI Forecast** to verify 7-day multi-horizon price trajectories and trading signals.
   - Click **IPO Intelligence** to verify upcoming IPO cards and historical performance.
   - Click **Research Assistant** to run an AI explainer query using Groq Llama-3.1.
   - The top status pill shows "System Online" with green indicators.

---

### Step 6: Prevent Free-Tier Backend Sleeping (24/7 Awake Setup)
Render free-tier web services spin down after 15 minutes of inactivity, causing a 30–50 second cold start on the next visit. Use a free external cron ping service to keep it active 24/7:
1. Create a free account on [cron-job.org](https://cron-job.org).
2. Go to **Cronjobs** → **Create Cronjob**:
   - **Title**: `Keep StockAI Backend Awake`
   - **URL**: `https://stockai-backend.onrender.com/health`
   - **Execution Schedule**: Every `10 minutes` (or `14 minutes`)
   - **Request Method**: `GET`
3. Click **Create**.
4. **Outcome**: The lightweight `/health` endpoint is pinged every 10 minutes, keeping your container warm so visitors experience instantaneous (<1 second) response times 24/7!

---

## 📊 Empirical Benchmarks

### Walk-Forward Validation (Production Validation)
**Method**: Rolling 750-day train / 21-day test / 21-day step across 19 NSE stocks (2019–2024).
**Total windows evaluated**: 1,026 independent forward-test windows.

| Horizon | Directional Accuracy | Model R² (Pooled) | Naive R² (Pooled) | Model MAE |
| :---: | :---: | :---: | :---: | :---: |
| **Day 1** | **70.67%** | 0.9763 | 0.9845 | 1.40 |
| **Day 2** | **64.63%** | 0.9568 | 0.9698 | 1.99 |
| **Day 3** | **61.13%** | 0.9367 | 0.9551 | 2.47 |
| **Day 4** | **59.28%** | 0.9162 | 0.9407 | 2.88 |
| **Day 5** | **57.60%** | 0.8948 | 0.9261 | 3.24 |
| **Day 6** | **57.14%** | 0.8748 | 0.9123 | 3.55 |
| **Day 7** | **57.07%** | 0.8564 | 0.8990 | 3.82 |

### Static Split Comparison (Architecture Benchmark)

| Horizon | Naive R² | XGBoost R² | LSTM R² | GRU R² | XGBoost MAE |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Day 1** | 0.9782 | **0.9856** | 0.9812 | 0.9798 | **1.18** |
| **Day 4** | 0.9238 | **0.9484** | 0.9441 | 0.9420 | **2.38** |
| **Day 7** | 0.8726 | **0.9123** | 0.9024 | 0.9001 | **3.31** |

> **Key Research Finding**: Walk-forward R² is consistently *lower* than the naive persistence baseline — as expected in efficient markets. The model's genuine predictive edge is **directional classification** (70.67% Day-1 accuracy, well above the 50% random-walk baseline), not absolute price-level prediction. This honest assessment distinguishes this project from overfitted demonstrations.

---

## 🔌 REST API Endpoints

### Core Stock Data
- `GET /api/health` — System health check
- `GET /api/system/status` — Full system diagnostics
- `GET /api/stocks` — Curated watchlist (NSE + Global)
- `GET /api/stock/{ticker}/overview` — Fundamentals, live quote, P/E, market cap
- `GET /api/stock/{ticker}/history?period=1y` — OHLCV with SMA/Bollinger/RSI/MACD
- `GET /api/stock/{ticker}/predict` — 7-day AI forecast + trading signal
- `GET /api/benchmark` — Walk-forward validation + model comparison

### IPO Intelligence
- `GET /api/ipo/upcoming` — Upcoming IPOs with AI recommendations
- `GET /api/ipo/history` — Historical IPO performance data
- `GET /api/ipo/{name}` — Detailed IPO analysis report

### Sentiment Analysis
- `GET /api/sentiment/{ticker}` — Per-stock sentiment scoring
- `GET /api/sentiment/market/mood` — Overall market mood index

### Financial Research
- `GET /api/reports/insights/{ticker}` — Pre-built stock research insights
- `POST /api/reports/analyze` — Analyze submitted financial text

---

## ⚠️ Disclaimer

This is a **research and educational demonstration system**. It is **not investment advice**. The accuracy numbers reflect genuine academic methodology with honest limitations clearly stated. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.
