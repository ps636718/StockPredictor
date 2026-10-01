import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

# In-memory cache to avoid yfinance rate limits
# Key: f"{ticker}_{period}_{interval}" -> {"data": df/dict, "timestamp": unix_time}
_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

CURATED_STOCKS: List[Dict[str, str]] = [
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Telecom", "market": "NSE"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Technology", "market": "NSE"},
    {"ticker": "INFY.NS", "name": "Infosys Ltd", "sector": "Technology", "market": "NSE"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking & Finance", "market": "NSE"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking & Finance", "market": "NSE"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel Ltd", "sector": "Metals & Mining", "market": "NSE"},
    {"ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Public Banking", "market": "NSE"},
    {"ticker": "WIPRO.NS", "name": "Wipro Ltd", "sector": "Technology", "market": "NSE"},
    {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India", "sector": "Automotive", "market": "NSE"},
    {"ticker": "ITC.NS", "name": "ITC Ltd", "sector": "Consumer Goods", "market": "NSE"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automotive", "market": "NSE"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Semiconductors & AI", "market": "NASDAQ"},
    {"ticker": "AAPL", "name": "Apple Inc", "sector": "Consumer Electronics", "market": "NASDAQ"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Software & Cloud", "market": "NASDAQ"},
    {"ticker": "TSLA", "name": "Tesla Inc", "sector": "Automotive & Clean Energy", "market": "NASDAQ"},
]


def clean_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance dataframe structure across different versions."""
    if df.empty:
        return df

    # Handle MultiIndex columns (introduced in recent yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.reset_index()
    df.columns.name = None

    # Standardize column naming
    rename_map = {
        'Date': 'Date',
        'Datetime': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume',
    }
    df.rename(columns=rename_map, inplace=True)

    # Filter out 0 or negative volume rows if volume exists
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0]

    # Convert Date to ISO string
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    return df


def fetch_stock_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV historical dataframe with caching."""
    cache_key = f"hist_{ticker}_{period}_{interval}"
    now = time.time()

    if cache_key in _CACHE and (now - _CACHE[cache_key]["timestamp"]) < CACHE_TTL_SECONDS:
        return _CACHE[cache_key]["data"].copy()

    try:
        raw_df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if raw_df.empty:
            # Fallback attempt with ticker object
            ticker_obj = yf.Ticker(ticker)
            raw_df = ticker_obj.history(period=period, interval=interval)

        cleaned_df = clean_yfinance_df(raw_df)

        if not cleaned_df.empty:
            _CACHE[cache_key] = {"data": cleaned_df, "timestamp": now}

        return cleaned_df.copy()

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_stock_overview(ticker: str) -> Dict[str, Any]:
    """Fetch current quote and key fundamentals for a ticker."""
    cache_key = f"overview_{ticker}"
    now = time.time()

    if cache_key in _CACHE and (now - _CACHE[cache_key]["timestamp"]) < CACHE_TTL_SECONDS:
        return _CACHE[cache_key]["data"]

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
        fast_info = getattr(ticker_obj, "fast_info", None)

        # Get latest price and previous close
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        if current_price is None and fast_info is not None:
            current_price = getattr(fast_info, "last_price", None)
            previous_close = getattr(fast_info, "previous_close", None)

        # If still None, grab from history
        if current_price is None:
            hist = fetch_stock_history(ticker, period="5d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price

        current_price = float(current_price) if current_price is not None else 0.0
        previous_close = float(previous_close) if previous_close is not None else current_price

        change = current_price - previous_close
        change_pct = (change / previous_close * 100) if previous_close else 0.0

        currency = info.get("currency", "INR" if ticker.endswith(".NS") or ticker.endswith(".BO") else "USD")

        # Range calculations
        fifty_two_low = info.get("fiftyTwoWeekLow")
        fifty_two_high = info.get("fiftyTwoWeekHigh")

        if fifty_two_low is None and fast_info is not None:
            fifty_two_low = getattr(fast_info, "year_low", None)
            fifty_two_high = getattr(fast_info, "year_high", None)

        fifty_two_low = float(fifty_two_low) if fifty_two_low is not None else current_price * 0.7
        fifty_two_high = float(fifty_two_high) if fifty_two_high is not None else current_price * 1.3

        # Range position (0 to 100%)
        range_span = max(fifty_two_high - fifty_two_low, 1e-6)
        range_pct = min(max(((current_price - fifty_two_low) / range_span) * 100, 0), 100)

        overview = {
            "ticker": ticker.upper(),
            "name": info.get("shortName") or info.get("longName") or ticker.upper(),
            "sector": info.get("sector", "Equities"),
            "industry": info.get("industry", "Financial & Commercial"),
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "currency": currency,
            "day_low": round(float(info.get("dayLow", current_price * 0.99)), 2),
            "day_high": round(float(info.get("dayHigh", current_price * 1.01)), 2),
            "fifty_two_low": round(fifty_two_low, 2),
            "fifty_two_high": round(fifty_two_high, 2),
            "range_pct": round(range_pct, 1),
            "volume": int(info.get("volume") or info.get("regularMarketVolume") or 0),
            "avg_volume": int(info.get("averageVolume") or 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": round(float(info.get("trailingPE", 0)), 2) if info.get("trailingPE") else None,
            "beta": round(float(info.get("beta", 1.0)), 2) if info.get("beta") else 1.0,
            "dividend_yield": round(float(info.get("dividendYield", 0) * 100), 2) if info.get("dividendYield") else 0.0,
            "market_status": "CLOSED",  # Can be updated by client
        }

        _CACHE[cache_key] = {"data": overview, "timestamp": now}
        return overview

    except Exception as e:
        print(f"Error fetching overview for {ticker}: {e}")
        return {
            "ticker": ticker.upper(),
            "name": ticker.upper(),
            "sector": "Equity",
            "industry": "Market",
            "current_price": 100.0,
            "previous_close": 100.0,
            "change": 0.0,
            "change_pct": 0.0,
            "currency": "INR" if ticker.endswith(".NS") else "USD",
            "day_low": 98.0,
            "day_high": 102.0,
            "fifty_two_low": 80.0,
            "fifty_two_high": 120.0,
            "range_pct": 50.0,
            "volume": 100000,
            "avg_volume": 100000,
            "market_cap": 0,
            "pe_ratio": None,
            "beta": 1.0,
            "dividend_yield": 0.0,
            "market_status": "CLOSED",
        }
