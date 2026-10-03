"""
Demo Fixtures Module
Provides offline fallback data for Indian and Global stocks with schema parity to live engines.
"""
from typing import Dict, List, Any
import datetime
import math
import random


class DemoFixtures:
    """Provides complete demo fixtures for offline mode and API fallback."""

    def __init__(self):
        self._overview_data = {
            "RELIANCE.NS": {
                "ticker": "RELIANCE.NS",
                "name": "Reliance Industries",
                "sector": "Energy & Telecom",
                "industry": "Oil, Gas & Telecom",
                "current_price": 2985.40,
                "previous_close": 2951.20,
                "change": 34.20,
                "change_pct": 1.16,
                "currency": "INR",
                "day_low": 2960.00,
                "day_high": 3010.50,
                "fifty_two_low": 2220.00,
                "fifty_two_high": 3217.90,
                "range_pct": 78.0,
                "volume": 4820000,
                "avg_volume": 5100000,
                "market_cap": 20180000000000,
                "pe_ratio": 28.4,
                "beta": 0.88,
                "dividend_yield": 0.35,
                "market_status": "CLOSED"
            },
            "TCS.NS": {
                "ticker": "TCS.NS",
                "name": "Tata Consultancy Services",
                "sector": "Technology",
                "industry": "IT Services & Consulting",
                "current_price": 3950.00,
                "previous_close": 3970.00,
                "change": -20.00,
                "change_pct": -0.50,
                "currency": "INR",
                "day_low": 3930.00,
                "day_high": 3985.00,
                "fifty_two_low": 3310.00,
                "fifty_two_high": 4550.00,
                "range_pct": 52.0,
                "volume": 2100000,
                "avg_volume": 2400000,
                "market_cap": 14200000000000,
                "pe_ratio": 32.1,
                "beta": 0.65,
                "dividend_yield": 1.15,
                "market_status": "CLOSED"
            },
            "INFY.NS": {
                "ticker": "INFY.NS",
                "name": "Infosys Ltd",
                "sector": "Technology",
                "industry": "IT Services",
                "current_price": 1650.20,
                "previous_close": 1637.10,
                "change": 13.10,
                "change_pct": 0.80,
                "currency": "INR",
                "day_low": 1635.00,
                "day_high": 1665.00,
                "fifty_two_low": 1350.00,
                "fifty_two_high": 1950.00,
                "range_pct": 50.0,
                "volume": 5600000,
                "avg_volume": 6200000,
                "market_cap": 6800000000000,
                "pe_ratio": 24.5,
                "beta": 0.75,
                "dividend_yield": 2.10,
                "market_status": "CLOSED"
            },
            "HDFCBANK.NS": {
                "ticker": "HDFCBANK.NS",
                "name": "HDFC Bank Ltd",
                "sector": "Banking & Finance",
                "industry": "Private Banking",
                "current_price": 1620.00,
                "previous_close": 1596.00,
                "change": 24.00,
                "change_pct": 1.50,
                "currency": "INR",
                "day_low": 1600.00,
                "day_high": 1635.00,
                "fifty_two_low": 1363.00,
                "fifty_two_high": 1794.00,
                "range_pct": 60.0,
                "volume": 14500000,
                "avg_volume": 16000000,
                "market_cap": 12300000000000,
                "pe_ratio": 16.8,
                "beta": 0.95,
                "dividend_yield": 1.20,
                "market_status": "CLOSED"
            },
            "TATASTEEL.NS": {
                "ticker": "TATASTEEL.NS",
                "name": "Tata Steel Ltd",
                "sector": "Metals & Mining",
                "industry": "Steel Manufacturing",
                "current_price": 150.50,
                "previous_close": 152.30,
                "change": -1.80,
                "change_pct": -1.18,
                "currency": "INR",
                "day_low": 149.00,
                "day_high": 153.50,
                "fifty_two_low": 115.00,
                "fifty_two_high": 184.00,
                "range_pct": 51.0,
                "volume": 25000000,
                "avg_volume": 28000000,
                "market_cap": 1850000000000,
                "pe_ratio": 12.4,
                "beta": 1.30,
                "dividend_yield": 2.40,
                "market_status": "CLOSED"
            }
        }

    def is_demo_mode(self) -> bool:
        return True

    def get_demo_overview(self, ticker: str) -> Dict[str, Any]:
        """Returns overview data with complete frontend schema parity."""
        t_clean = ticker.strip().upper()
        if t_clean in self._overview_data:
            return self._overview_data[t_clean].copy()

        # Dynamic fallback for unlisted ticker
        is_inr = t_clean.endswith(".NS") or t_clean.endswith(".BO")
        curr_p = 500.0 if is_inr else 150.0
        prev_p = round(curr_p * 0.99, 2)
        diff = round(curr_p - prev_p, 2)
        diff_pct = round((diff / prev_p) * 100, 2)

        return {
            "ticker": t_clean,
            "name": t_clean.replace(".NS", ""),
            "sector": "Equities",
            "industry": "Financial & Commercial",
            "current_price": curr_p,
            "previous_close": prev_p,
            "change": diff,
            "change_pct": diff_pct,
            "currency": "INR" if is_inr else "USD",
            "day_low": round(curr_p * 0.98, 2),
            "day_high": round(curr_p * 1.02, 2),
            "fifty_two_low": round(curr_p * 0.70, 2),
            "fifty_two_high": round(curr_p * 1.30, 2),
            "range_pct": 50.0,
            "volume": 2500000,
            "avg_volume": 3000000,
            "market_cap": 500000000000,
            "pe_ratio": 25.0,
            "beta": 1.0,
            "dividend_yield": 1.0,
            "market_status": "CLOSED"
        }

    def get_demo_history(self, ticker: str) -> List[Dict[str, Any]]:
        """Returns 60 trading days of OHLCV history with computed technical indicators."""
        ov = self.get_demo_overview(ticker)
        current_price = ov["current_price"]

        history = []
        today = datetime.date.today()

        # Deterministic variation
        days_generated = 0
        step_days = 90
        prices = []

        for i in range(step_days, 0, -1):
            date = today - datetime.timedelta(days=i)
            if date.weekday() > 4:
                continue

            # Deterministic pseudo-random based on day and ticker
            t_idx = days_generated
            trend = math.sin(t_idx / 8.0) * 0.02 + (t_idx * 0.0008)
            noise = math.cos(t_idx * 1.7) * 0.012

            price = current_price * (0.92 + trend + noise)
            prices.append(price)

            open_p = round(price * (1.0 - (math.sin(t_idx) * 0.005)), 2)
            high_p = round(max(open_p, price) * (1.0 + abs(math.cos(t_idx)) * 0.009), 2)
            low_p = round(min(open_p, price) * (1.0 - abs(math.sin(t_idx * 2)) * 0.009), 2)
            close_p = round(price, 2)
            vol = int(3000000 + math.sin(t_idx) * 1500000)

            history.append({
                "date": date.isoformat(),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol
            })
            days_generated += 1

        # Add technical overlays (SMA, BB, RSI, MACD)
        closes = [h["close"] for h in history]
        for idx, h in enumerate(history):
            # SMA 20, 50
            if idx >= 19:
                h["sma_20"] = round(sum(closes[idx - 19:idx + 1]) / 20.0, 2)
                sub_closes = closes[idx - 19:idx + 1]
                mean_20 = h["sma_20"]
                std_20 = math.sqrt(sum((x - mean_20) ** 2 for x in sub_closes) / 20.0)
                h["bb_middle"] = mean_20
                h["bb_upper"] = round(mean_20 + (2.0 * std_20), 2)
                h["bb_lower"] = round(mean_20 - (2.0 * std_20), 2)
            else:
                h["sma_20"] = h["close"]
                h["bb_middle"] = h["close"]
                h["bb_upper"] = round(h["close"] * 1.05, 2)
                h["bb_lower"] = round(h["close"] * 0.95, 2)

            h["sma_50"] = round(sum(closes[max(0, idx - 49):idx + 1]) / len(closes[max(0, idx - 49):idx + 1]), 2)
            h["sma_200"] = h["sma_50"]

            # RSI & MACD heuristics
            h["rsi"] = round(52.0 + math.sin(idx / 3.0) * 15.0, 1)
            h["macd_line"] = round(math.sin(idx / 4.0) * 3.5, 3)
            h["macd_signal"] = round(math.sin((idx - 2) / 4.0) * 3.2, 3)
            h["macd_hist"] = round(h["macd_line"] - h["macd_signal"], 3)

        return history

    def get_demo_prediction(self, ticker: str) -> Dict[str, Any]:
        """Returns 7-day multi-horizon forecast with complete production schema."""
        ov = self.get_demo_overview(ticker)
        current_price = ov["current_price"]
        today = datetime.date.today()

        history_30 = self.get_demo_history(ticker)
        last_10 = [
            {"date": h["date"], "price": h["close"]}
            for h in history_30[-10:]
        ]

        xgb_forecasts = []
        lstm_forecasts = []
        cum_gain = 0.0

        for day in range(1, 8):
            day_gain = (0.55 + (math.sin(day / 2.0) * 0.4))
            cum_gain += day_gain

            p_xgb = round(current_price * (1.0 + (cum_gain / 100.0)), 2)
            p_lstm = round(current_price * (1.0 + ((cum_gain * 0.95) / 100.0)), 2)

            spread = round(p_xgb * (0.015 * math.sqrt(day)), 2)

            xgb_forecasts.append({
                "day": day,
                "day_label": f"+{day}D",
                "predicted_price": p_xgb,
                "predicted_norm": round((p_xgb / current_price) * 100.0, 2),
                "change_pct": round(((p_xgb - current_price) / current_price) * 100.0, 2),
                "lower_bound": round(p_xgb - spread, 2),
                "upper_bound": round(p_xgb + spread, 2)
            })

            lstm_forecasts.append({
                "day": day,
                "day_label": f"+{day}D",
                "predicted_price": p_lstm,
                "predicted_norm": round((p_lstm / current_price) * 100.0, 2),
                "change_pct": round(((p_lstm - current_price) / current_price) * 100.0, 2)
            })

        overall_gain = xgb_forecasts[-1]["change_pct"]
        target_price = xgb_forecasts[-1]["predicted_price"]
        stop_loss = round(current_price * 0.965, 2)

        action = "STRONG BUY" if overall_gain > 3.0 else "BUY" if overall_gain > 1.0 else "HOLD"
        action_class = "action-strong-buy" if action == "STRONG BUY" else "action-buy" if action == "BUY" else "action-hold"

        signal = {
            "action": action,
            "action_class": action_class,
            "confluence_score": round(min(overall_gain * 18.0, 85.0), 1),
            "confidence_pct": 88,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "risk_level": "MODERATE",
            "risk_reward_ratio": "1:2.4",
            "volatility_regime": "NORMAL",
            "suggested_position_size": "50% (Moderate)",
            "key_levels": {
                "support_1": round(current_price * 0.98, 2),
                "support_2": round(current_price * 0.96, 2),
                "resistance_1": round(current_price * 1.02, 2),
                "resistance_2": round(current_price * 1.04, 2),
                "pivot": round(current_price, 2)
            },
            "drivers": [
                f"Multi-horizon models project consistent upward trajectory (+{overall_gain:.2f}% expected over 7 days).",
                "RSI at 54.2 confirms sustained bullish momentum without overbought exhaustion.",
                "Trading comfortably above 252-day normalized rolling baseline."
            ]
        }

        return {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "last_updated": today.isoformat(),
            "historical_anchor": last_10,
            "xgboost": {
                "model_name": "XGBoost Regressor",
                "current_price": current_price,
                "forecasts": xgb_forecasts,
                "overall_7d_change": overall_gain,
                "expected_target_price": target_price
            },
            "lstm": {
                "model_name": "LSTM Neural Network",
                "current_price": current_price,
                "forecasts": lstm_forecasts,
                "overall_7d_change": lstm_forecasts[-1]["change_pct"],
                "expected_target_price": lstm_forecasts[-1]["predicted_price"]
            },
            "signal": signal,
            "data_source": "demo"
        }


demo_fixtures = DemoFixtures()
