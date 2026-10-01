from typing import Dict, Any, List
import numpy as np
import pandas as pd


def generate_trading_signal(
    current_price: float,
    xgb_forecast: Dict[str, Any],
    lstm_forecast: Dict[str, Any],
    feat_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Synthesizes multi-factor technical analysis and AI model predictions
    into actionable trading signals with target price, stop loss, risk management,
    key levels, position sizing, and detailed rationale.
    """
    if feat_df.empty or "forecasts" not in xgb_forecast:
        return {
            "action": "HOLD",
            "action_class": "action-hold",
            "confluence_score": 0.0,
            "confidence_pct": 50,
            "target_price": current_price,
            "stop_loss": round(current_price * 0.97, 2),
            "risk_level": "MODERATE",
            "risk_reward_ratio": "1:1",
            "volatility_regime": "NORMAL",
            "suggested_position_size": "50%",
            "key_levels": {
                "support_1": round(current_price * 0.97, 2),
                "support_2": round(current_price * 0.94, 2),
                "resistance_1": round(current_price * 1.03, 2),
                "resistance_2": round(current_price * 1.06, 2),
            },
            "drivers": ["Insufficient technical data to formulate high-confidence signal."]
        }

    latest = feat_df.iloc[-1]
    rsi = float(latest.get("RSI", 50.0))
    macd_val = float(latest.get("MACD", 0.0))
    daily_return = float(latest.get("Daily_Return", 0.0))
    volatility = float(latest.get("std_5", 1.5))
    close_norm = float(latest.get("Close_norm", 100.0))
    sma_7 = float(latest.get("SMA_7", 100.0))

    xgb_7d_change = float(xgb_forecast.get("overall_7d_change", 0.0))
    lstm_7d_change = float(lstm_forecast.get("overall_7d_change", 0.0)) if "overall_7d_change" in lstm_forecast else xgb_7d_change
    blended_forecast_change = (0.6 * xgb_7d_change) + (0.4 * lstm_7d_change)

    drivers: List[str] = []
    confluence_score = 0.0  # -100 to +100

    # ── Factor 1: AI 7-day Forecast (Weight: 45) ──────────────────────
    forecast_points = float(np.clip(blended_forecast_change * 9.0, -45.0, 45.0))
    confluence_score += forecast_points
    if blended_forecast_change > 1.5:
        drivers.append(f"AI ensemble (XGBoost + LSTM) projects strong upside of +{blended_forecast_change:.1f}% over the next 7 trading days.")
    elif blended_forecast_change < -1.5:
        drivers.append(f"AI ensemble indicates short-term downward pressure of {blended_forecast_change:.1f}% over the next 7 trading days.")
    else:
        drivers.append(f"AI models forecast steady consolidation ({blended_forecast_change:+.1f}% expected 7-day variation).")

    # ── Factor 2: RSI Indicator (Weight: 25) ──────────────────────────
    if rsi < 32:
        confluence_score += 25.0
        drivers.append(f"RSI is currently {rsi:.1f} (Oversold), indicating an impending bullish mean-reversion opportunity.")
    elif rsi < 45:
        confluence_score += 12.0
        drivers.append(f"RSI at {rsi:.1f} shows reasonable valuation buffer without excessive speculative froth.")
    elif rsi > 68:
        confluence_score -= 25.0
        drivers.append(f"RSI has reached {rsi:.1f} (Overbought territory), signaling elevated risk of near-term profit taking.")
    elif rsi > 58:
        confluence_score -= 10.0
        drivers.append(f"RSI is elevated at {rsi:.1f}, warranting caution before entering aggressive long positions.")
    else:
        drivers.append(f"RSI is neutral at {rsi:.1f}, reflecting balanced supply and demand dynamics.")

    # ── Factor 3: MACD Momentum (Weight: 20) ──────────────────────────
    if macd_val > 0.05:
        confluence_score += 18.0
        drivers.append("MACD histogram displays positive momentum with expanding bullish separation.")
    elif macd_val < -0.05:
        confluence_score -= 18.0
        drivers.append("MACD remains in negative divergence territory, recommending defensive positioning.")
    else:
        drivers.append("MACD is hovering near the zero equilibrium centerline.")

    # ── Factor 4: Moving Average Trend (Weight: 10) ───────────────────
    if close_norm > sma_7:
        confluence_score += 10.0
        drivers.append("Price is trading above its 7-day short-term moving average.")
    else:
        confluence_score -= 10.0
        drivers.append("Price is currently tracking below its 7-day moving average.")

    # ── Map score to action badge ─────────────────────────────────────
    if confluence_score >= 40:
        action = "STRONG BUY"
        action_class = "action-strong-buy"
    elif confluence_score >= 15:
        action = "BUY"
        action_class = "action-buy"
    elif confluence_score <= -40:
        action = "STRONG SELL"
        action_class = "action-strong-sell"
    elif confluence_score <= -15:
        action = "SELL"
        action_class = "action-sell"
    else:
        action = "HOLD"
        action_class = "action-hold"

    # ── Target Price: 7th day predicted price ─────────────────────────
    xgb_target = xgb_forecast["forecasts"][-1]["predicted_price"] if "forecasts" in xgb_forecast and xgb_forecast["forecasts"] else current_price
    target_price = round(xgb_target, 2)

    # ── Dynamic Stop Loss ─────────────────────────────────────────────
    if action in ["BUY", "STRONG BUY"]:
        stop_loss = round(current_price * (1.0 - min(max(volatility * 0.02, 0.025), 0.055)), 2)
    elif action in ["SELL", "STRONG SELL"]:
        stop_loss = round(current_price * (1.0 + min(max(volatility * 0.02, 0.025), 0.055)), 2)
    else:
        stop_loss = round(current_price * 0.965, 2)

    # ── Volatility Regime Classification ──────────────────────────────
    if volatility > 4.0 or abs(daily_return) > 0.06:
        volatility_regime = "EXTREME_VOL"
    elif volatility > 2.5 or abs(daily_return) > 0.04:
        volatility_regime = "HIGH_VOL"
    elif volatility > 1.2 or abs(daily_return) > 0.015:
        volatility_regime = "NORMAL"
    else:
        volatility_regime = "LOW_VOL"

    # ── Risk Level Assessment ─────────────────────────────────────────
    if volatility_regime in ["EXTREME_VOL", "HIGH_VOL"]:
        risk_level = "HIGH"
    elif volatility_regime == "NORMAL":
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # ── Risk / Reward Ratio ───────────────────────────────────────────
    if action in ["BUY", "STRONG BUY"]:
        potential_gain = abs(target_price - current_price)
        potential_loss = abs(current_price - stop_loss)
        if potential_loss > 0:
            rr_ratio = potential_gain / potential_loss
            risk_reward_ratio = f"1:{rr_ratio:.1f}"
        else:
            risk_reward_ratio = "1:∞"
    elif action in ["SELL", "STRONG SELL"]:
        potential_gain = abs(current_price - target_price)
        potential_loss = abs(stop_loss - current_price)
        if potential_loss > 0:
            rr_ratio = potential_gain / potential_loss
            risk_reward_ratio = f"1:{rr_ratio:.1f}"
        else:
            risk_reward_ratio = "1:∞"
    else:
        risk_reward_ratio = "N/A (HOLD)"

    # ── Suggested Position Sizing ─────────────────────────────────────
    if risk_level == "HIGH":
        suggested_position_size = "25% (Conservative)"
    elif risk_level == "MODERATE" and abs(confluence_score) > 30:
        suggested_position_size = "50% (Moderate)"
    elif risk_level == "LOW" and abs(confluence_score) > 40:
        suggested_position_size = "75% (Aggressive)"
    elif risk_level == "LOW":
        suggested_position_size = "50% (Moderate)"
    else:
        suggested_position_size = "25% (Conservative)"

    # ── Key Support & Resistance Levels ───────────────────────────────
    # Calculate from recent price action and volatility
    vol_spread = current_price * (volatility * 0.01)
    key_levels = {
        "support_1": round(current_price - vol_spread, 2),
        "support_2": round(current_price - (vol_spread * 2.2), 2),
        "resistance_1": round(current_price + vol_spread, 2),
        "resistance_2": round(current_price + (vol_spread * 2.2), 2),
        "pivot": round(current_price, 2),
    }

    # ── Confidence Score ──────────────────────────────────────────────
    confidence = int(min(max(abs(confluence_score) * 0.8 + 35, 40), 96))

    return {
        "action": action,
        "action_class": action_class,
        "confluence_score": round(confluence_score, 1),
        "confidence_pct": confidence,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "risk_level": risk_level,
        "risk_reward_ratio": risk_reward_ratio,
        "volatility_regime": volatility_regime,
        "suggested_position_size": suggested_position_size,
        "key_levels": key_levels,
        "drivers": drivers,
    }
