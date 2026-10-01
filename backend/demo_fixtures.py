"""
Demo Fixtures Module
Provides offline fallback data for 5 Indian stocks.
"""
from typing import Dict, List, Any
import datetime
import random

class DemoFixtures:
    """Provides demo data for offline mode."""
    
    def __init__(self):
        self._overview_data = {
            "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy", "price": 2850.5, "change": 1.2, "market_cap": "19.5T", "pe_ratio": 28.4},
            "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT", "price": 3950.0, "change": -0.5, "market_cap": "14.2T", "pe_ratio": 32.1},
            "INFY.NS": {"name": "Infosys", "sector": "IT", "price": 1650.2, "change": 0.8, "market_cap": "6.8T", "pe_ratio": 24.5},
            "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking", "price": 1620.0, "change": 1.5, "market_cap": "12.3T", "pe_ratio": 16.8},
            "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Metals", "price": 150.5, "change": -1.2, "market_cap": "1.8T", "pe_ratio": 12.4}
        }
        
    def is_demo_mode(self) -> bool:
        """Returns True if running in demo mode (no live data)."""
        return True

    def get_demo_overview(self, ticker: str) -> Dict[str, Any]:
        """Returns overview data for a demo ticker."""
        return self._overview_data.get(ticker, {
            "name": ticker, 
            "sector": "Unknown", 
            "price": 100.0, 
            "change": 0.0,
            "market_cap": "N/A",
            "pe_ratio": 0.0
        })

    def get_demo_history(self, ticker: str) -> List[Dict[str, Any]]:
        """Returns 30 days of OHLCV history."""
        base_price = self.get_demo_overview(ticker)["price"]
        history = []
        # Fix seed for reproducibility in demo mode per ticker
        random.seed(hash(ticker))
        today = datetime.date.today()
        for i in range(30, 0, -1):
            date = today - datetime.timedelta(days=i)
            # Skip weekends
            if date.weekday() > 4:
                continue
                
            open_p = base_price * (1 + random.uniform(-0.015, 0.015))
            high_p = open_p * (1 + random.uniform(0.001, 0.02))
            low_p = open_p * (1 - random.uniform(0.001, 0.02))
            close_p = open_p * (1 + random.uniform(-0.015, 0.015))
            vol = int(random.uniform(1000000, 5000000))
            
            history.append({
                "date": date.isoformat(),
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": vol
            })
            base_price = close_p
        
        # Reset seed
        random.seed()
        return history

    def get_demo_prediction(self, ticker: str) -> Dict[str, Any]:
        """Returns a sample prediction."""
        random.seed(hash(ticker + str(datetime.date.today())))
        base_price = self.get_demo_overview(ticker)["price"]
        change_pct = random.uniform(-0.05, 0.05)
        pred_price = base_price * (1 + change_pct)
        
        if change_pct > 0.02:
            signal = "BUY"
        elif change_pct < -0.02:
            signal = "SELL"
        else:
            signal = "HOLD"
            
        res = {
            "ticker": ticker,
            "current_price": base_price,
            "predicted_price_7d": round(pred_price, 2),
            "expected_change_pct": round(change_pct * 100, 2),
            "signal": signal,
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "model_used": "XGBoost 7-day + LSTM Ensemble"
        }
        random.seed()
        return res

demo_fixtures = DemoFixtures()
