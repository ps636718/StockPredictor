"""
Market Sentiment Scorer
Generates realistic sentiment data for stocks.
"""
from typing import Dict, List, Any
import datetime
import hashlib

class SentimentEngine:
    """Generates sentiment scores and insights for stocks."""
    
    def __init__(self):
        self.news_templates = {
            "IT": [
                "{company} secures multi-million dollar digital transformation deal",
                "Analyst downgrades {company} on margin pressure concerns",
                "{company} announces new AI innovation hub",
                "Attrition rates stabilize at {company} in Q3",
                "European revenue growth drives {company} earnings beat"
            ],
            "Banking": [
                "{company} reports strong credit growth across segments",
                "NPA concerns limit upside for {company}",
                "{company} hikes deposit rates to attract liquidity",
                "Retail banking division of {company} shows robust performance",
                "RBI regulations may impact {company}'s fee income"
            ],
            "Energy": [
                "{company} announces major renewable energy expansion",
                "Crude price volatility impacts {company} margins",
                "Government windfall tax tweaks affect {company}",
                "{company} commissions new green hydrogen plant",
                "Strong refining margins boost {company} profits"
            ],
            "Metals": [
                "Global steel prices pressure {company} realizations",
                "{company} plans capacity expansion at flagship plant",
                "Strong domestic infrastructure demand aids {company}",
                "Chinese export data raises concerns for {company}",
                "{company} focuses on deleveraging balance sheet"
            ]
        }
        
    def _get_deterministic_random(self, seed_str: str) -> float:
        """Returns a deterministic random float between 0 and 1 based on the seed."""
        hash_obj = hashlib.md5(seed_str.encode())
        hash_int = int(hash_obj.hexdigest()[:8], 16)
        return hash_int / 0xFFFFFFFF

    def get_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Returns sentiment analysis for a given ticker."""
        # Use current date to ensure sentiment changes daily but is consistent for the day
        date_str = datetime.date.today().isoformat()
        seed = f"{ticker}_{date_str}"
        
        # Base sentiment score (-1.0 to 1.0)
        raw_val = self._get_deterministic_random(seed)
        score = (raw_val * 2) - 1.0
        
        # Determine label
        if score > 0.6:
            label = "Very Bullish"
        elif score > 0.2:
            label = "Bullish"
        elif score > -0.2:
            label = "Neutral"
        elif score > -0.6:
            label = "Bearish"
        else:
            label = "Very Bearish"
            
        # Social buzz (Low, Medium, High)
        buzz_val = self._get_deterministic_random(seed + "_buzz")
        if buzz_val > 0.7:
            buzz = "High"
        elif buzz_val > 0.3:
            buzz = "Medium"
        else:
            buzz = "Low"
            
        # Analyst Consensus
        consensus_val = self._get_deterministic_random(seed + "_consensus")
        if consensus_val > 0.6:
            consensus = "Buy"
        elif consensus_val > 0.3:
            consensus = "Hold"
        else:
            consensus = "Sell"
            
        # Generate fake news snippets
        company_name = ticker.replace(".NS", "")
        sector = "IT" if "TCS" in ticker or "INFY" in ticker else "Banking" if "BANK" in ticker else "Energy" if "RELIANCE" in ticker else "Metals" if "STEEL" in ticker else "IT"
        templates = self.news_templates.get(sector, self.news_templates["IT"])
        
        news = []
        for i in range(3):
            # Select template deterministically
            t_idx = int(self._get_deterministic_random(f"{seed}_news_{i}") * len(templates))
            news.append(templates[t_idx].format(company=company_name))
            
        return {
            "ticker": ticker,
            "score": round(score, 2),
            "label": label,
            "news_snippets": list(set(news)),  # Remove potential duplicates
            "social_buzz": buzz,
            "analyst_consensus": consensus,
            "date": date_str
        }
        
    def get_market_mood(self) -> Dict[str, Any]:
        """Returns overall market sentiment."""
        date_str = datetime.date.today().isoformat()
        market_val = self._get_deterministic_random(f"NIFTY50_{date_str}")
        score = (market_val * 2) - 1.0
        
        if score > 0.5:
            mood = "Greed"
        elif score > 0.1:
            mood = "Optimistic"
        elif score > -0.1:
            mood = "Neutral"
        elif score > -0.5:
            mood = "Cautious"
        else:
            mood = "Fear"
            
        return {
            "index": "NIFTY 50",
            "mood_index": round((score + 1.0) * 50, 2), # 0 to 100 scale
            "label": mood,
            "date": date_str
        }

sentiment_engine = SentimentEngine()
