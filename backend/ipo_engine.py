"""
IPO Analysis Engine
Provides standardized data and analysis for Indian IPOs.
"""
from typing import Dict, List, Any, Optional

class IPOAnalyzer:
    """Engine for analyzing and scoring IPOs with a unified data shape."""

    def __init__(self):
        # Base sector PE medians
        self.sector_medians = {
            "IT": 30.0,
            "Pharma": 35.0,
            "Finance": 15.0,
            "Tech": 50.0,
            "Consumer": 45.0,
            "Auto": 25.0,
            "E-commerce": 80.0,
            "Logistics": 40.0,
            "Insurance": 25.0,
            "Healthcare": 40.0,
            "Real Estate": 30.0,
            "Travel": 60.0,
            "FMCG": 55.0,
            "Manufacturing": 35.0,
            "Retail": 45.0,
            "Energy": 20.0,
            "Transport": 25.0,
            "Paints": 70.0,
            "Food": 60.0,
            "Consumer Tech": 45.0,
        }

        # Raw historical data (40 real Indian IPOs)
        self._raw_history = [
            {"name": "Tata Technologies", "year": 2023, "issue_price": 500, "listing_price": 1200, "gmp": 350, "qib": 203.4, "nii": 62.1, "rii": 16.5, "sector": "IT", "pe_ratio": 32.5, "promoter_holding": 79.5, "lot_size": 30, "issue_size_cr": 3042},
            {"name": "Mankind Pharma", "year": 2023, "issue_price": 1080, "listing_price": 1300, "gmp": 120, "qib": 49.1, "nii": 3.8, "rii": 0.9, "sector": "Pharma", "pe_ratio": 30.0, "promoter_holding": 79.0, "lot_size": 13, "issue_size_cr": 4326},
            {"name": "IREDA", "year": 2023, "issue_price": 32, "listing_price": 50, "gmp": 12, "qib": 104.5, "nii": 24.1, "rii": 7.7, "sector": "Finance", "pe_ratio": 8.5, "promoter_holding": 75.0, "lot_size": 460, "issue_size_cr": 2150},
            {"name": "Jio Financial Services", "year": 2023, "issue_price": 261, "listing_price": 265, "gmp": 0, "qib": 1.0, "nii": 1.0, "rii": 1.0, "sector": "Finance", "pe_ratio": 0.0, "promoter_holding": 46.7, "lot_size": 1, "issue_size_cr": 166000},
            {"name": "Zaggle", "year": 2023, "issue_price": 164, "listing_price": 164, "gmp": 0, "qib": 16.7, "nii": 8.8, "rii": 5.9, "sector": "Tech", "pe_ratio": 45.0, "promoter_holding": 44.0, "lot_size": 90, "issue_size_cr": 563},
            {"name": "Cello World", "year": 2023, "issue_price": 648, "listing_price": 831, "gmp": 150, "qib": 108.9, "nii": 24.4, "rii": 3.0, "sector": "Consumer", "pe_ratio": 42.0, "promoter_holding": 78.0, "lot_size": 23, "issue_size_cr": 1900},
            {"name": "Ola Electric", "year": 2024, "issue_price": 76, "listing_price": 76, "gmp": 15, "qib": 5.3, "nii": 2.4, "rii": 3.9, "sector": "Auto", "pe_ratio": -15.0, "promoter_holding": 36.7, "lot_size": 195, "issue_size_cr": 6145},
            {"name": "FirstCry", "year": 2024, "issue_price": 465, "listing_price": 625, "gmp": 80, "qib": 19.3, "nii": 4.6, "rii": 2.3, "sector": "E-commerce", "pe_ratio": -40.0, "promoter_holding": 46.0, "lot_size": 32, "issue_size_cr": 4193},
            {"name": "Bajaj Housing Finance", "year": 2024, "issue_price": 70, "listing_price": 150, "gmp": 75, "qib": 209.3, "nii": 41.5, "rii": 7.0, "sector": "Finance", "pe_ratio": 24.5, "promoter_holding": 88.7, "lot_size": 214, "issue_size_cr": 6560},
            {"name": "Zomato", "year": 2021, "issue_price": 76, "listing_price": 115, "gmp": 20, "qib": 51.7, "nii": 32.9, "rii": 7.4, "sector": "E-commerce", "pe_ratio": -100.0, "promoter_holding": 0.0, "lot_size": 195, "issue_size_cr": 9375},
            {"name": "Nykaa", "year": 2021, "issue_price": 1125, "listing_price": 2001, "gmp": 600, "qib": 91.1, "nii": 112.0, "rii": 12.2, "sector": "E-commerce", "pe_ratio": 800.0, "promoter_holding": 52.5, "lot_size": 12, "issue_size_cr": 5352},
            {"name": "Paytm", "year": 2021, "issue_price": 2150, "listing_price": 1955, "gmp": -50, "qib": 2.7, "nii": 0.2, "rii": 1.6, "sector": "Fintech", "pe_ratio": -50.0, "promoter_holding": 0.0, "lot_size": 6, "issue_size_cr": 18300},
            {"name": "LIC", "year": 2022, "issue_price": 949, "listing_price": 867, "gmp": -20, "qib": 2.8, "nii": 2.9, "rii": 1.9, "sector": "Insurance", "pe_ratio": 100.0, "promoter_holding": 96.5, "lot_size": 15, "issue_size_cr": 21000},
            {"name": "Delhivery", "year": 2022, "issue_price": 487, "listing_price": 493, "gmp": 5, "qib": 2.6, "nii": 0.3, "rii": 0.5, "sector": "Logistics", "pe_ratio": -60.0, "promoter_holding": 0.0, "lot_size": 30, "issue_size_cr": 5235},
            {"name": "Swiggy", "year": 2024, "issue_price": 390, "listing_price": 420, "gmp": 12, "qib": 6.0, "nii": 1.9, "rii": 1.1, "sector": "E-commerce", "pe_ratio": -85.0, "promoter_holding": 0.0, "lot_size": 38, "issue_size_cr": 11327},
            {"name": "Hyundai Motor India", "year": 2024, "issue_price": 1960, "listing_price": 1934, "gmp": -25, "qib": 6.9, "nii": 0.6, "rii": 0.5, "sector": "Auto", "pe_ratio": 26.0, "promoter_holding": 82.5, "lot_size": 7, "issue_size_cr": 27870},
            {"name": "Aadhar Housing Finance", "year": 2024, "issue_price": 315, "listing_price": 315, "gmp": 45, "qib": 72.7, "nii": 16.5, "rii": 2.4, "sector": "Finance", "pe_ratio": 22.0, "promoter_holding": 76.4, "lot_size": 47, "issue_size_cr": 3000},
            {"name": "Indegene", "year": 2024, "issue_price": 452, "listing_price": 659, "gmp": 260, "qib": 192.7, "nii": 55.0, "rii": 7.9, "sector": "Healthcare", "pe_ratio": 38.5, "promoter_holding": 20.8, "lot_size": 33, "issue_size_cr": 1842},
            {"name": "TBO Tek", "year": 2024, "issue_price": 920, "listing_price": 1426, "gmp": 540, "qib": 125.5, "nii": 50.6, "rii": 14.4, "sector": "Tech", "pe_ratio": 64.0, "promoter_holding": 51.2, "lot_size": 16, "issue_size_cr": 1551},
            {"name": "Go Digit", "year": 2024, "issue_price": 272, "listing_price": 281, "gmp": 10, "qib": 12.5, "nii": 7.2, "rii": 4.2, "sector": "Insurance", "pe_ratio": -45.0, "promoter_holding": 83.3, "lot_size": 55, "issue_size_cr": 2615},
            {"name": "Awfis Space Solutions", "year": 2024, "issue_price": 383, "listing_price": 432, "gmp": 125, "qib": 116.9, "nii": 129.2, "rii": 53.2, "sector": "Real Estate", "pe_ratio": -35.0, "promoter_holding": 26.6, "lot_size": 39, "issue_size_cr": 599},
            {"name": "Ixigo", "year": 2024, "issue_price": 93, "listing_price": 135, "gmp": 30, "qib": 106.7, "nii": 110.2, "rii": 54.9, "sector": "Travel", "pe_ratio": 80.0, "promoter_holding": 0.0, "lot_size": 161, "issue_size_cr": 740},
            {"name": "Allied Blenders", "year": 2024, "issue_price": 281, "listing_price": 318, "gmp": 50, "qib": 86.9, "nii": 32.4, "rii": 4.5, "sector": "FMCG", "pe_ratio": 50.5, "promoter_holding": 82.2, "lot_size": 53, "issue_size_cr": 1500},
            {"name": "Emcure Pharma", "year": 2024, "issue_price": 1008, "listing_price": 1325, "gmp": 350, "qib": 195.8, "nii": 48.3, "rii": 7.2, "sector": "Pharma", "pe_ratio": 36.5, "promoter_holding": 78.0, "lot_size": 14, "issue_size_cr": 1952},
            {"name": "Bansal Wire", "year": 2024, "issue_price": 256, "listing_price": 356, "gmp": 75, "qib": 146.0, "nii": 51.4, "rii": 13.6, "sector": "Manufacturing", "pe_ratio": 35.0, "promoter_holding": 95.7, "lot_size": 58, "issue_size_cr": 745},
            {"name": "Brainbees (FirstCry)", "year": 2024, "issue_price": 465, "listing_price": 625, "gmp": 80, "qib": 19.3, "nii": 4.6, "rii": 2.3, "sector": "E-commerce", "pe_ratio": -40.0, "promoter_holding": 46.0, "lot_size": 32, "issue_size_cr": 4193},
            {"name": "Saraswati Saree", "year": 2024, "issue_price": 160, "listing_price": 194, "gmp": 45, "qib": 64.1, "nii": 358.4, "rii": 61.5, "sector": "Retail", "pe_ratio": 28.5, "promoter_holding": 100.0, "lot_size": 90, "issue_size_cr": 160},
            {"name": "Premier Energies", "year": 2024, "issue_price": 450, "listing_price": 990, "gmp": 390, "qib": 216.6, "nii": 50.0, "rii": 7.6, "sector": "Energy", "pe_ratio": 85.0, "promoter_holding": 72.0, "lot_size": 33, "issue_size_cr": 2830},
            {"name": "Ecos Mobility", "year": 2024, "issue_price": 334, "listing_price": 390, "gmp": 160, "qib": 136.8, "nii": 71.2, "rii": 19.7, "sector": "Transport", "pe_ratio": 35.8, "promoter_holding": 100.0, "lot_size": 44, "issue_size_cr": 601},
            {"name": "Baazar Style", "year": 2024, "issue_price": 389, "listing_price": 389, "gmp": 65, "qib": 81.8, "nii": 59.4, "rii": 9.1, "sector": "Retail", "pe_ratio": 132.0, "promoter_holding": 59.4, "lot_size": 38, "issue_size_cr": 835},
            {"name": "Gala Precision", "year": 2024, "issue_price": 529, "listing_price": 750, "gmp": 250, "qib": 232.5, "nii": 414.6, "rii": 91.9, "sector": "Manufacturing", "pe_ratio": 42.0, "promoter_holding": 74.5, "lot_size": 28, "issue_size_cr": 168},
            {"name": "Tolins Tyres", "year": 2024, "issue_price": 226, "listing_price": 227, "gmp": 30, "qib": 25.4, "nii": 27.4, "rii": 21.5, "sector": "Auto", "pe_ratio": 30.0, "promoter_holding": 83.3, "lot_size": 66, "issue_size_cr": 230},
            {"name": "Kross", "year": 2024, "issue_price": 240, "listing_price": 240, "gmp": 20, "qib": 23.3, "nii": 22.2, "rii": 10.7, "sector": "Auto", "pe_ratio": 29.5, "promoter_holding": 98.7, "lot_size": 62, "issue_size_cr": 500},
            {"name": "PNB Housing Finance", "year": 2021, "issue_price": 425, "listing_price": 400, "gmp": 0, "qib": 1.2, "nii": 1.5, "rii": 0.8, "sector": "Finance", "pe_ratio": 15.0, "promoter_holding": 32.5, "lot_size": 35, "issue_size_cr": 3000},
            {"name": "Gland Pharma", "year": 2020, "issue_price": 1500, "listing_price": 1701, "gmp": 100, "qib": 6.4, "nii": 0.5, "rii": 0.2, "sector": "Pharma", "pe_ratio": 20.5, "promoter_holding": 58.0, "lot_size": 10, "issue_size_cr": 6480},
            {"name": "CAMS", "year": 2020, "issue_price": 1230, "listing_price": 1518, "gmp": 350, "qib": 73.1, "nii": 111.8, "rii": 5.5, "sector": "Finance", "pe_ratio": 35.0, "promoter_holding": 43.5, "lot_size": 12, "issue_size_cr": 2242},
            {"name": "Burger King India", "year": 2020, "issue_price": 60, "listing_price": 115, "gmp": 45, "qib": 86.6, "nii": 354.1, "rii": 68.1, "sector": "Food", "pe_ratio": -25.0, "promoter_holding": 52.8, "lot_size": 250, "issue_size_cr": 810},
            {"name": "Indigo Paints", "year": 2021, "issue_price": 1490, "listing_price": 2607, "gmp": 850, "qib": 189.5, "nii": 263.0, "rii": 15.9, "sector": "Paints", "pe_ratio": 135.0, "promoter_holding": 54.0, "lot_size": 10, "issue_size_cr": 1170},
            {"name": "MTAR Tech", "year": 2021, "issue_price": 575, "listing_price": 1063, "gmp": 500, "qib": 164.9, "nii": 650.7, "rii": 28.4, "sector": "Manufacturing", "pe_ratio": 47.0, "promoter_holding": 50.8, "lot_size": 26, "issue_size_cr": 596},
            {"name": "Nazara Tech", "year": 2021, "issue_price": 1101, "listing_price": 1971, "gmp": 800, "qib": 103.7, "nii": 389.8, "rii": 75.2, "sector": "Tech", "pe_ratio": 120.0, "promoter_holding": 20.5, "lot_size": 13, "issue_size_cr": 583},
        ]

        # Raw upcoming and active IPOs
        self._raw_upcoming = [
            {
                "name": "National Stock Exchange",
                "sector": "Finance",
                "status": "Draft Filed",
                "price_band_low": 3200,
                "price_band_high": 3500,
                "issue_price": 3500,
                "listing_price": None,
                "lot_size": 4,
                "issue_size_cr": 10000,
                "gmp": 1200,
                "expected_date": "Q1 2026",
                "qib": None,
                "nii": None,
                "rii": None,
                "pe_ratio": 28.5,
                "promoter_holding": 0.0,
            },
            {
                "name": "Haldiram Snacks",
                "sector": "FMCG",
                "status": "Upcoming",
                "price_band_low": 420,
                "price_band_high": 450,
                "issue_price": 450,
                "listing_price": None,
                "lot_size": 33,
                "issue_size_cr": 8000,
                "gmp": 65,
                "expected_date": "Q2 2026",
                "qib": None,
                "nii": None,
                "rii": None,
                "pe_ratio": 42.0,
                "promoter_holding": 85.0,
            },
            {
                "name": "Boat Lifestyle (Imagine Marketing)",
                "sector": "Consumer Tech",
                "status": "Draft Filed",
                "price_band_low": 180,
                "price_band_high": 200,
                "issue_price": 200,
                "listing_price": None,
                "lot_size": 75,
                "issue_size_cr": 2000,
                "gmp": 45,
                "expected_date": "Q1 2026",
                "qib": None,
                "nii": None,
                "rii": None,
                "pe_ratio": -30.0,
                "promoter_holding": 42.0,
            },
            {
                "name": "Hero Fincorp",
                "sector": "Finance",
                "status": "Upcoming",
                "price_band_low": 850,
                "price_band_high": 900,
                "issue_price": 900,
                "listing_price": None,
                "lot_size": 16,
                "issue_size_cr": 3668,
                "gmp": 110,
                "expected_date": "Q1 2026",
                "qib": None,
                "nii": None,
                "rii": None,
                "pe_ratio": 21.0,
                "promoter_holding": 79.2,
            },
            {
                "name": "Ather Energy",
                "sector": "Auto",
                "status": "Draft Filed",
                "price_band_low": 310,
                "price_band_high": 340,
                "issue_price": 340,
                "listing_price": None,
                "lot_size": 44,
                "issue_size_cr": 4500,
                "gmp": 55,
                "expected_date": "Q2 2026",
                "qib": None,
                "nii": None,
                "rii": None,
                "pe_ratio": -25.0,
                "promoter_holding": 38.0,
            }
        ]

    def _normalize_ipo(self, item: Dict[str, Any], is_history: bool = False) -> Dict[str, Any]:
        """
        Produce ONE unified shape for all IPO records.
        Every key is ALWAYS present; not applicable fields are None.
        """
        issue_price = item.get("issue_price")
        listing_price = item.get("listing_price")
        gmp = item.get("gmp")

        # Calculate listing gain % or estimated GMP gain %
        gain_pct = None
        if listing_price is not None and issue_price and issue_price > 0:
            gain_pct = round(((listing_price - issue_price) / issue_price) * 100, 2)
        elif gmp is not None and issue_price and issue_price > 0:
            gain_pct = round((gmp / issue_price) * 100, 2)

        # Compute recommendation if not explicitly provided
        recommendation = item.get("recommendation")
        if not recommendation:
            recommendation = self._compute_recommendation(item, is_history)

        status = item.get("status")
        if not status:
            status = "Listed" if is_history else "Upcoming"

        expected_date = item.get("expected_date")
        if not expected_date and is_history:
            expected_date = f"Listed ({item.get('year', 2024)})"

        return {
            "name": item.get("name"),
            "sector": item.get("sector"),
            "status": status,
            "recommendation": recommendation if recommendation else "Pending",
            "price_band_low": item.get("price_band_low"),
            "price_band_high": item.get("price_band_high"),
            "issue_price": issue_price,
            "listing_price": listing_price,
            "lot_size": item.get("lot_size"),
            "issue_size_cr": item.get("issue_size_cr"),
            "gmp": gmp,
            "expected_date": expected_date,
            "qib": item.get("qib"),
            "nii": item.get("nii"),
            "rii": item.get("rii"),
            "gain_pct": gain_pct,
            # Additional contextual fundamentals
            "pe_ratio": item.get("pe_ratio"),
            "promoter_holding": item.get("promoter_holding"),
            "year": item.get("year"),
        }

    def _compute_recommendation(self, item: Dict[str, Any], is_history: bool) -> str:
        """Deterministic scoring for recommendation."""
        # For upcoming without subscription data yet:
        qib = item.get("qib")
        issue_price = item.get("issue_price") or 1
        gmp = item.get("gmp") or 0
        gmp_pct = (gmp / issue_price) * 100 if issue_price > 0 else 0

        if qib is None:
            # Upcoming evaluation based on GMP & valuation
            if gmp_pct > 25:
                return "SUBSCRIBE"
            elif gmp_pct > 10:
                return "NEUTRAL"
            elif gmp_pct < 0:
                return "AVOID"
            return "PENDING"

        # Subscription-based evaluation
        nii = item.get("nii") or 1.0
        rii = item.get("rii") or 1.0
        total_sub = (qib + nii + rii) / 3

        score = 0
        if total_sub > 50:
            score += 35
        elif total_sub > 10:
            score += 20
        elif total_sub > 1:
            score += 10

        if gmp_pct > 30:
            score += 30
        elif gmp_pct > 10:
            score += 20
        elif gmp_pct >= 0:
            score += 10

        promoter = item.get("promoter_holding") or 50
        if promoter > 70:
            score += 20
        elif promoter > 40:
            score += 10

        if score >= 75:
            return "STRONG SUBSCRIBE"
        elif score >= 50:
            return "SUBSCRIBE"
        elif score >= 35:
            return "NEUTRAL"
        return "AVOID"

    def get_upcoming_ipos(self) -> List[Dict[str, Any]]:
        """Return list of upcoming and active IPOs in unified format."""
        return [self._normalize_ipo(ipo, is_history=False) for ipo in self._raw_upcoming]

    def get_ipo_history(self) -> List[Dict[str, Any]]:
        """Return historical IPO data in unified format."""
        return [self._normalize_ipo(ipo, is_history=True) for ipo in self._raw_history]

    def analyze_ipo(self, ipo_name: str) -> Dict[str, Any]:
        """Analyze an IPO and provide structured scoring and recommendation."""
        target_name = ipo_name.strip().lower()

        # Search history then upcoming
        all_ipos = self.get_ipo_history() + self.get_upcoming_ipos()
        matched = None
        for ipo in all_ipos:
            if ipo["name"].lower() == target_name:
                matched = ipo
                break

        if not matched:
            # Substring match fallback
            for ipo in all_ipos:
                if target_name in ipo["name"].lower():
                    matched = ipo
                    break

        if not matched:
            return {"error": f"IPO '{ipo_name}' not found in database."}

        # Calculate score (0-100)
        score = 0
        
        # 1. Subscription Demand (30%)
        qib = matched["qib"] or 1.0
        nii = matched["nii"] or 1.0
        rii = matched["rii"] or 1.0
        total_sub = (qib + nii + rii) / 3 if matched["qib"] is not None else 10.0
        
        if total_sub > 100:
            score += 30
        elif total_sub > 50:
            score += 25
        elif total_sub > 10:
            score += 15
        elif total_sub > 1:
            score += 5

        # 2. GMP Trend (25%)
        issue_price = matched["issue_price"] or 100
        gmp = matched["gmp"] or 0
        gmp_pct = (gmp / issue_price) * 100 if issue_price > 0 else 0
        if gmp_pct > 50:
            score += 25
        elif gmp_pct > 20:
            score += 20
        elif gmp_pct > 5:
            score += 12
        elif gmp_pct < 0:
            score += 0

        # 3. Valuation & Sector Median (25%)
        sector_pe = self.sector_medians.get(matched["sector"], 30.0)
        pe_ratio = matched.get("pe_ratio")
        if pe_ratio is None or pe_ratio <= 0:
            score += 8  # Loss-making or growth tech
        elif pe_ratio < sector_pe * 0.7:
            score += 25  # Undervalued
        elif pe_ratio < sector_pe * 1.2:
            score += 18  # Fairly valued
        else:
            score += 8   # Expensive

        # 4. Promoter Holding (20%)
        promoter = matched.get("promoter_holding") or 50.0
        if promoter > 75:
            score += 20
        elif promoter > 50:
            score += 15
        elif promoter > 25:
            score += 10
        else:
            score += 5

        # Recommendation
        if score >= 80:
            rec = "STRONG SUBSCRIBE"
        elif score >= 60:
            rec = "SUBSCRIBE"
        elif score >= 40:
            rec = "NEUTRAL"
        else:
            rec = "AVOID"

        return {
            "ipo": matched,
            "analysis": {
                "score": score,
                "recommendation": rec,
                "gmp_premium_pct": round(gmp_pct, 2),
                "sector_pe": sector_pe,
                "total_subscription_avg": round(total_sub, 2),
                "drivers": [
                    f"GMP Premium currently at +{round(gmp_pct, 1)}% over issue price.",
                    f"Promoter holding stands at {promoter}%.",
                    f"Sector average P/E is {sector_pe}x (Company: {pe_ratio if pe_ratio else 'N/A'})."
                ]
            }
        }

ipo_analyzer = IPOAnalyzer()
