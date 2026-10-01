import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from joblib import dump, load
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from backend.feature_pipeline import FEATURE_COLS, compute_features

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "saved_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

XGB_BUNDLE_PATH = os.path.join(WEIGHTS_DIR, "xgb_bundle.joblib")


class XGBoostPredictor:
    def __init__(self):
        self.models: Dict[int, XGBRegressor] = {}
        self.scaler: Optional[StandardScaler] = None
        self.is_loaded = False
        self._load_bundle()

    def _load_bundle(self) -> bool:
        if os.path.exists(XGB_BUNDLE_PATH):
            try:
                bundle = load(XGB_BUNDLE_PATH)
                self.models = bundle.get("models", {})
                self.scaler = bundle.get("scaler")
                self.is_loaded = True
                print("Loaded XGBoost bundle successfully.")
                return True
            except Exception as e:
                print(f"Failed to load XGBoost bundle: {e}")
        return False

    def train_bundle(self, training_df: pd.DataFrame) -> bool:
        """Train 7-day XGBoost regressors on combined historical feature data."""
        try:
            print("Training XGBoost 7-day models...")
            X = training_df[FEATURE_COLS].values
            y = training_df[[f'target_day_{i}' for i in range(1, 8)]].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            best_params = {
                'n_estimators': 120,
                'max_depth': 3,
                'learning_rate': 0.075,
                'subsample': 0.90,
                'colsample_bytree': 0.91,
                'random_state': 42,
                'n_jobs': -1
            }

            models = {}
            for day in range(1, 8):
                model = XGBRegressor(**best_params)
                model.fit(X_scaled, y[:, day - 1])
                models[day] = model

            bundle = {
                "models": models,
                "scaler": scaler,
                "feature_cols": FEATURE_COLS
            }
            dump(bundle, XGB_BUNDLE_PATH)
            self.models = models
            self.scaler = scaler
            self.is_loaded = True
            print("Saved XGBoost models bundle.")
            return True
        except Exception as e:
            print(f"XGBoost training error: {e}")
            return False

    def predict(self, stock_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate 7-day forward predictions using XGBoost."""
        if stock_df.empty or len(stock_df) < 15:
            return {"error": "Insufficient historical data for prediction"}

        feat_df = compute_features(stock_df, include_targets=False)
        if feat_df.empty:
            return {"error": "Feature engineering failed"}

        latest_row = feat_df.iloc[[-1]]
        latest_features = latest_row[FEATURE_COLS].values
        current_price = float(latest_row['Close'].iloc[0])
        rolling_mean = float(latest_row['rolling_mean'].iloc[0])

        # If models are not loaded, train on the fly using this stock's history
        if not self.is_loaded or self.scaler is None:
            train_features = compute_features(stock_df, include_targets=True)
            if not train_features.empty and len(train_features) > 100:
                self.train_bundle(train_features)

        # Scale input
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(latest_features)
            except Exception:
                # If feature dimension mismatch, fit temporary
                temp_scaler = StandardScaler()
                X_scaled = temp_scaler.fit_transform(latest_features)
        else:
            X_scaled = latest_features

        forecasts: List[Dict[str, Any]] = []
        cumulative_uncertainty = 0.015

        for day in range(1, 8):
            if day in self.models:
                pred_norm = float(self.models[day].predict(X_scaled)[0])
            else:
                # Fallback heuristic if day model missing
                pred_norm = float(latest_row['Close_norm'].iloc[0]) * (1.0 + (day * 0.001))

            # Denormalize to actual currency price
            predicted_price = (pred_norm * rolling_mean) / 100.0

            # Guard against negative or unrealistic prices
            predicted_price = max(predicted_price, current_price * 0.4)
            pct_change = ((predicted_price - current_price) / current_price) * 100.0

            # Dynamic confidence bands (widens as days advance)
            spread = predicted_price * (cumulative_uncertainty * np.sqrt(day))
            lower_bound = max(predicted_price - spread, 0.01)
            upper_bound = predicted_price + spread

            forecasts.append({
                "day": day,
                "day_label": f"+{day}D",
                "predicted_price": round(predicted_price, 2),
                "predicted_norm": round(pred_norm, 2),
                "change_pct": round(pct_change, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
            })

        return {
            "model_name": "XGBoost Regressor",
            "current_price": round(current_price, 2),
            "rolling_mean": round(rolling_mean, 2),
            "forecasts": forecasts,
            "overall_7d_change": round(forecasts[-1]["change_pct"], 2),
            "expected_target_price": forecasts[-1]["predicted_price"],
        }


# Global singleton instance
xgb_engine = XGBoostPredictor()
