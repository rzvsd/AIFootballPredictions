"""
Train probability calibration models (Isotonic Regression) for CGM.
Reads history, generates out-of-sample predictions, and trains calibrators 
to map raw model probabilities to true observed frequencies.
"""
import argparse
import joblib
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibration")

def load_data(history_path: Path):
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} not found")
    df = pd.read_csv(history_path)
    
    # Map frankenstein columns to expected history columns
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "time" in df.columns:
        # Only use time if datetime is missing (legacy)
        df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        
    if "y_home" in df.columns:
        df["ft_home"] = df["y_home"]
    if "y_away" in df.columns:
        df["ft_away"] = df["y_away"]
        
    return df.sort_values("datetime")

def train_calibration(history_path: str, models_dir: str, out_dir: str):
    # Prefer frankenstein_training_full.csv if passed, else fallback
    input_path = Path(history_path)
    
    # Logic to find the full feature file (best for calibration)
    if "frankenstein" not in str(input_path).lower():
        frank_path = input_path.parent / "frankenstein_training_full.csv"
        if frank_path.exists():
            logger.info(f"Switched to FULL feature file: {frank_path}")
            input_path = frank_path
        else:
             # Try standard frankenstein
            frank_path = input_path.parent / "frankenstein_training.csv"
            if frank_path.exists():
                logger.info(f"Switched to feature file: {frank_path}")
                input_path = frank_path
            
    logger.info(f"Loading data from {input_path}")
    df = load_data(input_path)
    
    # Load models
    model_h_path = Path(models_dir) / "frankenstein_mu_home.pkl"
    model_a_path = Path(models_dir) / "frankenstein_mu_away.pkl"
    
    if not model_h_path.exists() or not model_a_path.exists():
        raise FileNotFoundError("Frankenstein models not found. Train them first.")
        
    mh = joblib.load(model_h_path)
    ma = joblib.load(model_a_path)
    
    # Calibration set: 2024-2025 matches
    # Ensure datetime is valid
    if "datetime" not in df.columns:
         raise ValueError(f"Missing datetime column in {input_path}")

    calib_set = df[df["datetime"].dt.year >= 2024].copy()
    if len(calib_set) < 100:
        logger.warning(f"Calibration set small ({len(calib_set)} rows). Expanding to 2023...")
        calib_set = df[df["datetime"].dt.year >= 2023].copy()
        
    logger.info(f"Calibration set: {len(calib_set)} matches")
    
    # Generate raw predictions
    feat_cols_h = mh.feature_names_in_ if hasattr(mh, "feature_names_in_") else mh.get_booster().feature_names
    feat_cols_a = ma.feature_names_in_ if hasattr(ma, "feature_names_in_") else ma.get_booster().feature_names
    
    # Filter valid rows (targets must exist)
    calib_set = calib_set.dropna(subset=["ft_home", "ft_away"]).copy()
    
    # We DO NOT drop missing features because XGBoost handles NaNs natively.
    # The production model predicts on incomplete data, so calibration should too.
    valid_h = calib_set.copy()
    valid_a = calib_set.copy()
    
    # Predict Goal Expectancy (mu)
    valid_h["pred_mu_home"] = mh.predict(valid_h[feat_cols_h])
    valid_a["pred_mu_away"] = ma.predict(valid_a[feat_cols_a])
    
    # Combine predictions (inner join on index)
    preds = valid_h[["pred_mu_home", "ft_home", "ft_away", "datetime"]].join(
        valid_a[["pred_mu_away"]], how="inner"
    )
    
    if len(preds) == 0:
        raise ValueError("No valid rows for calibration after feature dropna")

    # Arrays for calibration
    raw_probs = []
    
    for _, row in preds.iterrows():
        mu_h, mu_a = row["pred_mu_home"], row["pred_mu_away"]
        
        # Poisson PMFs
        h_prob = [poisson.pmf(i, mu_h) for i in range(10)]
        a_prob = [poisson.pmf(i, mu_a) for i in range(10)]
        
        # 1. Home Win
        p_h = sum(h_prob[i] * sum(a_prob[j] for j in range(i)) for i in range(10))
        # 2. Away Win
        p_a = sum(a_prob[j] * sum(h_prob[i] for i in range(j)) for j in range(10))
        # 3. Over 2.5
        p_under = sum(h_prob[i] * a_prob[j] for i in range(10) for j in range(10) if i+j <= 2)
        p_over = 1.0 - p_under
        # 4. BTTS
        p_btts = sum(h_prob[i] * a_prob[j] for i in range(1, 10) for j in range(1, 10))
        
        raw_probs.append({
            "p_home_raw": p_h,
            "p_away_raw": p_a,
            "p_over_raw": p_over,
            "p_btts_raw": p_btts,
            "actual_home": row["ft_home"] > row["ft_away"],
            "actual_away": row["ft_home"] < row["ft_away"],
            "actual_over": (row["ft_home"] + row["ft_away"]) > 2.5,
            "actual_btts": (row["ft_home"] > 0) & (row["ft_away"] > 0)
        })
        
    calib_df = pd.DataFrame(raw_probs)
    
    # Train Isotonic Regressors
    iso_home = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_away = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_over = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_btts = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    
    logger.info("Training Isotonic Regression models...")
    iso_home.fit(calib_df["p_home_raw"], calib_df["actual_home"].astype(int))
    iso_away.fit(calib_df["p_away_raw"], calib_df["actual_away"].astype(int))
    iso_over.fit(calib_df["p_over_raw"], calib_df["actual_over"].astype(int))
    iso_btts.fit(calib_df["p_btts_raw"], calib_df["actual_btts"].astype(int))
    
    # Save models
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    joblib.dump(iso_home, Path(out_dir) / "calib_home.pkl")
    joblib.dump(iso_away, Path(out_dir) / "calib_away.pkl")
    joblib.dump(iso_over, Path(out_dir) / "calib_over.pkl")
    joblib.dump(iso_btts, Path(out_dir) / "calib_btts.pkl")
    
    logger.info(f"Saved 4 calibration models to {out_dir}")
    
    # Quick Check
    calib_df["p_over_calib"] = iso_over.predict(calib_df["p_over_raw"])
    mse_raw = ((calib_df["p_over_raw"] - calib_df["actual_over"].astype(int))**2).mean()
    mse_calib = ((calib_df["p_over_calib"] - calib_df["actual_over"].astype(int))**2).mean()
    
    logger.info(f"Over 2.5 MSE: Raw={mse_raw:.4f} -> Calib={mse_calib:.4f}")
    if mse_raw > 0:
        logger.info(f"Improvement: {(mse_raw-mse_calib)/mse_raw*100:.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="data/enhanced/cgm_match_history_with_elo_stats_xg.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="models/calibration")
    args = ap.parse_args()
    
    train_calibration(args.history, args.models_dir, args.out_dir)
