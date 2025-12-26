
import pandas as pd
import sys
import numpy as np
from pathlib import Path
sys.path.append(".")
from cgm.predict_upcoming import _rolling_stats, _add_franken_features
from cgm.pressure_inputs import ensure_pressure_inputs
from cgm.pressure_form import add_pressure_form_features
from cgm.xg_form import add_xg_form_features

def run():
    path = "data/enhanced/cgm_match_history_with_elo_stats_xg.csv"
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    subset = df[df["datetime"] < "2025-09-13"].copy()
    print(f"Subset shape: {subset.shape}")
    
    print("Running _rolling_stats...")
    try:
        subset = _rolling_stats(subset, [5, 10])
        print("Done _rolling_stats")
    except Exception:
        print("Crashed in _rolling_stats")
        import traceback
        traceback.print_exc()
        return

    print("Running _add_franken_features...")
    try:
        subset = _add_franken_features(subset, [5, 10])
        print("Done _add_franken_features")
    except Exception:
        print("Crashed in _add_franken_features")
        import traceback
        traceback.print_exc()
        return

    print("Running ensure_pressure_inputs...")
    subset = ensure_pressure_inputs(subset)
    
    print("Running add_pressure_form_features...")
    try:
        subset = add_pressure_form_features(subset, window=10)
        print("Done add_pressure_form_features")
    except Exception:
        print("Crashed in add_pressure_form_features")
        import traceback
        traceback.print_exc()
        return

    print("Running add_xg_form_features...")
    try:
        subset = add_xg_form_features(subset, window=10)
        print("Success final!")
    except Exception:
        print("Crashed in add_xg_form_features")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
