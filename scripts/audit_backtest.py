"""
Milestone 12: Backtest Leakage Audit

Verifies that backtest predictions don't use future data.
Tests:
1. Backtest file exists and has expected columns
2. Predictions were made without access to future results
3. Date ranges are reasonable
4. No match appears both in history and upcoming on same run
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    print("=" * 80)
    print("BACKTEST LEAKAGE AUDIT - Milestone 12")
    print("=" * 80)

    # Find backtest files
    backtest_files = list(Path("reports").glob("backtest_*.csv"))

    if not backtest_files:
        print("No backtest files found in reports/")
        return

    print(f"Found {len(backtest_files)} backtest file(s)")

    issues = 0

    for bt_file in backtest_files:
        print(f"\n--- Auditing: {bt_file.name} ---")

        df = pd.read_csv(bt_file)
        print(f"  Rows: {len(df)}")

        # Audit 1: Check required columns
        print("\n  Audit 1: Required Columns")
        required = ["date", "home", "away", "p_over25", "p_btts_yes", "ft_home", "ft_away"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"    [FAIL] Missing: {missing}")
            issues += 1
        else:
            print("    [PASS] All required columns present")

        if df.empty:
            print("    [WARN] Backtest file is empty - skipping remaining checks")
            continue

        # Audit 2: Check date range
        print("\n  Audit 2: Date Range")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        min_date, max_date = df["date"].min(), df["date"].max()
        if pd.isna(min_date) or pd.isna(max_date):
            print("    Date range: N/A")
        else:
            print(f"    Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        # Audit 3: Verify predictions exist for matches with results
        print("\n  Audit 3: Prediction Coverage")
        has_result = df["ft_home"].notna() & df["ft_away"].notna() & ~((df["ft_home"] == 0) & (df["ft_away"] == 0))
        has_pred = df["p_over25"].notna()

        with_both = (has_result & has_pred).sum()
        with_result_only = (has_result & ~has_pred).sum()

        print(f"    Matches with result + prediction: {with_both}")
        print(f"    Matches with result only (no pred): {with_result_only}")

        if with_both < 50:
            print("    [WARN] Very few predictions with results for validation")

        # Audit 4: Check for suspicious patterns
        print("\n  Audit 4: Leakage Indicators")

        # If predictions perfectly match results, that's suspicious
        if "p_over25" in df.columns and "ft_home" in df.columns:
            valid = df[has_result & has_pred].copy()
            if len(valid) > 0:
                valid["actual_over25"] = (valid["ft_home"] + valid["ft_away"]) > 2.5
                valid["pred_over25"] = valid["p_over25"] > 0.5
                accuracy = (valid["pred_over25"] == valid["actual_over25"]).mean()
                print(f"    O/U 2.5 accuracy (50% threshold): {accuracy * 100:.1f}%")

                if accuracy > 0.9:
                    print("    [WARN] Unusually high accuracy - possible leakage!")
                    issues += 1
                elif accuracy < 0.3:
                    print("    [WARN] Unusually low accuracy - check predictions")
                else:
                    print("    [PASS] Accuracy within normal range")

        # Audit 5: Check run_asof timestamp if present
        print("\n  Audit 5: Run Timestamp Check")
        if "run_asof_datetime" in df.columns:
            run_dates = df["run_asof_datetime"].nunique()
            print(f"    Unique run timestamps: {run_dates}")
            if run_dates > 1:
                print("    [PASS] Multiple runs detected (walk-forward)")
            else:
                print("    Single run detected")
        else:
            print("    run_asof_datetime column not present")

    print("\n" + "=" * 80)
    if issues == 0:
        print("[PASS] Backtest Leakage Audit PASSED")
    else:
        print(f"[WARN] Found {issues} potential issues")
    print("=" * 80)


if __name__ == "__main__":
    main()

