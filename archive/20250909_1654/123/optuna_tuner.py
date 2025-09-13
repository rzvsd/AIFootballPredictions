# optuna_tuner.py
# Hyperparameter search for bet_fusion weights using a walk-forward backtest.
# Objective: maximize weighted hit-rate proxy (until bookmaker odds are integrated).

import os, argparse, json
from datetime import datetime
import numpy as np
import pandas as pd
import optuna
import subprocess
import tempfile
import sys

# You can import and call bet_fusion as a module if you prefer.
# Here we shell out to your existing script to keep it decoupled and identical to prod run.

DEFAULT_MARKETS = "1X2,DC,OU25,BTTS"

def run_fusion(league, dfrom, dto, weights, min_prob, min_conf, markets):
    # Write a small JSON file the fusion script can optionally read for WEIGHTS override
    wd = tempfile.mkdtemp()
    wpath = os.path.join(wd, "weights.json")
    with open(wpath,"w") as f: json.dump(weights, f)

    # Call bet_fusion.py with env override
    env = os.environ.copy()
    env["FUSION_WEIGHTS_JSON"] = wpath

    cmd = [
        sys.executable, "bet_fusion.py",
        "--league", league, "--from", dfrom, "--to", dto, "--source", "local",
        "--min-prob", str(min_prob), "--min-conf", str(min_conf),
        "--markets", markets, "--top-k", "9999"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # fusion writes CSV "fusion_picks_<league>_<from>_to_<to>.csv"
    out_csv = f"fusion_picks_{league}_{dfrom}_to_{dto}.csv"
    if not os.path.exists(out_csv):
        # if no picks, return empty
        return pd.DataFrame(columns=["league","kickoff_utc","home","away","market","side","model_prob","conf"])
    return pd.read_csv(out_csv)

def score_week(picks_csv, results_csv):
    # use your compare script logic inline (minimal)
    picks=pd.read_csv(picks_csv)
    res=pd.read_csv(results_csv)
    picks["date_key"]=pd.to_datetime(picks.get("kickoff_utc", picks.get("date"))).dt.date
    res["date_key"]=pd.to_datetime(res["date"]).dt.date
    j = picks.merge(res, left_on=["date_key","home","away"], right_on=["date_key","home_team","away_team"], how="inner")
    if j.empty: return 0.0, 0
    # Basic evaluator: count correct by market types you care about
    def one_x_two(hg,ag):
        return "HOME" if hg>ag else ("AWAY" if ag>hg else "DRAW")
    hits=0; total=0
    for _,r in j.iterrows():
        m=r["market"]; side=str(r["side"])
        hg=int(r["home_goals"]); ag=int(r["away_goals"]); tot=hg+ag
        ok=False
        if m=="1X2":
            if side==r["home"]: ok = hg>ag
            elif side=="Draw":  ok = hg==ag
            elif side==r["away"]: ok = ag>hg
        elif m=="DC":
            if side=="1X": ok = not (ag>hg)
            elif side=="12": ok = hg!=ag
            elif side=="X2": ok = not (hg>ag)
        elif m=="OU2.5":
            ok = (tot>=3) if side=="Over" else (tot<=2)
        elif m=="BTTS":
            ok = (hg>=1 and ag>=1) if side=="Yes" else not (hg>=1 and ag>=1)
        total+=1; hits+=1 if ok else 0
    return (hits/total if total else 0.0), total

def objective(trial, league, weeks, results_dir, markets):
    # Search space
    w_dc  = trial.suggest_float("w_dc_matrix", 0.0, 1.0)
    w_model_prob = trial.suggest_float("w_model_prob", 0.3, 0.9)
    w_signal_bias= 1.0 - w_model_prob
    form_raw = trial.suggest_float("form_w", 0.0, 1.0)
    tempo_raw= trial.suggest_float("tempo_w", 0.0, 1.0)
    elo_raw  = trial.suggest_float("elo_w", 0.0, 1.0)
    s = form_raw + tempo_raw + elo_raw + 1e-9
    form = form_raw/s; tempo=tempo_raw/s; elo=elo_raw/s
    slope_strength = trial.suggest_float("slope_strength", 0.5, 5.0)
    slope_tempo    = trial.suggest_float("slope_tempo", 0.5, 5.0)
    slope_elo      = trial.suggest_float("slope_elo", 0.5, 5.0)

    # thresholds (global for now)
    min_prob = trial.suggest_float("min_prob", 0.50, 0.65)
    min_conf = trial.suggest_float("min_conf", 0.55, 0.75)

    weights = {
        "w_dc_matrix": w_dc,
        "w_model_prob": w_model_prob,
        "w_signal_bias": w_signal_bias,
        "form_w": form,
        "tempo_w": tempo,
        "elo_w": elo,
        "slope_strength": slope_strength,
        "slope_tempo": slope_tempo,
        "slope_elo": slope_elo,
    }

    # Walk-forward across provided week windows
    scores=[]; picks_count=0
    for (dfrom,dto) in weeks:
        # caller ensures fixtures & results csv exist for each week
        _picks = run_fusion(league, dfrom, dto, weights, min_prob, min_conf, markets)
        p_csv = f"fusion_picks_{league}_{dfrom}_to_{dto}.csv"
        r_csv = os.path.join(results_dir, f"{league}_{dfrom}_to_{dto}_results.csv")
        if not os.path.exists(r_csv): continue
        acc, n = score_week(p_csv, r_csv)
        picks_count += n
        scores.append(acc)

        # Prune early if we’re doing terribly
        trial.report(np.mean(scores) if scores else 0.0, step=len(scores))
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    if not scores: return 0.0
    # Objective: average accuracy; small penalty if we pick too few (encourage coverage)
    avg_acc = float(np.mean(scores))
    coverage_bonus = np.tanh(picks_count/100.0) * 0.02  # tiny nudge
    return avg_acc + coverage_bonus

def main():
    ap=argparse.ArgumentParser(description="Tune fusion weights via Optuna.")
    ap.add_argument("--league", required=True)
    ap.add_argument("--weeks-file", required=True, help="CSV with columns: from,to (YYYY-MM-DD)")
    ap.add_argument("--results-dir", required=True, help="Folder with week result files named {league}_{from}_to_{to}_results.csv")
    ap.add_argument("--markets", default=DEFAULT_MARKETS)
    ap.add_argument("--trials", type=int, default=200)
    args=ap.parse_args()

    weeks = pd.read_csv(args.weeks_file)[["from","to"]].values.tolist()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, args.league, weeks, args.results_dir, args.markets), n_trials=args.trials, gc_after_trial=True)

    print("\nBest value:", study.best_value)
    print("Best params:", study.best_params)

    with open("optuna_best_params.json","w") as f: json.dump(study.best_params, f, indent=2)
    print("Saved → optuna_best_params.json")

if __name__=="__main__":
    main()
