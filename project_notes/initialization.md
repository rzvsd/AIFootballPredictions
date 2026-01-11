Initialization / Quick Reference

Core commands (run from repo root)
- One-command pipeline: `python predict.py --max-date YYYY-MM-DD`
- Predict-only (requires existing artifacts + models): `python predict.py --predict-only`
- Recompute Elo (clean, cutoff to avoid future leakage): `python -m scripts.calc_cgm_elo --max-date YYYY-MM-DD` (defaults to today UTC). Output: `data/enhanced/cgm_match_history_with_elo.csv`.
- Build features: `python -m cgm.build_frankenstein --data-dir data/enhanced --match-history cgm_match_history_with_elo_stats.csv`.
- Train models: `python -m cgm.train_frankenstein_mu`.
- Predict upcoming: `python -m cgm.predict_upcoming`.

---

## CGM Data Management

### Required Files (4 only)
```
CGM data/
├── multiple seasons.csv   ← Match history (outcomes, odds, Elo)
├── goals statistics.csv   ← Per-match stats (shots, corners)
├── multiple leagues and seasons/allratingv.csv    ← Today's fixtures
└── AGS.CSV               ← Goal timing data
```

### Updating Data (New Downloads)
1. Download new CSVs from CGM (any filename works!)
2. Put them in `CGM data/` folder
3. Run merge script to combine and standardize:
   ```bash
   python scripts/merge_cgm_data.py
   ```
4. Rebuild the pipeline:
   ```bash
   python predict.py --rebuild-history
   ```

The merge script auto-detects file types and saves to standard names.

---

Main engine files
- `scripts/calc_cgm_elo.py`: Elo from CGM results only (start 1500, K=20, HFA 65, margin multiplier, date cutoff).
- `cgm/build_match_history.py`: normalize/merge CGM tables, drop future rows, prep base history.
- `cgm/build_frankenstein.py`: rolling stats/features for training (uses clean Elo).
- `cgm/train_frankenstein_mu.py`: trains μ_home/μ_away models.
- `cgm/predict_upcoming.py`: inference bridge (builds features, predicts μ, Poisson probs, EV).

Legacy NGBoost pipeline archived under `archive/legacy_workflows` as of 2025-12-25.

---

## Audit Scripts

Run all audits (10 scripts, validates all features):
```bash
python scripts/run_all_audits.py
```

Individual audits:
```bash
# Multi-league coverage verification
python scripts/audit_multi_league.py

# Picks engine validation
(legacy) python archive/legacy_full_engine/audit_picks.py
python -m scripts.audit_picks_goals

# Feature validation
python -m scripts.audit_pressure --cutoff YYYY-MM-DD
python -m scripts.audit_xg
python -m scripts.audit_decay
python -m scripts.audit_h2h
python -m scripts.audit_league_features

# Feed and output validation
python -m scripts.audit_upcoming_feed --as-of-date YYYY-MM-DD
python -m scripts.audit_narrator
```

## Prediction Report

Generate formatted predictions table:
```bash
python scripts/generate_predictions_report.py
python scripts/generate_predictions_report.py --date YYYY-MM-DD
python scripts/generate_predictions_report.py --league "Premier L"
```
