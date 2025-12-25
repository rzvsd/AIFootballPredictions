Initialization / Quick Reference

Core commands (run from repo root)
- One-command pipeline: `python predict.py --max-date YYYY-MM-DD`
- Predict-only (requires existing artifacts + models): `python predict.py --predict-only`
- Recompute Elo (clean, cutoff to avoid future leakage): `python -m scripts.calc_cgm_elo --max-date YYYY-MM-DD` (defaults to today UTC). Output: `data/enhanced/cgm_match_history_with_elo.csv`.
- Build features: `python -m cgm.build_frankenstein --data-dir data/enhanced --match-history cgm_match_history_with_elo_stats.csv`.
- Train models: `python -m cgm.train_frankenstein_mu`.
- Predict upcoming: `python -m cgm.predict_upcoming`.

Main engine files
- `scripts/calc_cgm_elo.py`: Elo from CGM results only (start 1500, K=20, HFA 65, margin multiplier, date cutoff).
- `cgm/build_match_history.py`: normalize/merge CGM tables, drop future rows, prep base history.
- `cgm/build_frankenstein.py`: rolling stats/features for training (uses clean Elo).
- `cgm/train_frankenstein_mu.py`: trains μ_home/μ_away models.
- `cgm/predict_upcoming.py`: inference bridge (builds features, predicts μ, Poisson probs, EV).

Legacy NGBoost pipeline archived under `archive/legacy_workflows` as of 2025-12-25.
