# CI: Train NGBoost Poisson Models

This workflow trains NGBoost Poisson models for home/away goals per league on a Linux runner and uploads the `.pkl` models as artifacts.

## Run the workflow
- Go to GitHub → Actions → "Train NGBoost Poisson" → "Run workflow".
- Input leagues (space-separated), e.g.: `E0 D1 F1 I1 SP1`.
- The job installs deps (`requirements.txt`, `ngboost`) and runs:
  - `python -m scripts.ngboost_trainer --league <LEAGUE>` for each league.
- Artifacts: `ngboost-models` containing
  - `advanced_models/{LEAGUE}_ngb_poisson_home.pkl`
  - `advanced_models/{LEAGUE}_ngb_poisson_away.pkl`

## Use the models locally
- Download artifacts and place files under `advanced_models/`.
- Enable NGBoost in engine via config feature flag if desired:
  ```yaml
  prob_model:
    enabled: true
    kind: ngboost_poisson
  ```
- You can benchmark via:
  ```bash
  python -m scripts.metrics_report --league E0 --start 2024-08-01 --end 2025-06-30 \
    --prob-model ngb --dist poisson --out-json reports/metrics/E0_ngb_pois.json
  ```

## Notes
- The trainer first attempts the requested NGBoost configuration; if it fails, it falls back to a robust setup so models are produced reliably.
- To limit scope, set `NGB_SUBSET_START` env var for last-season training if needed.
