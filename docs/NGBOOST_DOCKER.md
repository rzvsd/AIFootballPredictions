# Train NGBoost in Docker (Linux environment)

If GitHub Actions cannot run (billing/spending limits), you can train NGBoost Poisson models locally in a Linux container. This avoids Windows-specific numerical issues and produces the same artifacts as CI.

## Build the image
```bash
# from repo root
docker build -t aifootball-ngb -f docker/ngboost/Dockerfile .
```

## Run training for multiple leagues
```bash
# Mount the repo so models are saved to your working tree
# PowerShell (Windows):
docker run --rm -it -v ${PWD}:/app -e LEAGUES="E0 D1 F1 I1 SP1" aifootball-ngb

# Bash (Linux/macOS):
# docker run --rm -it -v "$(pwd)":/app -e LEAGUES="E0 D1 F1 I1 SP1" aifootball-ngb
```

Outputs:
- advanced_models/{LEAGUE}_ngb_poisson_home.pkl
- advanced_models/{LEAGUE}_ngb_poisson_away.pkl

## Benchmark locally
```bash
python -m scripts.metrics_report --league E0 --start 2024-08-01 --end 2025-06-30 \
  --prob-model ngb --dist poisson --out-json reports/metrics/E0_ngb_pois.json
```

Tip: You can adjust NGBoost config in `scripts/ngboost_trainer.py`. The trainer first tries the requested config, then falls back to a robust setting if needed.

