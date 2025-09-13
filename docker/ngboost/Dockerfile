FROM python:3.11-slim

WORKDIR /app

# System deps (optional but useful for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# Install a minimal set of deps needed for NGBoost training
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        numpy==1.23.5 \
        pandas==2.3.1 \
        scikit-learn==1.3.2 \
        scipy==1.10.1 \
        xgboost==2.0.3 \
        "statsmodels>=0.14.0,<0.15" \
        patsy>=0.5.6 \
        joblib>=1.3.0 \
        PyYAML>=6.0 \
        "pyarrow>=14.0.0,<19.0" \
        ngboost \
        requests \
        shap

# Copy repo
COPY . /app

# Default env: leagues can be overridden at runtime
ENV LEAGUES="E0"
ENV TASK="ngboost"
ENV UNDERSTAT_LEAGUE="EPL"
ENV UNDERSTAT_SEASONS="2024 2023"
ENV UNDERSTAT_LIMIT="100"

# Entrypoint orchestrator (uses TASK env to choose path)
CMD bash -lc 'set -e; if [ "$TASK" = "xg_pipeline" ]; then echo "[CMD] Running xG shot pipeline"; python scripts/fetch_understat_simple.py --league "$UNDERSTAT_LEAGUE" --seasons $UNDERSTAT_SEASONS --out-dir data/understat --limit "$UNDERSTAT_LIMIT" || true; python -m scripts.shots_ingest_understat --inputs data/understat/*.json --out data/shots/understat_shots.csv; python -m xg_shot_model.train_logistic --shots data/shots/understat_shots.csv --out models/xg_logistic.pkl --calib models/xg_logistic_calib.pkl; python -m xg_shot_model.train_gbm --shots data/shots/understat_shots.csv --out models/xg_xgb.pkl --report reports/xg_shot_gbm_metrics.json; python -m scripts.build_micro_aggregates --shots data/shots/understat_shots.csv --out data/enhanced/micro_agg.csv; echo "[CMD] xG shot pipeline complete."; else echo "[CMD] Training NGBoost for leagues: $LEAGUES"; for lg in $LEAGUES; do echo "==> $lg"; python -m scripts.ngboost_trainer --league "$lg"; done; echo "Done. Models in advanced_models/."; fi'

