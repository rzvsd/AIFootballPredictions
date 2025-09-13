defaults:
  permissions:
    workspace: read-write
    network: false
    git: ask
  env:
    - name: DRY_RUN
      value: "1"
    - name: REPORTS_DIR
      value: "reports"

agents:
  - name: phase1_data_and_odds
    description: Sample name-map and odds fetch (existing scripts only).
    permissions: { workspace: read-write, network: true, git: ask }
    env:
      - name: LEAGUES
        value: "E0,D1,F1,SP1,I1"
    tasks:
      - name: git-state
        run: |
          git status --porcelain
          git branch --show-current
      - name: name-map-sample
        run: |
          mkdir -p "%REPORTS_DIR%" || true
          python - << "PY"
          import glob, json, pandas as pd
          from pathlib import Path
          files = glob.glob('data/fixtures/*.csv')
          out = {}
          for f in files[:10]:
              try:
                  df = pd.read_csv(f)
                  home = df.columns[df.columns.str.lower().str.contains('home')]
                  away = df.columns[df.columns.str.lower().str.contains('away')]
                  out[Path(f).name] = {
                    "home_sample": df[home[0]].dropna().astype(str).unique().tolist()[:10] if len(home) else [],
                    "away_sample": df[away[0]].dropna().astype(str).unique().tolist()[:10] if len(away) else [],
                  }
              except Exception as e:
                  out[Path(f).name] = {"error": str(e)}
          Path("reports/name_samples.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
          print("report: reports/name_samples.json")
          PY
      - name: odds-fetch
        run: |
          python scripts/fetch_odds_api_football.py --leagues "%LEAGUES%" --out-dir data/odds || true

  - name: phase2_training_and_calibration
    description: Train XGB and compute simple metrics using existing utilities.
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: LEAGUE
        value: "E0"
    tasks:
      - name: train-xgb
        run: |
          python xgb_trainer.py --league %LEAGUE%
      - name: metrics
        run: |
          python scripts/metrics_report.py --league %LEAGUE% --out "%REPORTS_DIR%/metrics_%LEAGUE%.json" || true

  - name: phase3_monitoring_dashboard
    description: Streamlit dashboard (bankroll, P/L, bets, upcoming).
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: DASH_PORT
        value: "8501"
    tasks:
      - name: dev-server
        run: |
          streamlit run dashboard/app.py --server.port %DASH_PORT%

  - name: phase4_weekly_one_click
    description: Build snapshot, ensure models, and compute top picks.
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: LEAGUE
        value: "E0"
    tasks:
      - name: weekly-picks
        run: |
          python one_click_predictor.py --league %LEAGUE% --fixtures-csv data/fixtures/%LEAGUE%_manual.csv

