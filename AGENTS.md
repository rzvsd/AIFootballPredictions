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
  - name: phase1_reliability_features
    description: Canonical Team IDs, Absences MVP (o sursă), Odds movement MVP. Produce rapoarte în reports/.
    permissions: { workspace: read-write, network: true, git: ask }
    env:
      - name: LEAGUES
        value: "E0,D1,F1,SP1,I1"
      - name: ABSENCE_SOURCE
        value: "transfermarkt"
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
                  out[Path(f).name] = {
                    "home_sample": df["home_team"].dropna().unique().tolist()[:10],
                    "away_sample": df["away_team"].dropna().unique().tolist()[:10],
                  }
              except Exception as e:
                  out[Path(f).name] = {"error": str(e)}
          Path("reports/phase1_name_samples.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
          print("report: reports/phase1_name_samples.json")
          PY
      - name: absences-fetch-mvp
        run: |
          python scripts/absences/fetch_transfermarkt.py --leagues "%LEAGUES%" --out "data/absences/latest.json"
      - name: odds-movement-mvp
        run: |
          python scripts/odds/ingest_open_close.py --leagues "%LEAGUES%" --out "data/odds/open_close.csv"

  - name: phase2_uncertainty_calibration
    description: Overdispersion (NegBin/variance-scaling), TG calibration (isotonic), CRPS metric & raport.
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: YEARS
        value: "2021,2022,2023,2024,2025"
    tasks:
      - name: overdispersion
        run: |
          mkdir -p "%REPORTS_DIR%" || true
          python scripts/uncertainty/overdispersion_proto.py --years "%YEARS%" --out "reports/overdispersion.json"
      - name: tg-calibration
        run: |
          python scripts/uncertainty/tg_calibrate.py --preds data/preds/latest.csv --out "reports/tg_calib.json"
      - name: crps
        run: |
          python scripts/metrics/compute_crps.py --preds data/preds/latest.csv --truth data/results/latest.csv --out "reports/crps_report.json"

  - name: phase3_probabilistic_modeling
    description: NGBoost/TFP prototip + A/B (CRPS, LogLoss). Pregătește feature-flag pentru integrare.
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: LEAGUES
        value: "E0,D1,F1,SP1,I1"
    tasks:
      - name: train-ngboost
        run: |
          mkdir -p models || true
          python scripts/train/ngboost_proto.py --leagues "%LEAGUES%" --out models/ngb_proto.pkl
      - name: ab-compare
        run: |
          python scripts/ab_test/run_ab.py --a engine_current --b engine_ngb --leagues "%LEAGUES%" --out "reports/ab_ngb_vs_current.json"

  - name: phase4_monitoring_dashboard
    description: Streamlit dashboard (bankroll, P/L, bets, upcoming). Export PNG/CSV demo în reports/.
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: DASH_PORT
        value: "8501"
    tasks:
      - name: build-demo
        run: |
          mkdir -p "%REPORTS_DIR%" || true
          python scripts/dashboard/generate_demo.py --out "reports/dashboard_demo.png"
      - name: dev-server
        run: |
          streamlit run scripts/dashboard/app.py --server.port %DASH_PORT%

  - name: phase5_optimization_automation
    description: Portfolio risk tuning, absences multi-source, automatizări nightly/weekly/monthly.
    permissions: { workspace: read-write, network: true, git: ask }
    env:
      - name: SCHEDULE_DIR
        value: ".github/workflows"
    tasks:
      - name: risk-tuning
        run: |
          python scripts/risk/tune_portfolio.py --in reports/backtest_summary.json --out "reports/risk_tuning.json"
      - name: absences-merge
        run: |
          python scripts/absences/merge_sources.py --inputs data/absences/* --out data/absences/merged.json
      - name: workflows
        run: |
          mkdir -p "%SCHEDULE_DIR%" || true
          python scripts/automation/gen_schedules.py --out ".github/workflows/pipeline.yml"
