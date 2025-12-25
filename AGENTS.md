defaults:
  permissions:
    workspace: read-write
    network: false
    git: ask
  env:
    - name: REPORTS_DIR
      value: "reports"

agents:
  - name: cgm_only_pipeline
    description: CGM-only pipeline (history -> Elo -> features -> train -> upcoming predictions).
    permissions: { workspace: read-write, network: false, git: ask }
    env:
      - name: MAX_DATE
        value: "YYYY-MM-DD"
    tasks:
      - name: git-state
        run: |
          git status --porcelain
          git branch --show-current
      - name: full-refresh
        run: |
          python predict.py --max-date %MAX_DATE%
      - name: predict-only
        run: |
          python predict.py --predict-only
