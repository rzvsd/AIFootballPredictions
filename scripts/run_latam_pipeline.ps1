param(
    [string]$MaxDate = (Get-Date).ToString("yyyy-MM-dd"),
    [int]$HistoryDays = 365,
    [int]$HorizonDays = 7,
    [int]$SeasonHistory = 2025,
    [int]$SeasonCurrent = 2026,
    [int]$MaxRequests = 7500,
    [int]$RatePerMinute = 120,
    [switch]$SkipPicks
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Run-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    Write-Host ""
    Write-Host "== $Name =="
    Write-Host "python $($Args -join ' ')"
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

Write-Host "LATAM pipeline start (Argentina + Brazil, isolated)"
Write-Host "MaxDate=$MaxDate HistoryDays=$HistoryDays HorizonDays=$HorizonDays SeasonHistory=$SeasonHistory SeasonCurrent=$SeasonCurrent"

New-Item -ItemType Directory -Path "data/enhanced_latam" -Force | Out-Null
New-Item -ItemType Directory -Path "models_latam" -Force | Out-Null
New-Item -ItemType Directory -Path "reports_latam" -Force | Out-Null

# 1) Sync history snapshot (season 2025)
Run-PythonStep -Name "Sync LATAM season $SeasonHistory (no odds)" -Args @(
    "scripts/sync_api_football.py",
    "--data-dir", "data/api_football_latam_s2025",
    "--league-ids", "71,128",
    "--season", "$SeasonHistory",
    "--history-days", "$HistoryDays",
    "--horizon-days", "$HorizonDays",
    "--max-requests", "$MaxRequests",
    "--rate-per-minute", "$RatePerMinute",
    "--no-fetch-odds"
)

# 2) Sync current snapshot (season 2026 with odds/upcoming)
Run-PythonStep -Name "Sync LATAM season $SeasonCurrent (with odds)" -Args @(
    "scripts/sync_api_football.py",
    "--data-dir", "data/api_football_latam_s2026",
    "--league-ids", "71,128",
    "--season", "$SeasonCurrent",
    "--history-days", "$HistoryDays",
    "--horizon-days", "$HorizonDays",
    "--max-requests", "$MaxRequests",
    "--rate-per-minute", "$RatePerMinute",
    "--fetch-odds"
)

# 3) Merge snapshots into isolated LATAM dataset
Run-PythonStep -Name "Build merged LATAM dataset" -Args @(
    "scripts/build_latam_dataset.py"
)

# 4) Build enhanced history + baselines
Run-PythonStep -Name "Build match history (LATAM)" -Args @(
    "-m", "cgm.build_match_history",
    "--data-dir", "data/api_football_latam",
    "--out", "data/enhanced_latam/cgm_match_history.csv",
    "--max-date", "$MaxDate"
)
Run-PythonStep -Name "Build baselines (LATAM)" -Args @(
    "-m", "cgm.build_baselines",
    "--data-dir", "data/api_football_latam",
    "--match-history", "data/enhanced_latam/cgm_match_history.csv",
    "--out-team-baselines", "data/enhanced_latam/team_baselines.csv"
)

# 5) Elo + stats + xG proxy
Run-PythonStep -Name "Calculate Elo (LATAM)" -Args @(
    "-m", "scripts.calc_cgm_elo",
    "--history", "data/enhanced_latam/cgm_match_history.csv",
    "--out", "data/enhanced_latam/cgm_match_history_with_elo.csv",
    "--data-dir", "data/api_football_latam",
    "--max-date", "$MaxDate",
    "--log-json", "reports_latam/run_log.jsonl"
)
Run-PythonStep -Name "Backfill match stats (LATAM)" -Args @(
    "-m", "cgm.backfill_match_stats",
    "--history", "data/enhanced_latam/cgm_match_history_with_elo.csv",
    "--stats", "data/api_football_latam/multiple leagues and seasons/history_fixtures.csv",
    "--out", "data/enhanced_latam/cgm_match_history_with_elo_stats.csv",
    "--data-dir", "data/api_football_latam"
)
Run-PythonStep -Name "Build xG proxy (LATAM)" -Args @(
    "-m", "cgm.build_xg_proxy",
    "--history", "data/enhanced_latam/cgm_match_history_with_elo_stats.csv",
    "--out", "data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv"
)

# 6) Strict module engine has no separate training step.

# 7) Predict next round
Run-PythonStep -Name "Predict upcoming next round (LATAM)" -Args @(
    "-m", "cgm.predict_upcoming",
    "--history", "data/enhanced_latam/cgm_match_history_with_elo_stats_xg.csv",
    "--models-dir", "models_latam",
    "--model-variant", "full",
    "--out", "reports_latam/cgm_upcoming_predictions.csv",
    "--data-dir", "data/api_football_latam",
    "--as-of-date", "$MaxDate",
    "--next-round-only",
    "--next-round-span-days", "3",
    "--log-json", "reports_latam/run_log.jsonl",
    "--trace-json", "reports_latam/elo_trace.jsonl"
)

if (-not $SkipPicks) {
    Run-PythonStep -Name "Generate picks (LATAM)" -Args @(
        "-m", "cgm.pick_engine_goals",
        "--in", "reports_latam/cgm_upcoming_predictions.csv",
        "--out", "reports_latam/picks.csv",
        "--debug-out", "reports_latam/picks_debug.csv"
    )
}

Write-Host ""
Write-Host "LATAM pipeline completed successfully."
Write-Host "Predictions: reports_latam/cgm_upcoming_predictions.csv"
if (-not $SkipPicks) {
    Write-Host "Picks: reports_latam/picks.csv"
}

