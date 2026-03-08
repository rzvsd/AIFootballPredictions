param(
    [string]$MaxDate = (Get-Date).ToString("yyyy-MM-dd"),
    [string]$LeagueIds = "39,40,79,140,78,135,136,61,62,88,94,203,283",
    [int]$HistoryDays = 365,
    [int]$HorizonDays = 14,
    [int]$MaxRequests = 7500,
    [int]$RatePerMinute = 120,
    [int]$NextRoundSpanDays = 3,
    [switch]$AllUpcoming,
    [switch]$DryRunTelegram,
    [switch]$NoFetchOdds,
    [switch]$SkipLatam
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Run-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Args
    )

    Write-Host ""
    Write-Host "== $Name =="
    Write-Host ("python " + ($Args -join " "))
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

Write-Host "Full pipeline + Telegram start"
Write-Host "MaxDate=$MaxDate LeagueIds=$LeagueIds HistoryDays=$HistoryDays HorizonDays=$HorizonDays MaxRequests=$MaxRequests RatePerMinute=$RatePerMinute"

# 1) API sync (history + upcoming + optional odds)
$syncArgs = @(
    "scripts/sync_api_football.py",
    "--data-dir", "data/api_football",
    "--league-ids", "$LeagueIds",
    "--history-days", "$HistoryDays",
    "--horizon-days", "$HorizonDays",
    "--max-requests", "$MaxRequests",
    "--rate-per-minute", "$RatePerMinute"
)
if ($NoFetchOdds) {
    $syncArgs += "--no-fetch-odds"
} else {
    $syncArgs += "--fetch-odds"
}
Run-PythonStep -Name "Sync API-Football data" -Args $syncArgs

# 2) Rebuild enhanced history stack
Run-PythonStep -Name "Build match history" -Args @(
    "-m", "cgm.build_match_history",
    "--data-dir", "data/api_football",
    "--out", "data/enhanced/cgm_match_history.csv",
    "--max-date", "$MaxDate"
)
Run-PythonStep -Name "Build baselines" -Args @(
    "-m", "cgm.build_baselines",
    "--data-dir", "data/api_football",
    "--match-history", "data/enhanced/cgm_match_history.csv",
    "--out-team-baselines", "data/enhanced/team_baselines.csv"
)
Run-PythonStep -Name "Calculate Elo" -Args @(
    "-m", "scripts.calc_cgm_elo",
    "--history", "data/enhanced/cgm_match_history.csv",
    "--out", "data/enhanced/cgm_match_history_with_elo.csv",
    "--data-dir", "data/api_football",
    "--max-date", "$MaxDate",
    "--log-json", "reports/run_log.jsonl"
)
Run-PythonStep -Name "Backfill stats" -Args @(
    "-m", "cgm.backfill_match_stats",
    "--history", "data/enhanced/cgm_match_history_with_elo.csv",
    "--stats", "data/api_football/history_fixtures.csv",
    "--out", "data/enhanced/cgm_match_history_with_elo_stats.csv",
    "--data-dir", "data/api_football"
)
Run-PythonStep -Name "Build xG proxy" -Args @(
    "-m", "cgm.build_xg_proxy",
    "--history", "data/enhanced/cgm_match_history_with_elo_stats.csv",
    "--out", "data/enhanced/cgm_match_history_with_elo_stats_xg.csv"
)

# 3) Strict module engine has no separate training step.

# 4) Predict
$scopeArgs = @("--next-round-only")
if ($AllUpcoming) {
    $scopeArgs = @("--all-upcoming")
}

$predictArgs = @(
    "-m", "cgm.predict_upcoming",
    "--history", "data/enhanced/cgm_match_history_with_elo_stats_xg.csv",
    "--models-dir", "models",
    "--model-variant", "full",
    "--out", "reports/cgm_upcoming_predictions.csv",
    "--data-dir", "data/api_football",
    "--as-of-date", "$MaxDate",
    "--horizon-days", "$HorizonDays",
    "--next-round-span-days", "$NextRoundSpanDays",
    "--log-json", "reports/run_log.jsonl",
    "--trace-json", "reports/elo_trace.jsonl"
)
$predictArgs += $scopeArgs
Run-PythonStep -Name "Generate predictions" -Args $predictArgs

# 5) Telegram send
$telegramArgs = @("scripts/send_telegram_predictions.py")
if ($DryRunTelegram) {
    $telegramArgs += "--dry-run"
}
Run-PythonStep -Name "Send Telegram messages" -Args $telegramArgs

# 6) Optional LATAM isolated pipeline + Telegram
if (-not $SkipLatam) {
    Write-Host ""
    Write-Host "== Run isolated LATAM pipeline (Argentina + Brazil) =="
    $latamArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts/run_latam_pipeline.ps1",
        "-MaxDate", "$MaxDate",
        "-HistoryDays", "$HistoryDays",
        "-HorizonDays", "$HorizonDays",
        "-MaxRequests", "$MaxRequests",
        "-RatePerMinute", "$RatePerMinute",
        "-SkipPicks"
    )
    Write-Host ("powershell " + ($latamArgs -join " "))
    & powershell @latamArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: isolated LATAM pipeline (exit code $LASTEXITCODE)"
    }

    $telegramLatamArgs = @(
        "scripts/send_telegram_predictions.py",
        "--predictions", "reports_latam/cgm_upcoming_predictions.csv"
    )
    if ($DryRunTelegram) {
        $telegramLatamArgs += "--dry-run"
    }
    Run-PythonStep -Name "Send Telegram messages (LATAM)" -Args $telegramLatamArgs
}

Write-Host ""
Write-Host "Completed successfully."
Write-Host "Predictions file: reports/cgm_upcoming_predictions.csv"
