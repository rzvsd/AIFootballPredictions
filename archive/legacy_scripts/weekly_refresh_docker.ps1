Param(
  [string]$League = "E0",
  [string]$Seasons = "2425 2324",
  [string]$UnderstatSeasons = "2025,2024",
  [string]$FixturesCsv = "data/fixtures/${League}_manual.csv",
  [string]$AbsencesCsv = "",
  [switch]$Build,
  [switch]$TryInstallUnderstat
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..  # repo root

if ($Build) {
  Write-Step "Building Docker image aifootball:latest"
  docker build -t aifootball:latest -f docker/ngboost/Dockerfile .
}

$cwd = (Get-Location).Path
$cmdParts = @()
$cmdParts += "set -e"
# download and preprocess
$cmdParts += "python scripts/data_acquisition.py --leagues $League --seasons $Seasons --raw_data_output_dir data/raw || true"
$cmdParts += "python scripts/data_preprocessing.py --raw_data_input_dir data/raw --processed_data_output_dir data/processed --num_features 30 --clustering_threshold 0.5"
# Understat fetch + ingest
if ($TryInstallUnderstat) { $cmdParts += "python -m pip install --user understat || true" }
$cmdParts += "python -m scripts.fetch_understat_simple --league $League --seasons $UnderstatSeasons || true"
$cmdParts += "python -m scripts.shots_ingest_understat --inputs data/understat/*_shots.json --out data/shots/understat_shots.csv"
# micro aggregates
$cmdParts += "python -m scripts.build_micro_aggregates --shots data/shots/understat_shots.csv --league $League --out data/enhanced/micro_agg.csv"
# absences (optional)
if ($AbsencesCsv -and (Test-Path $AbsencesCsv)) { $cmdParts += "python -m scripts.absences_import --league $League --input $AbsencesCsv" }
# train + picks
$cmdParts += "python xgb_trainer.py --league $League"
$cmdParts += "python scripts/betting_bot.py --league $League"

$cmd = ($cmdParts | Where-Object { $_ -and $_.Trim().Length -gt 0 }) -join ' && '

Write-Step "Running weekly pipeline inside Docker"
docker run --rm -v "${cwd}:/app" aifootball:latest bash -lc "$cmd"

Write-Step "Docker weekly refresh complete."
