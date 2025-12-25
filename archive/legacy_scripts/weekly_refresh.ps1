Param(
  [string]$League = "E0",
  [string]$Seasons = "2425 2324",
  [string]$UnderstatSeasons = "2025,2024",
  [string]$AbsencesCsv = "",
  [switch]$TryInstallUnderstat,
  [switch]$SkipDownload
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..  # move to repo root

# 0) Ensure directories
New-Item -ItemType Directory -Force -Path data/raw | Out-Null
New-Item -ItemType Directory -Force -Path data/processed | Out-Null
New-Item -ItemType Directory -Force -Path data/enhanced | Out-Null
New-Item -ItemType Directory -Force -Path data/absences | Out-Null

if (-not $SkipDownload) {
  # 1) Download raw league CSVs (includes HC/AC; possession if present)
  Write-Step "Downloading raw CSVs from football-data.co.uk ($($League): $($Seasons))"
  $seasonList = $Seasons -split '\s+'
  python scripts/data_acquisition.py --leagues $League --seasons @($seasonList) --raw_data_output_dir data/raw
}

# 2) Preprocess to create processed CSV (timeline for EWMAs)
Write-Step "Preprocessing raw data to processed CSV"
python scripts/data_preprocessing.py --raw_data_input_dir data/raw --processed_data_output_dir data/processed --num_features 30 --clustering_threshold 0.5

# 3) Fetch Understat shots and ingest
Write-Step "Fetching Understat shots ($UnderstatSeasons) and building shots CSV"
if ($TryInstallUnderstat) {
  Write-Step "Ensuring Python package 'understat' is available"
  & python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('understat') else 1)" | Out-Null
  if ($LASTEXITCODE -ne 0) {
    try { python -m pip install --user understat } catch { Write-Host "[warn] pip install understat failed; continuing with existing shots if present" -ForegroundColor Yellow }
  }
}
try {
  python -m scripts.fetch_understat_simple --league $League --seasons $UnderstatSeasons
  if ($LASTEXITCODE -ne 0) { throw "fetch_understat_simple failed with exit code $LASTEXITCODE" }
}
catch {
  Write-Host "[warn] Understat fetch failed; proceeding with existing shots" -ForegroundColor Yellow
}
python -m scripts.shots_ingest_understat --inputs data/understat/*_shots.json --out data/shots/understat_shots.csv

# 4) Build micro aggregates (injects possession/corners if available in processed/raw)
Write-Step "Building micro aggregates (possession, corners, xG/poss, xG from corners)"
python -m scripts.build_micro_aggregates --shots data/shots/understat_shots.csv --league $League --out data/enhanced/micro_agg.csv

# 5) Ensure absences snapshot (use provided CSV or seed defaults)
$absPath = "data/absences/${League}_availability.csv"
if ($AbsencesCsv -and (Test-Path $AbsencesCsv)) {
  Write-Step "Importing absences from $AbsencesCsv"
  python -m scripts.absences_import --league $League --input $AbsencesCsv
}
elseif (-not (Test-Path $absPath)) {
  Write-Step "Creating default absences snapshot (availability_index=1.0)"
  $proc = "data/processed/${League}_merged_preprocessed.csv"
  if (-not (Test-Path $proc)) { throw "Processed file not found: $proc" }
  $teams = @{}
  Import-Csv $proc | ForEach-Object {
    if ($_.HomeTeam) { $teams[$_.HomeTeam] = $true }
    if ($_.AwayTeam) { $teams[$_.AwayTeam] = $true }
  }
  $out = @()
  foreach ($k in $teams.Keys) { $out += [PSCustomObject]@{ team=$k; availability_index=1.0 } }
  $seed = "data/absences/${League}_availability_seed.csv"
  $out | Export-Csv -NoTypeInformation -Path $seed
  python -m scripts.absences_import --league $League --input $seed
}

# 6) Train XGB models
Write-Step "Training XGB models for $League"
python xgb_trainer.py --league $League

# 7) Refresh Understat fixtures for this league (next 14 days)
Write-Step "Fetching Understat fixtures for $League"
python -m scripts.fetch_fixtures_understat --leagues $League --days 14

# 8) Fetch Bet365 odds from football-data (best-effort)
Write-Step "Fetching football-data Bet365 odds for $League"
python -m scripts.fetch_odds_fd_simple --leagues $League --days 14

# 9) Generate picks
Write-Step "Generating market book and top picks for $League"
python bet_fusion.py --top 20

Write-Step "Weekly refresh complete."
