"""
CGM Data Merge Script

Reads all CSV files from CGM data folder (including subdirectories) and merges
them into standard files the bot expects:
  - multiple seasons.csv (legacy history)
  - multiple leagues and seasons/allratingv.csv (multi-league history)
  - goals statistics.csv (per-match stats)
  - upcoming - Copy.CSV (legacy fixtures)
  - multiple leagues and seasons/upcoming.csv (multi-league fixtures)
  - AGS.CSV (goal timing)

Usage:
  python scripts/merge_cgm_data.py
  
The script:
1. Recursively scans CGM data/ and all subdirectories
2. Auto-detects file type by examining columns
3. Concatenates and deduplicates rows
4. Saves to standard filenames in CGM data/ root and multi-league subfolder
5. Backs up existing files first

Notes:
- Drop new CSV exports into any subfolder (e.g., "multiple leagues and seasons/")
- No need to rename files - just run the merge script
- Output files are written to CGM data/ root and multi-league subfolder
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("CGM data")

# Standard output filenames
OUT_HISTORY = "multiple seasons.csv"
OUT_HISTORY_MULTI = "multiple leagues and seasons/allratingv.csv"
OUT_STATS = "goals statistics.csv"
OUT_UPCOMING = "upcoming - Copy.CSV"
OUT_UPCOMING_MULTI = "multiple leagues and seasons/upcoming.csv"
OUT_TIMING = "AGS.CSV"

# Files to skip (output files and backups)
SKIP_PATTERNS = {
    OUT_HISTORY,
    Path(OUT_HISTORY_MULTI).name,
    OUT_STATS,
    OUT_UPCOMING,
    Path(OUT_UPCOMING_MULTI).name,
    OUT_TIMING,
}


def detect_file_type(df: pd.DataFrame, filename: str) -> str:
    """
    Detect file type by examining columns and filename.
    Returns: 'history', 'stats', 'upcoming', 'timing', or 'unknown'
    """
    cols = [c.lower() for c in df.columns]
    fname = filename.lower()
    
    # AGS/Timing: check filename first (most reliable), then columns
    if 'ags' in fname:
        return 'timing'
    
    # Upcoming: filename-based detection (most reliable)
    if 'upcoming' in fname:
        return 'upcoming'
    
    # Stats (goals statistics): has per-match stats like sut, sutt, cor, ballp
    # AND is relatively small (match-level stats, not full history)
    if all(c in cols for c in ['sut', 'sutt', 'cor', 'ballp']):
        if 'country' not in cols:  # Stats file doesn't have country column usually
            return 'stats'
    
    # History (multiple seasons): has country, league, full match data
    if 'country' in cols and 'league' in cols:
        if 'scor1' in cols or 'result' in cols or 'homeprob' in cols:
            return 'history'
    
    # Fallback: check filename patterns for history files
    # This catches allratingv.csv, epl1.csv, bundesliga.csv, etc.
    if any(pattern in fname for pattern in ['multiple', 'season', 'rating', 
                                             'epl', 'bundesliga', 'laliga', 'ligue',
                                             'serie', 'olanda', 'portg', 'rom', 'turk']):
        return 'history'
    if 'statistic' in fname or 'stats' in fname:
        return 'stats'
    
    return 'unknown'


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV with fallback encodings."""
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path} with any encoding")


def merge_data():
    print("="*70)
    print("CGM DATA MERGE SCRIPT (with subdirectory scanning)")
    print("="*70)
    
    if not DATA_DIR.exists():
        print(f"❌ '{DATA_DIR}' does not exist!")
        return
    
    # Collect all files by type
    files = {
        'history': [],
        'stats': [],
        'upcoming': [],
        'timing': [],
        'unknown': []
    }
    
    # Scan all CSVs recursively (including subdirectories)
    all_csvs = list(DATA_DIR.rglob("*.csv")) + list(DATA_DIR.rglob("*.CSV"))
    # Remove duplicates (case-insensitive match on Windows)
    seen = set()
    unique_csvs = []
    for p in all_csvs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique_csvs.append(p)
    all_csvs = unique_csvs
    
    print(f"\nFound {len(all_csvs)} CSV files in {DATA_DIR}/ (including subdirectories)")
    
    for csv_path in all_csvs:
        # Skip backup folder
        if 'backups' in csv_path.parts:
            continue
            
        # Skip the output files themselves to avoid circular merge
        if csv_path.name in SKIP_PATTERNS:
            print(f"  ⏭️  {csv_path.relative_to(DATA_DIR)}: Skipping (output file)")
            continue
            
        try:
            df = read_csv_safe(csv_path)
            ftype = detect_file_type(df, csv_path.name)
            rel_path = csv_path.relative_to(DATA_DIR)
            print(f"  📄 {rel_path}: Detected as {ftype.upper()} ({len(df)} rows)")
            files[ftype].append((csv_path, df))
        except Exception as e:
            print(f"  ❌ {csv_path.name}: Error - {e}")
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = DATA_DIR / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Backing up to {backup_dir}/")
    
    for out_name in [OUT_HISTORY, OUT_HISTORY_MULTI, OUT_STATS, OUT_UPCOMING, OUT_UPCOMING_MULTI, OUT_TIMING]:
        out_path = DATA_DIR / out_name
        if out_path.exists():
            shutil.copy2(out_path, backup_dir / out_name)
            print(f"   Backed up: {out_name}")
    
    # Merge each type
    def merge_and_save(file_list, out_name, dedupe_cols=None):
        if not file_list:
            print(f"\n⚠️ No files detected for {out_name}")
            return
        
        print(f"\n🔄 Merging {len(file_list)} file(s) into {out_name}...")
        
        dfs = [df for _, df in file_list]
        combined = pd.concat(dfs, ignore_index=True)
        before = len(combined)
        
        # Deduplicate
        combined = combined.drop_duplicates()
        after = len(combined)
        
        print(f"   Rows: {before} → {after} (removed {before-after} exact duplicates)")
        
        # Save
        out_path = DATA_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False, encoding='latin1')
        print(f"   ✅ Saved: {out_path}")
    
    # Process each type
    merge_and_save(files['history'], OUT_HISTORY)
    merge_and_save(files['history'], OUT_HISTORY_MULTI)
    merge_and_save(files['stats'], OUT_STATS)
    merge_and_save(files['upcoming'], OUT_UPCOMING)
    merge_and_save(files['upcoming'], OUT_UPCOMING_MULTI)
    merge_and_save(files['timing'], OUT_TIMING)
    
    # Report unknown files
    if files['unknown']:
        print("\n⚠️ Unknown files (not merged):")
        for path, _ in files['unknown']:
            print(f"   - {path.name}")
    
    print("\n" + "="*70)
    print("✅ Merge complete!")
    print("="*70)
    print("\nNext step: python predict.py --rebuild-history")


if __name__ == "__main__":
    merge_data()
