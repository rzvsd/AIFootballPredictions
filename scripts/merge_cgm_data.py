"""
CGM Data Merge Script

Reads all CSV files from CGM data folder and merges them into the 4 standard files
that the bot expects:
  - multiple seasons.csv (history)
  - goals statistics.csv (per-match stats) 
  - upcoming - Copy.CSV (fixtures)
  - AGS.CSV (goal timing)

Usage:
  python scripts/merge_cgm_data.py
  
The script:
1. Auto-detects file type by examining columns
2. Concatenates and deduplicates rows
3. Saves to standard filenames
4. Backs up existing files first
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("CGM data")

# Standard output filenames
OUT_HISTORY = "multiple seasons.csv"
OUT_STATS = "goals statistics.csv"
OUT_UPCOMING = "upcoming - Copy.CSV"
OUT_TIMING = "AGS.CSV"


def detect_file_type(df: pd.DataFrame, filename: str) -> str:
    """
    Detect file type by examining columns and filename.
    Returns: 'history', 'stats', 'upcoming', 'timing', or 'unknown'
    """
    cols = [c.lower() for c in df.columns]
    fname = filename.lower()
    
    # AGS/Timing: has goalmina or goalmin columns AND is named AGS
    if 'goalmina' in cols or 'goalmin' in cols or 'ags' in fname:
        # But AGS in history also has goalmina - check if it has timing-specific structure
        if 'datamecic' in cols:  # Timing file uses datamecic instead of datameci
            return 'timing'
    
    # Upcoming: has future matches indicators and specific column patterns
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
    
    # Fallback: check filename
    if 'multiple' in fname or 'season' in fname or 'rating' in fname:
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
    print("CGM DATA MERGE SCRIPT")
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
    
    # Scan all CSVs
    all_csvs = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.CSV"))
    print(f"\nFound {len(all_csvs)} CSV files in {DATA_DIR}/")
    
    for csv_path in all_csvs:
        # Skip the output files themselves to avoid circular merge
        if csv_path.name in [OUT_HISTORY, OUT_STATS, OUT_UPCOMING, OUT_TIMING]:
            print(f"  ⏭️ {csv_path.name}: Skipping (output file)")
            continue
            
        try:
            df = read_csv_safe(csv_path)
            ftype = detect_file_type(df, csv_path.name)
            print(f"  📄 {csv_path.name}: Detected as {ftype.upper()} ({len(df)} rows)")
            files[ftype].append((csv_path, df))
        except Exception as e:
            print(f"  ❌ {csv_path.name}: Error - {e}")
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = DATA_DIR / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Backing up to {backup_dir}/")
    
    for out_name in [OUT_HISTORY, OUT_STATS, OUT_UPCOMING, OUT_TIMING]:
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
        combined.to_csv(out_path, index=False, encoding='latin1')
        print(f"   ✅ Saved: {out_path}")
    
    # Process each type
    merge_and_save(files['history'], OUT_HISTORY)
    merge_and_save(files['stats'], OUT_STATS)
    merge_and_save(files['upcoming'], OUT_UPCOMING)
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
