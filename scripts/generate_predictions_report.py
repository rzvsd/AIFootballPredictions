#!/usr/bin/env python3
"""
Generate Calibrated Predictions Report
=======================================
Produces a nicely formatted predictions table with:
- Match name
- O/U 2.5 probabilities
- BTTS probabilities
- Best Value pick with EV%

Output: reports/predictions_report.txt (console) and reports/predictions_report.md (markdown)

Usage:
    python scripts/generate_predictions_report.py
    python scripts/generate_predictions_report.py --league "Premier L"
    python scripts/generate_predictions_report.py --date 2025-12-28
"""

import argparse
from pathlib import Path
import pandas as pd


def get_best_value(row):
    """Determine the best value bet from EV columns."""
    evs = {
        'Over 2.5': row.get('EV_over25', 0) or 0,
        'Under 2.5': row.get('EV_under25', 0) or 0,
        'BTTS Yes': row.get('EV_btts_yes', 0) or 0,
        'BTTS No': row.get('EV_btts_no', 0) or 0,
    }
    
    # Find best EV
    best_pick = max(evs, key=evs.get)
    best_ev = evs[best_pick]
    
    if best_ev <= 0:
        return "-", 0
    
    # Add emoji based on EV strength
    if best_ev >= 0.20:
        emoji = " 🔥"  # Hot pick
    elif best_ev >= 0.10:
        emoji = " ✅"  # Good value
    elif best_ev >= 0:
        emoji = ""
    else:
        emoji = " ❌"  # Negative EV
    
    return f"{best_pick} (+{best_ev*100:.0f}%){emoji}", best_ev


def format_ou(row):
    """Format Over/Under 2.5 probabilities."""
    p_over = row.get('p_over25', 0.5)
    p_over = p_over * 100 if p_over < 1 else p_over
    p_under = 100 - p_over
    return f"O:{p_over:.0f}% / U:{p_under:.0f}%"


def format_btts(row):
    """Format BTTS probabilities."""
    p_btts = row.get('p_btts_yes', 0.5)
    p_btts = p_btts * 100 if p_btts and p_btts < 1 else (p_btts if p_btts else 50)
    return f"Y:{p_btts:.0f}% / N:{100-p_btts:.0f}%"


def generate_report(df, output_path=None, league_filter=None, date_filter=None):
    """Generate the predictions report."""
    
    # Apply filters
    if league_filter:
        df = df[df['league'].str.contains(league_filter, case=False, na=False)]
    if date_filter:
        df = df[df['date'] == date_filter]
    
    if df.empty:
        print("No predictions match the filters.")
        return
    
    lines = []
    md_lines = []
    
    # Header
    lines.append("=" * 100)
    lines.append("📊 CALIBRATED PREDICTIONS REPORT")
    lines.append("=" * 100)
    lines.append("")
    
    md_lines.append("# 📊 Calibrated Predictions Report\n")
    md_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # Group by date
    for date in sorted(df['date'].unique()):
        date_df = df[df['date'] == date].sort_values(['league', 'home'])
        
        lines.append(f"📅 {date}")
        lines.append("-" * 100)
        lines.append(f"{'Match':<35} {'O/U 2.5':<16} {'BTTS':<16} {'Best Value':<20}")
        lines.append("-" * 100)
        
        md_lines.append(f"\n## 📅 {date}\n")
        md_lines.append("| Match | O/U 2.5 | BTTS | Best Value |")
        md_lines.append("|-------|---------|------|------------|")
        
        for _, row in date_df.iterrows():
            match = f"{row['home']} vs {row['away']}"
            match_short = match[:33] if len(match) > 33 else match
            
            p_ou = format_ou(row)
            p_btts = format_btts(row)
            best_val, ev = get_best_value(row)
            
            lines.append(f"{match_short:<35} {p_ou:<16} {p_btts:<16} {best_val:<20}")
            md_lines.append(f"| {match_short} | {p_ou} | {p_btts} | {best_val} |")
        
        lines.append("")
    
    # Summary
    lines.append("=" * 100)
    lines.append(f"TOTAL: {len(df)} predictions | {df['league'].nunique()} leagues | {df['date'].nunique()} dates")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Legend:")
    lines.append("  🔥 = EV >= 20% (Hot pick)")
    lines.append("  ✅ = EV >= 10% (Good value)")
    lines.append("  ❌ = Negative EV (Avoid)")
    
    md_lines.append(f"\n---\n**TOTAL:** {len(df)} predictions | {df['league'].nunique()} leagues | {df['date'].nunique()} dates\n")
    md_lines.append("\n**Legend:** 🔥 EV≥20% | ✅ EV≥10% | ❌ Negative EV")
    
    # Print to console
    report = "\n".join(lines)
    print(report)
    
    # Save to files
    if output_path:
        txt_path = Path(output_path).with_suffix('.txt')
        md_path = Path(output_path).with_suffix('.md')
        
        txt_path.write_text(report, encoding='utf-8')
        md_path.write_text("\n".join(md_lines), encoding='utf-8')
        
        print(f"\n✅ Saved: {txt_path}")
        print(f"✅ Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Calibrated Predictions Report")
    parser.add_argument("--input", default="reports/cgm_upcoming_predictions.csv", help="Input predictions CSV")
    parser.add_argument("--output", default="reports/predictions_report", help="Output path (without extension)")
    parser.add_argument("--league", default=None, help="Filter by league (e.g., 'Premier L')")
    parser.add_argument("--date", default=None, help="Filter by date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # Load predictions
    df = pd.read_csv(args.input)
    df['date'] = pd.to_datetime(df['fixture_datetime']).dt.strftime('%Y-%m-%d')
    
    generate_report(df, args.output, args.league, args.date)


if __name__ == "__main__":
    main()
