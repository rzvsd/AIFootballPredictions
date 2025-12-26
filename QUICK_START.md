# ⚽ Football Predictions Bot — Quick Start Guide

Welcome! This bot analyzes football matches and tells you which bets have good value.

---

## 🎯 What Does This Bot Do?

In simple terms:
1. **Looks at history** — How did teams perform in past matches?
2. **Calculates strength** — Uses Elo ratings, shot stats, goals data
3. **Predicts goals** — "This team will score ~1.5 goals on average"
4. **Compares to odds** — "The bookmaker thinks 30%, but we think 45%"
5. **Picks the best bets** — Only outputs bets where you have an edge

---

## 📁 What You Need

Put these files in the `CGM data/` folder:

| File | What It Contains |
|------|-----------------|
| `multiple seasons.csv` | Historical match results |
| `upcoming - Copy.CSV` | Future matches with odds |
| `goals statistics.csv` | Shot/corner/possession stats |
| `AGS.CSV` | Goal timing data (when goals were scored) |

💡 **Important:** The `upcoming - Copy.CSV` file must contain FUTURE matches. If it only has past matches, you'll get 0 predictions.

---

## 🚀 How to Run

### First Time (Full Setup)
```bash
python predict.py --rebuild-history
```
This builds everything from scratch. Takes ~1 minute.

### Daily Use (Quick Run)
```bash
python predict.py --predict-only
```
Uses existing models, just runs predictions. Takes ~10 seconds.

### If You Updated Your Data Files
```bash
python predict.py
```
The bot detects new data and rebuilds automatically.

---

## 📊 What You Get

After running, check the `reports/` folder:

| File | What's Inside |
|------|--------------|
| `picks.csv` | **Your betting picks** with odds, probabilities, stakes |
| `picks_explained_preview.txt` | **Human-readable** explanations of each pick |
| `picks_debug.csv` | Why certain bets were rejected (too risky, low edge) |
| `pipeline_summary.json` | Did anything fail during the run? |

---

## 📖 Reading Your Picks

Here's what the columns mean:

| Column | Meaning | Example |
|--------|---------|---------|
| `market` | What bet type | `OU25_OVER` = Over 2.5 goals |
| `odds` | Bookmaker odds | `2.10` |
| `p_model` | Our probability | `0.55` = 55% |
| `p_implied` | Bookmaker's probability | `0.476` = 47.6% |
| `ev` | Expected Value (edge) | `0.10` = 10% edge |
| `stake_tier` | How much to bet | `T1`=0.5u, `T2`=1u, `T3`=1.5u, `T4`=2u |

### Bet Types Explained

| Code | English |
|------|---------|
| `1X2_HOME` | Home team wins |
| `1X2_AWAY` | Away team wins |
| `OU25_OVER` | More than 2.5 goals total |
| `OU25_UNDER` | Less than 2.5 goals total |
| `BTTS_YES` | Both teams score |
| `BTTS_NO` | At least one team doesn't score |

---

## ⚠️ Troubleshooting

### "0 picks"
- Your `upcoming - Copy.CSV` might be outdated
- Export fresh data from CGM

### "[UNSEEN] team not in history"
- This team is new (promoted, etc.)
- The bot skips it because it doesn't have enough data

### "picks.csv is empty"
- No bets passed the quality filters
- This is NORMAL — better to skip than take bad bets

---

## 🔧 Settings You Can Change

Edit `config.py` to adjust:

| Setting | What It Does |
|---------|-------------|
| `EV_MIN_OU25` | Minimum edge required (default 4%) |
| `NEFF_MIN_*` | How much history required |
| `ODDS_MIN_*` | Minimum odds to consider |

---

## 📞 Need Help?

Check these files for more details:
- `project_notes/blueprint.md` — Technical architecture
- `project_notes/bugs_fixed.md` — Known issues that were fixed
- `reports/run_log.jsonl` — Detailed run logs

---

**Happy Betting!** 🎰⚽
