```
BUSINESS REPORT (NON-TECHNICAL)
Generated: 2026-02-18 05:29 UTC

What this report answers:
1. What happened in the last rounds
2. What future games are loaded and what the bot recommends
3. How good historical predictions were in backtest

Data source files:
- History: C:\BOTS\AIFootballPredictions\data\enhanced\cgm_match_history_with_elo_stats_xg.csv
- Upcoming predictions: C:\BOTS\AIFootballPredictions\reports\cgm_upcoming_predictions.csv
- Backtest: C:\BOTS\AIFootballPredictions\reports\full_backtest_2025.csv
- League filter: Premier L

SECTION 1 - Last 5 rounds (recent played matchdays)
Found 20 played matches in the most recent 5 matchdays.
      date    league        home        away score bot_prediction_available bot_market_if_any bot_market_result
2026-01-08 Premier L     Arsenal   Liverpool   0-0                       no                                    
2026-01-07 Premier L Bournemouth   Tottenham   3-2                       no                                    
2026-01-07 Premier L   Brentford  Sunderland   3-0                       no                                    
2026-01-07 Premier L     Burnley     Man Utd   2-2                       no                                    
2026-01-07 Premier L    C Palace Aston Villa   0-0                       no                                    
2026-01-07 Premier L     Everton      Wolves   1-1                       no                                    
2026-01-07 Premier L      Fulham     Chelsea   2-1                       no                                    
2026-01-07 Premier L    Man City    Brighton   1-1                       no                                    
2026-01-07 Premier L   Newcastle       Leeds   4-3                       no                                    
2026-01-06 Premier L    West Ham  Nottingham   1-2                       no                                    
2026-01-04 Premier L     Everton   Brentford   2-4                       no                                    
2026-01-04 Premier L      Fulham   Liverpool   2-2                       no                                    
2026-01-04 Premier L       Leeds     Man Utd   1-1                       no                                    
2026-01-04 Premier L    Man City     Chelsea   1-1                       no                                    
2026-01-04 Premier L   Newcastle    C Palace   2-0                       no                                    
2026-01-04 Premier L   Tottenham  Sunderland   1-1                       no                                    
2026-01-03 Premier L Aston Villa  Nottingham   3-1                       no                                    
2026-01-03 Premier L Bournemouth     Arsenal   2-3                       no                                    
2026-01-03 Premier L    Brighton     Burnley   2-0                       no                                    
2026-01-03 Premier L      Wolves    West Ham   3-0                       no                                    

SECTION 2 - Upcoming games and bot recommendation
Found 15 upcoming fixtures in the report window.
Note: 15 fixtures have missing/invalid odds, so no real recommendation was made.
      date    league        home        away  bot_recommendation model_probability_pct odds expected_value_pct official_pick_engine_pick
2026-02-21 Premier L    Man City   Newcastle NO_BET_DATA_MISSING                                                                        
2026-02-21 Premier L Aston Villa       Leeds NO_BET_DATA_MISSING                                                                        
2026-02-21 Premier L   Brentford    Brighton NO_BET_DATA_MISSING                                                                        
2026-02-21 Premier L    C Palace      Wolves NO_BET_DATA_MISSING                                                                        
2026-02-21 Premier L     Chelsea     Burnley NO_BET_DATA_MISSING                                                                        
2026-02-21 Premier L  Nottingham   Liverpool NO_BET_DATA_MISSING                                                                        
2026-02-21 Premier L    West Ham Bournemouth NO_BET_DATA_MISSING                                                                        
2026-02-22 Premier L  Sunderland      Fulham NO_BET_DATA_MISSING                                                                        
2026-02-22 Premier L   Tottenham     Arsenal NO_BET_DATA_MISSING                                                                        
2026-02-23 Premier L     Everton     Man Utd NO_BET_DATA_MISSING                                                                        
2026-02-27 Premier L      Wolves Aston Villa NO_BET_DATA_MISSING                                                                        
2026-02-28 Premier L Bournemouth  Sunderland NO_BET_DATA_MISSING                                                                        
2026-02-28 Premier L    Brighton  Nottingham NO_BET_DATA_MISSING                                                                        
2026-02-28 Premier L     Burnley   Brentford NO_BET_DATA_MISSING                                                                        
2026-02-28 Premier L   Liverpool    West Ham NO_BET_DATA_MISSING                                                                        

SECTION 3 - Backtest summary (historical quality)
Actionable picks analyzed: 160
Wins: 82 | Losses: 78
Hit rate: 51.25%
ROI (1 unit per pick): -5.19%
Average model EV on picks: 8.50%

By market:
  market  picks  wins  win_rate_pct  avg_ev_pct  roi_pct
BTTS_YES    160    82         51.25         8.5    -5.19

Plain-language conclusion:
- Future games are loaded, and the bot is producing recommendations.
- For recent rounds, prediction history could not be matched from available backtest files.
- Current backtest snapshot shows 51.2% hit rate and -5.2% ROI on positive-EV picks.
```
