# TODO List

## ✅ Completed
- [x] Milestone 9: Time Decay Weighting
- [x] Milestone 10: Head-to-Head History
- [x] Milestone 11: League-Specific Features
- [x] Milestone 13: Probability Calibration
- [x] Milestone 14: Streamlit UI Redesign
- [x] Milestone 15: Multi-League Support (12 leagues now supported)
- [x] Milestone 16: Audit Infrastructure (`run_all_audits.py` - 10/10 pass)

## 🚧 In Progress
- [/] **Milestone 17: Telegram Notifications** — Scripts created, needs user setup

### High Priority
- [ ] **Milestone 17: Multi-League Backtest** — Complete backtest for all 12 leagues
  - `run_multi_league_backtest.py` created (needs debugging)
  - Aggregate results across leagues
  - Calibration per league

- [ ] **Milestone 18: Google Sheets Integration** — Push daily picks to a Google Sheet
  - View picks on phone without PC
  - Shareable with friends
  - ~30 min implementation

### Medium Priority
- [ ] **Milestone 19: Cloud Scheduler Automation** — Auto-run predictions daily at 8am
  - No manual triggering needed
  - Uses Google Cloud Functions
  - Free tier available

- [ ] **Milestone 20: Dixon-Coles Adjustment** — Legacy full engine only (1X2); not used in goals-only outputs

### Low Priority / Future
- [ ] Vertex AI Integration (only if scaling to 100+ leagues)
- [ ] Mobile App (Firebase backend)
- [ ] Real-time odds streaming (Pub/Sub)

## 🗒️ Notes
- Google Cloud services are optional enhancements, not critical for core functionality
- Current XGBoost model performs well locally
- Historical data (2018+) is needed for Elo accuracy
- 12 leagues supported: Premier L, Championship, Serie A, Serie B, Primera, Bundesliga, Ligue 1, Ligue 2, Primeira L, Eredivisie, Super Lig, Liga 1
