- (Process) Define a "multi-agent pipeline" concept (builder/tester/QA gates) as repeatable scripts + CI checks, then list them as `AGENTS.md` tasks.
- (Data) Export multi-season EPL stats into `cgmbetdatabase.csv` (or `.xls`) to push Pressure/xG coverage toward 100% for those seasons (current coverage is current-season-only).
- (Milestone 3) Tune `sterile_*` / `assassin_*` thresholds and decide whether to standardize xG/Pressure z-scores using only teams with evidence (vs including neutral defaults).
- (Data) Expand scope to multiple leagues + multiple seasons (target: >= 5 seasons per league) while keeping strict "history vs upcoming" separation (history can be huge; upcoming must be future-only).
- (Model) Add Dixon-Coles adjustment for 1X2 probabilities as an extra layer over Poisson (apply only to 1X2; keep O/U derived from base goal model unless explicitly extended).

## Upcoming Milestones

- **Milestone 12: Dixon-Coles Adjustment** — Apply Dixon-Coles correction to 1X2 probabilities for low-scoring matches.
- **Milestone 13: Multi-League Expansion** — Expand data scope to 5+ leagues to fully activate league-specific features.


