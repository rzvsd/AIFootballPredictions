ELO V2 - Simple Explanation (Non-Technical)

What this is
- ELO is the bot's "team strength score".
- Every team starts around the same level.
- After each match, scores move up or down based on what happened.

What changed in V2
- Before: one global rule for all leagues.
- Now: each league can have its own behavior.
- The bot now treats matches with more context:
  - league type,
  - match type (league/cup/playoff/friendly),
  - goal difference,
  - surprise results (underdog wins),
  - new teams with little history.

How it works in simple words
1. Before a match, each team has a strength score.
2. The bot estimates what should happen.
3. After the match, it compares expectation vs reality.
4. Team scores are updated:
   - small change if result was expected,
   - bigger change if result was surprising,
   - bigger/smaller change depending on match importance and league settings.

Why this is better
- More realistic across different leagues.
- Better reaction to important surprises.
- Better handling of teams with very little data.
- More stable over time (big wins are controlled with caps).

What this means for predictions
- Team strength inputs are now cleaner and more relevant.
- Home advantage is no longer a single global number for every league.
- The prediction engine reads the new ELO V2 values directly.

Safety and transparency
- We keep detailed per-match trace fields (internal logs) so we can audit:
  - what factors were used,
  - why a rating moved,
  - how much it moved.

Important note
- ELO is one component, not the whole model.
- Final picks still combine ELO with xG/pressure/form logic and EV filters.

