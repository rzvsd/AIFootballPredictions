# fixtures_template.py
# Creates blank CSV templates for next Tuesday → next Monday for the leagues you list.

import os
from datetime import date, timedelta

LEAGUES = ["E0"]  # add "D1", "E1", etc. when you’re ready

def next_tue_to_mon():
    today = date.today()
    days_to_tue = (1 - today.weekday()) % 7
    start = today + timedelta(days=days_to_tue or 7)  # always NEXT Tuesday
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()

def main():
    dfrom, dto = next_tue_to_mon()
    os.makedirs(os.path.join("data", "fixtures"), exist_ok=True)

    for lg in LEAGUES:
        path = os.path.join("data", "fixtures", f"{lg}_{dfrom}_to_{dto}.csv")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("date,home_team,away_team\n")
            print(f"Created: {path}")
        else:
            print(f"Exists:  {path}")

    print("\nPaste fixtures using ISO UTC, e.g.:")
    print("2025-08-26T18:30:00Z,Man City,Chelsea")
    print("2025-08-27T19:00:00Z,Arsenal,Tottenham")

if __name__ == "__main__":
    main()
