import argparse
import csv
from pathlib import Path


def summarize(log_file: str = "data/bets_log.csv") -> None:
    path = Path(log_file)
    if not path.exists():
        print("Log file not found:", log_file)
        return

    total_bets = 0
    wins = 0
    total_stake = 0.0
    total_profit = 0.0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_bets += 1
            stake = float(row["stake"])
            profit = float(row["profit_loss"])
            total_stake += stake
            total_profit += profit
            if row["result"].lower() == "win":
                wins += 1

    if total_bets == 0 or total_stake == 0:
        print("No bet data available.")
        return

    roi = (total_profit / total_stake) * 100
    win_rate = (wins / total_bets) * 100

    print(f"Total bets: {total_bets}")
    print(f"Total stake: {total_stake:.2f}")
    print(f"Total profit: {total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    print(f"Win rate: {win_rate:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize betting log")
    parser.add_argument("--log-file", default="data/bets_log.csv", help="Path to bets log")
    args = parser.parse_args()
    summarize(args.log_file)


if __name__ == "__main__":
    main()