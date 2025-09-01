# run_weekly_predictions.py
import subprocess
import sys

# --- CONFIGURATION ---
LEAGUES = ['E0', 'D1'] # The leagues you want to run predictions for

def run_command(command):
    """Runs a command and checks for errors."""
    print(f"\n--- Running command: {' '.join(command)} ---")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running command!")
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def main():
    for league in LEAGUES:
        print(f"\n=========================================")
        print(f"   STARTING PIPELINE FOR LEAGUE: {league}")
        print(f"=========================================")
        
        # Step 1: Automatically generate the fixtures file
        if not run_command([sys.executable, 'fixtures_autogen.py', '--league', league]):
            continue # Skip to next league if fixture generation fails

        # Step 2: Run the fusion predictor using the file we just created
        if not run_command([sys.executable, 'bet_fusion.py', '--league', league]):
            continue # Skip to next league if prediction fails
            
    print("\n\n--- Weekly Prediction Pipeline Complete ---")

if __name__ == "__main__":
    main()