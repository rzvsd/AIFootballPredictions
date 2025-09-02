
**Ligue 1**:
- ⚽ **Monaco** 🆚 **Reims**: Over 2.5 Goals! 🔥 (73.58% chance)
- ⚽ **St Etienne** 🆚 **Nice**: Under 2.5 Goals (93.4% chance)
- ⚽ **Lens** 🆚 **Le Havre**: Over 2.5 Goals! 🔥 (65.85% chance)
- ⚽ **Paris SG** 🆚 **Lille**: Over 2.5 Goals! 🔥 (66.92% chance)
- ⚽ **Lyon** 🆚 **Brest**: Over 2.5 Goals! 🔥 (92.07% chance)
- ⚽ **Montpellier** 🆚 **Rennes**: Over 2.5 Goals! 🔥 (76.95% chance)
- ⚽ **Auxerre** 🆚 **Strasbourg**: Under 2.5 Goals (70.88% chance)
- ⚽ **Angers** 🆚 **Toulouse**: Over 2.5 Goals! 🔥 (78.66% chance)
- ⚽ **Marseille** 🆚 **Nantes**: Under 2.5 Goals (56.8% chance)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Data Acquisition](#data-acquisition)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Upcoming Matches Acquisition](#upcoming-matches-acquisition)
    - [Set up the API_KEY](#setu-up-the-api_key)
8. [Making Predictions](#making-predictions)
9. [Betting Bot](#betting-bot)
10. [Supported Leagues](#supported-leagues)
11. [Contributing](#contributing)
12. [License](#license)
13. [Disclaimer](#disclaimer)

## Project Overview

AIFootballPredictions aims to create a predictive model to forecast whether a football match will exceed 2.5 goals. The project is divided into four main stages:

1. **Data Acquisition**: Download and merge historical football match data from multiple European leagues.
2. **Data Preprocessing**: Process the raw data to engineer features, handle missing values, and select the most relevant features.
3. **Model Training**: Train several machine learning models, perform hyperparameter tuning, and combine the best models into a voting classifier to make predictions.
4. **Making Predictions**: Use the trained models to predict outcomes for upcoming matches and generate a formatted message for sharing.

## Directory Structure

The project is organized into the following directories:

```
└─── `AIFootballPredictions`
    ├─── `conda`: all the conda environemnts
    ├─── `data`: the folder for the data
    │       ├─── `processed`
    │       └─── `raw`
    ├─── `models`: the folder with the saved and trained models
    ├─── `notebooks`: all the notebooks if any
    └─── `scripts`: all the python scripts
            ├─── `data_acquisition.py`
            ├─── `data_preprocessing.py`
@@ -191,50 +192,78 @@ In order to properly execute the `acquire_next_matches.py` script it is first ne
   - Write down the following line, replacing `your_personal_key` with your actual API key:
     - `API_FOOTBALL_DATA=your_personal_key`

4. **Save and Exit:**
   - Press the `Esc` key to exit insert mode.
   - Then, type `:wq!` and press `Enter` to save the changes and exit the editor.

5. **Verify the Variable:**
   - To check if the variable has been properly set, run the following command from the terminal:
     - `cat ~/.env`
   - You should see the `API_FOOTBALL_DATA` variable listed with your API key.

## Making Predictions

To predict the outcomes for upcoming matches and generate a formatted message for sharing, run the `make_predictions.py` script:

```bash
python scripts/make_predictions.py --models_dir models --data_dir data/processed --output_file final_predictions.txt --json_competitions data/next_matches.json
```
This script will:

- Load the pre-trained models and the processed data.
- Make predictions for upcoming matches based on the next matches data.
- Format the predictions into a redable `.txt` message and save it to the specified output file.

## Betting Bot

The optional betting bot can place wagers based on the model's predictions.

### Prerequisites

1. A working Python 3 environment with this project's dependencies installed.
2. Trained models and recent match data.
3. Access to a betting platform that allows automated wagers.

### Legal Considerations

Always verify that automated betting is legal in your jurisdiction and permitted by your betting platform. The authors do not encourage gambling where it is prohibited.

### Setup Steps

1. Review the predictions generated in the previous step.
2. Configure any required credentials or API keys for your betting platform.
3. Run the bot for a specific league:

```bash
python scripts/betting_bot.py --league E0
```

### Risk Warning

Betting involves financial risk. Use the bot responsibly and comply with all local regulations.

## Supported Leagues

For the moment, the team name mapping has been done manually. The predictions currently support the following leagues:

- *Premier League*: **E0**
- *Serie A*: **I1**
- *Ligue 1*: **F1**
- *La Liga (Primera Division)*: **SP1**
- *Bundesliga*: **D1**

For this reason be carful when executing the [data acquisition](#data-acquisition) step. 

## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [BSD-3-Claude license](LICENSE) - see the `LICENSE` file for details.

## Disclaimer

This project is intended for educational and informational purposes only. While the AIFootballPredictions system aims to provide accurate predictions for football matches, it is important to understand that predictions are inherently uncertain and should not be used as the sole basis for any decision-making, including betting or financial investments.

The predictions generated by this system can be used as an additional tool during the decision-making process. However, they should be considered alongside other factors and sources of information.