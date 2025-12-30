"""
NBA Over/Under Predictor - Odds Update Script

This script manages the update of NBA betting odds data by fetching new odds
from external APIs.
"""

from utils.general_utils import get_nba_season_nullable

from .update_odds_utils import update_and_get_odds_df


def update_odds(date_to_predict, odds_folder: str, ODDS_API_KEY, BASE_URL, save_csv: bool = False):
    # Get Season to Update
    season_nullable = get_nba_season_nullable(date_to_predict)

    df_name = "odds_data.csv"

    df_odds = update_and_get_odds_df(
        date_to_predict, odds_folder, df_name, season_nullable, ODDS_API_KEY, BASE_URL, save_csv
    )

    return df_odds


if __name__ == "__main__":
    # Define Data Folder
    ODDS_DATA_FOLDER = ".././odds_data/"
    update_odds(ODDS_DATA_FOLDER)
