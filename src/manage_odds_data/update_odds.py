import os
import sys

import pandas as pd
from .update_odds_utils import update_odds_df

def update_odds(date_to_predict,odds_folder: str, ODDS_API_KEY, BASE_URL):
    # from .update_database_utils import fetch_nba_data, get_nba_season_to_update  # Try to sort the 300 games block issue
    remove_modules = [
        module
        for module in sys.modules
        if module.startswith(
            (
                "requests",
                "urllib",
                "urllib3",
                "chardet",
                "charset_normalizer",
                "idna",
                "certifi",
                "http",
                "socket",
                "json",
                "ssl",
                "h11",
                "h2",
                "hpack",
                "brotli",
                "zlib",
                "nba",
            )
        )
    ]

    for module in remove_modules:
        del sys.modules[module]

    # Remove `update_database_utils` module if it exists
    if "src.update_database_utils" in sys.modules:
        del sys.modules["src.update_database_utils"]

    from .update_database_utils import get_nba_season_to_update

    # Get Season to Update
    season_nullable = get_nba_season_to_update()

    df_name = "odds_data.csv"

    df_odds = update_odds_df(date_to_predict,odds_folder, df_name, season_nullable, ODDS_API_KEY, BASE_URL)
    
    return df_odds


if __name__ == "__main__":
    # Define Data Folder
    ODDS_DATA_FOLDER = ".././odds_data/"
    update_odds(ODDS_DATA_FOLDER)
