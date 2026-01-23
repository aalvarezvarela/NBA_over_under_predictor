# """
# NBA Over/Under Predictor - Odds Data Utilities

# This module contains utility functions for fetching, processing, and merging
# NBA betting odds data from external APIs.
# """

# import os

# import pandas as pd
# from nba_api.stats.endpoints import LeagueGameFinder
# from tqdm import tqdm

# from nba_ou.fetch_data.fetch_odds_data.get_odds_date import process_odds_date
# from nba_ou.postgre_db.odds.update_odds.upload_to_odds_db import (
#     get_existing_odds_from_db,
#     update_odds_db,
# )


# def update_and_get_odds_df(
#     date_to_predict,
#     odds_folder,
#     df_name,
#     season_to_download,
#     ODDS_API_KEY,
#     BASE_URL,
#     save_csv: bool = False,
# ):
#     HEADERS = {
#         "x-rapidapi-key": ODDS_API_KEY,
#         "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
#     }

#     # Load from database (primary source)
#     season_year = season_to_download[:4]  # Extract year from season like "2024-25"
#     df_odds = get_existing_odds_from_db(season_year)

#     # Ensure game_date is datetime
#     df_odds["game_date"] = pd.to_datetime(df_odds["game_date"])

#     game_finder = LeagueGameFinder(
#         season_nullable=season_to_download, league_id_nullable="00"
#     )
#     games_df = game_finder.get_data_frames()[0]

#     unique_dates = sorted(
#         pd.to_datetime(games_df["GAME_DATE"]).dt.strftime("%Y-%m-%d").unique(),
#         reverse=True,
#     )
#     unique_dates = [
#         date
#         for date in unique_dates
#         if date not in df_odds["game_date"].dt.strftime("%Y-%m-%d").values
#     ]

#     # Loop through valid dates and collect processed data
#     for date in tqdm(unique_dates, desc="Processing odds per date"):
#         # if date is month 10 9 or 8, skip it
#         if pd.to_datetime(date).month in [8, 9, 10]:
#             print(f"Skipping date {date} as it is outside the NBA season.")
#             continue

#         df_day = process_odds_date(date, BASE_URL, HEADERS)
#         if df_day.empty:
#             print(f"No data for {date}")
#             continue
#         # append it to df_odds
#         df_odds = pd.concat([df_odds, df_day], ignore_index=True)
#         df_odds.sort_values(by="game_date", inplace=True, ascending=False)
#         df_odds.reset_index(drop=True, inplace=True)

#     if save_csv and odds_folder:
#         # Save updated odds data to CSV
#         odds_path = os.path.join(odds_folder, df_name)
#         df_odds.to_csv(odds_path, index=False)
#         print(f"✓ Odds data saved to CSV at {odds_path}")
#     # Here update after today and tomorrow are loaded
#     update_odds_db(df_odds)

#     df_today = process_odds_date(date_to_predict, BASE_URL, HEADERS, is_today=True)
#     df_odds = pd.concat([df_odds, df_today], ignore_index=True)
#     # Add tomorrow date as the API sometimes gets this in tomorrows date
#     tomorrow_date = pd.to_datetime(date_to_predict) + pd.Timedelta(days=1)
#     df_tomorrow = process_odds_date(
#         tomorrow_date.strftime("%Y-%m-%d"), BASE_URL, HEADERS, is_today=True
#     )
#     df_odds = pd.concat([df_odds, df_tomorrow], ignore_index=True)
#     df_odds.sort_values(by="game_date", inplace=True, ascending=False)
#     df_odds.reset_index(drop=True, inplace=True)

#     return df_odds


# if __name__ == "__main__":
#     ODDS_API_KEY = ""
#     odds_folder = "/home/adrian_alvarez/Projects/NBA-predictor/odds_data/"

#     df_name = "odds_data.csv"

#     df = update_and_get_odds_df(
#         odds_folder,
#         df_name,
#         ODDS_API_KEY,
#         BASE_URL="https://therundown-therundown-v1.p.rapidapi.com",
#     )
