import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION
from nba_ou.config.settings import SETTINGS
from nba_ou.data_preparation.referees.process_refs_scheduled_game import (
    process_scheduled_referee_assignments,
)
from nba_ou.data_preparation.scheduled_games.manage_injury_data import (
    process_injury_data,
)
from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    retrieve_injury_report_as_df,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import scrape_sportsbook_days
from nba_ou.fetch_data.odds_yahoo.process_yahoo_day import yahoo_one_row_per_game
from nba_ou.fetch_data.odds_yahoo.scrape_yahoo import scrape_yahoo_days
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games


def merge_odds_with_scheduled_games(
    odds_df: pd.DataFrame, scheduled_games_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge odds data with scheduled games to add game_id.

    Merges based on game_date, team_home, and team_away to attach the game_id
    from scheduled games to the odds data.

    Args:
        odds_df (pd.DataFrame): Odds dataframe with game_date, team_home, team_away
        scheduled_games_df (pd.DataFrame): Scheduled games with game_id, game_date, team_home, team_away

    Returns:
        pd.DataFrame: Odds data with game_id added
    """
    # drop game_id cpolumn in odds_df if it exists to avoid duplicates after merge
    if odds_df.empty:
        raise ValueError("Odds dataframe is empty, cannot merge with scheduled games")

    if scheduled_games_df.empty:
        print("No scheduled games data provided for merging")
        raise ValueError(
            "Scheduled games dataframe is empty, cannot merge with odds data"
        )

    odds_df = odds_df.drop(columns=["game_id"], errors="ignore")

    # Normalize team names in both dataframes using TEAM_NAME_STANDARDIZATION
    odds_df = odds_df.copy()
    scheduled_games_df = scheduled_games_df.copy()

    # Apply normalization to team_home and team_away in odds_df
    if "team_home" in odds_df.columns:
        odds_df["team_home"] = (
            odds_df["team_home"]
            .map(TEAM_NAME_STANDARDIZATION)
            .fillna(odds_df["team_home"])
        )
    if "team_away" in odds_df.columns:
        odds_df["team_away"] = (
            odds_df["team_away"]
            .map(TEAM_NAME_STANDARDIZATION)
            .fillna(odds_df["team_away"])
        )

    # Apply normalization to team_home and team_away in scheduled_games_df
    if "team_home" in scheduled_games_df.columns:
        scheduled_games_df["team_home"] = (
            scheduled_games_df["team_home"]
            .map(TEAM_NAME_STANDARDIZATION)
            .fillna(scheduled_games_df["team_home"])
        )
    if "team_away" in scheduled_games_df.columns:
        scheduled_games_df["team_away"] = (
            scheduled_games_df["team_away"]
            .map(TEAM_NAME_STANDARDIZATION)
            .fillna(scheduled_games_df["team_away"])
        )

    # Ensure required columns exist in scheduled_games_df
    required_cols = ["game_id", "game_date", "team_home", "team_away"]
    missing_cols = [
        col for col in required_cols if col not in scheduled_games_df.columns
    ]
    if missing_cols:
        print(f"Missing columns in scheduled_games: {missing_cols}")
        return odds_df

    # Prepare merge keys
    merge_keys = ["game_date", "team_home", "team_away"]

    # Select only needed columns from scheduled_games to avoid duplicates
    scheduled_subset = scheduled_games_df[
        ["game_id", "game_date", "team_home", "team_away"]
    ].copy()

    # Merge to add game_id
    merged_df = odds_df.merge(
        scheduled_subset,
        on=merge_keys,
        how="left",
    )

    return merged_df


def get_yahoo_prediction_data(
    date_to_predict: str, scheduled_games: pd.DataFrame, *, headless: bool
) -> pd.DataFrame:
    """Get Yahoo odds data for the prediction date.

    Fetches odds for both the date and date+1 to account for timezone differences
    (Yahoo uses local time in Europe). Merges with scheduled games to get game_id.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        scheduled_games (pd.DataFrame): DataFrame with scheduled games including game_id

    Returns:
        pd.DataFrame: Yahoo odds data merged with game_id for the scheduled games
    """
    # Parse the date and get date + 1
    base_date = datetime.strptime(date_to_predict, "%Y-%m-%d").date()
    next_date = base_date + timedelta(days=1)

    # Scrape both dates to account for timezone differences
    days_to_scrape = [base_date, next_date]
    df_yahoo = asyncio.run(scrape_yahoo_days(days_to_scrape, headless=headless))

    # Check if we got any data
    if df_yahoo.empty:
        print(f"No Yahoo odds data found for {date_to_predict}")
        return pd.DataFrame()

    # Convert from 2 rows per game to 1 row per game
    df_yahoo_processed = yahoo_one_row_per_game(df_yahoo)

    # Merge with scheduled_games to get the game_id
    df_yahoo_with_game_id = merge_odds_with_scheduled_games(
        df_yahoo_processed, scheduled_games
    )
    # drop NA rows on game ID as they are not shceuled games
    df_yahoo_with_game_id = df_yahoo_with_game_id.dropna(subset=["game_id"])

    return df_yahoo_with_game_id


def get_sportsbook_prediction_data(
    date_to_predict: str, scheduled_games: pd.DataFrame, *, headless: bool
) -> pd.DataFrame:
    """Get Sportsbook Review odds data for the prediction date.

    Fetches odds (totals, spread, moneyline) for date of prediction, merges with scheduled games to get game_id.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        scheduled_games (pd.DataFrame): DataFrame with scheduled games including game_id

    Returns:
        pd.DataFrame: Sportsbook odds data merged with game_id for the scheduled games
    """
    # Parse the date and get date + 1
    base_date = datetime.strptime(date_to_predict, "%Y-%m-%d").date()

    # Scrape both dates to account for timezone differences
    days_to_scrape = [base_date]
    df_sportsbook = asyncio.run(
        scrape_sportsbook_days(days_to_scrape, headless=headless)
    )

    # Check if we got any data
    if df_sportsbook.empty:
        print(f"No Sportsbook odds data found for {date_to_predict}")
        raise ValueError("No Sportsbook odds data found")

    # Merge with scheduled_games to get the game_id
    df_sportsbook_with_game_id = merge_odds_with_scheduled_games(
        df_sportsbook, scheduled_games
    )
    # drop NA rows on game ID as they are not scheduled games
    df_sportsbook_with_game_id = df_sportsbook_with_game_id.dropna(subset=["game_id"])

    return df_sportsbook_with_game_id


def get_all_info_for_scheduled_games(
    date_to_predict: str,
    nba_injury_reports_url,
    save_reports_path = None,
    headless: bool | None = None,
) -> dict:
    """Get all information needed for scheduled games prediction.

    Fetches scheduled games, referee assignments, injury data, and odds from
    both Yahoo and Sportsbook Review for the specified date.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        nba_injury_reports_url: URL for NBA injury reports
        save_reports_path: Path to save injury reports
        headless (bool | None): Playwright mode. If None, uses SETTINGS.headless.

    Returns:
        dict: Dictionary containing:
            - scheduled_games (pd.DataFrame): Scheduled games data
            - df_referees_scheduled (pd.DataFrame): Referee assignments
            - injury_dict_scheduled (dict): Injury information
            - df_odds_yahoo_scheduled (pd.DataFrame): Yahoo odds data
            - df_odds_sportsbook_scheduled (pd.DataFrame): Sportsbook odds data
    """
    if not date_to_predict:
        date_to_predict = pd.Timestamp.now(tz=ZoneInfo("US/Eastern")).strftime(
            "%Y-%m-%d"
        )
    if headless is None:
        headless = SETTINGS.headless

    # First Get the games itself
    scheduled_games = get_schedule_games(date_to_predict)
    if scheduled_games.empty:
        print(f"No scheduled games found for {date_to_predict}")
        return {
            "scheduled_games": pd.DataFrame(),
            "df_referees_scheduled": pd.DataFrame(),
            "injury_dict_scheduled": {},
            "df_odds_yahoo_scheduled": pd.DataFrame(),
            "df_odds_sportsbook_scheduled": pd.DataFrame(),
        }

    # Convert team IDs to team names using TEAM_ID_MAP
    # Create reverse mapping: ID -> Name
    id_to_name = {team_id: name for name, team_id in TEAM_ID_MAP.items()}

    # Create a temporary DataFrame with the additional columns for merging
    scheduled_games_for_merge = scheduled_games.copy()
    scheduled_games_for_merge["team_home"] = (
        scheduled_games_for_merge["HOME_TEAM_ID"].astype(str).map(id_to_name)
    )
    scheduled_games_for_merge["team_away"] = (
        scheduled_games_for_merge["VISITOR_TEAM_ID"].astype(str).map(id_to_name)
    )
    scheduled_games_for_merge["game_date"] = pd.to_datetime(
        scheduled_games_for_merge["GAME_DATE_EST"]
    ).dt.date
    scheduled_games_for_merge["game_id"] = scheduled_games_for_merge["GAME_ID"].astype(
        str
    )

    # Then get refs
    df_referees_scheduled = process_scheduled_referee_assignments(scheduled_games)
    # Then Injuries
    injury_report_df = retrieve_injury_report_as_df(
        nba_injury_reports_url, reports_path=save_reports_path
    )

    injury_dict_scheduled, games_not_updated = process_injury_data(
        scheduled_games, injury_report_df
    )

    if len(games_not_updated) == len(scheduled_games):
        raise ValueError("No games were updated with injury data")

    print("Fetched and processed scheduled games, referees, and injuries")
    print("Processing odds data for scheduled games...")
    # Fetch Yahoo odds data for the scheduled games using the temp dataframe
    df_odds_yahoo_scheduled = get_yahoo_prediction_data(
        date_to_predict, scheduled_games_for_merge, headless=headless
    )
    df_odds_sportsbook_scheduled = get_sportsbook_prediction_data(
        date_to_predict, scheduled_games_for_merge, headless=headless
    )

    return {
        "scheduled_games": scheduled_games,
        "df_referees_scheduled": df_referees_scheduled,
        "injury_dict_scheduled": injury_dict_scheduled,
        "df_odds_yahoo_scheduled": df_odds_yahoo_scheduled,
        "df_odds_sportsbook_scheduled": df_odds_sportsbook_scheduled,
    }
