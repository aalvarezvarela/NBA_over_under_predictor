import pandas as pd
from nba_ou.config.constants import (
    TEAM_NAME_STANDARDIZATION as TEAM_NAME_EQUIVALENT_DICT,
)


def process_odds_df(df_odds, use_metric: str = "most_common"):
    """
    Process odds dataframe with option to use 'average' or 'most_common' metrics.

    Args:
        df_odds: DataFrame with odds data
        use_metric: Either "average" or "most_common" (default: "average")
    """
    # Ensure game_date is datetime (handle both timezone-aware and naive datetimes)
    if not pd.api.types.is_datetime64_any_dtype(df_odds["game_date"]):
        df_odds["game_date"] = pd.to_datetime(df_odds["game_date"], utc=True)
        df_odds["game_date"] = df_odds["game_date"].dt.tz_localize(None)

    df_odds = df_odds[df_odds["most_common_total_line"] > 50]

    df_odds.sort_values(by=["game_date", "team_home"], ascending=False, inplace=True)
    df_odds = df_odds.drop_duplicates(subset=["game_date", "team_home"], keep="first")

    df_odds["game_date_adjusted"] = df_odds["game_date"].copy()

    df_odds = df_odds.sort_values(by="game_date", ascending=False)

    mask = df_odds["game_date_adjusted"].dt.hour < 6
    df_odds.loc[mask, "game_date_adjusted"] = df_odds.loc[
        mask, "game_date_adjusted"
    ] - pd.Timedelta(days=1)
    df_odds["game_date"] = pd.to_datetime(df_odds["game_date"]).dt.strftime("%Y-%m-%d")
    df_odds["game_date_adjusted"] = pd.to_datetime(
        df_odds["game_date_adjusted"]
    ).dt.strftime("%Y-%m-%d")

    df_odds["team_home"] = df_odds["team_home"].map(
        lambda x: TEAM_NAME_EQUIVALENT_DICT.get(x, x)
    )
    df_odds["team_away"] = df_odds["team_away"].map(
        lambda x: TEAM_NAME_EQUIVALENT_DICT.get(x, x)
    )

    # Determine which columns to use based on use_metric parameter
    prefix = use_metric if use_metric in ["average", "most_common"] else "most_common"

    # Step 2: Rename df_new columns to match
    df_odds_renamed = df_odds.rename(
        columns={
            "game_date": "GAME_DATE_ORIGINAL",
            "game_date_adjusted": "GAME_DATE",
            "team_home": "TEAM_NAME_TEAM_HOME",
            "team_away": "TEAM_NAME_TEAM_AWAY",
            f"{prefix}_total_line": "TOTAL_OVER_UNDER_LINE",
            f"{prefix}_moneyline_home": "MONEYLINE_HOME",
            f"{prefix}_moneyline_away": "MONEYLINE_AWAY",
            f"{prefix}_spread_home": "SPREAD_HOME",
        }
    )

    # Step 3: Reorder both dataframes to the same column order
    final_columns = [
        "GAME_DATE",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "MONEYLINE_HOME",
        "MONEYLINE_AWAY",
        "TOTAL_OVER_UNDER_LINE",
        "SPREAD_HOME",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]

    df_odds_processed = df_odds_renamed[final_columns + ["GAME_DATE_ORIGINAL"]]

    df_odds_processed["SPREAD_AWAY"] = -df_odds_processed["SPREAD_HOME"]

    return df_odds_processed
