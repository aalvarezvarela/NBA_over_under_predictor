import pandas as pd

from nba_ou.config.constants import (
    TEAM_NAME_STANDARDIZATION as TEAM_NAME_EQUIVALENT_DICT,
)
from nba_ou.data_preparation.odds.process_odds_data import process_odds_df


def merge_teams_df_with_odds(df_odds, df_team, use_metric: str = "most_common"):
    """
    Merge team dataframe with odds data.

    Args:
        df_odds: Odds dataframe
        df_team: Team dataframe
        use_metric: Either "average" or "most_common" (default: "most_common")
    """
    df_odds = process_odds_df(df_odds, use_metric=use_metric)

    df_team["GAME_DATE"] = pd.to_datetime(df_team["GAME_DATE"], format="%Y-%m-%d")

    df_team["TEAM_NAME"] = df_team["TEAM_NAME"].map(
        lambda x: TEAM_NAME_EQUIVALENT_DICT.get(x, x)
    )

    df_odds[df_odds.duplicated(subset=["GAME_DATE", "TEAM_NAME_TEAM_HOME"], keep=False)]

    # Find duplicated rows based on GAME_DATE and TEAM_NAME_TEAM_HOME
    dupes_mask = df_odds.duplicated(
        subset=["GAME_DATE", "TEAM_NAME_TEAM_HOME"], keep=False
    )

    # Replace GAME_DATE with GAME_DATE_ORIGINAL only in those duplicated rows
    df_odds.loc[dupes_mask, "GAME_DATE"] = df_odds.loc[dupes_mask, "GAME_DATE_ORIGINAL"]

    df_odds[df_odds.duplicated(subset=["GAME_DATE", "TEAM_NAME_TEAM_AWAY"], keep=False)]
    dupes_mask = df_odds.duplicated(
        subset=["GAME_DATE", "TEAM_NAME_TEAM_AWAY"], keep=False
    )

    # Replace GAME_DATE with GAME_DATE_ORIGINAL only in those duplicated rows
    df_odds.loc[dupes_mask, "GAME_DATE"] = df_odds.loc[dupes_mask, "GAME_DATE_ORIGINAL"]

    df_team["GAME_DATE"] = pd.to_datetime(df_team["GAME_DATE"])
    df_odds["GAME_DATE"] = pd.to_datetime(df_odds["GAME_DATE"])
    df_odds["GAME_DATE_ORIGINAL"] = pd.to_datetime(df_odds["GAME_DATE_ORIGINAL"])

    # Home teams from df_odds
    df_home = df_odds[
        [
            "GAME_DATE",
            "TEAM_NAME_TEAM_HOME",
            "MONEYLINE_HOME",
            "TOTAL_OVER_UNDER_LINE",
            "SPREAD_HOME",
            "GAME_DATE_ORIGINAL",
            "average_total_over_money",
            "average_total_under_money",
            "most_common_total_over_money",
            "most_common_total_under_money",
        ]
    ].copy()

    df_home = df_home.rename(
        columns={
            "TEAM_NAME_TEAM_HOME": "TEAM_NAME",
            "MONEYLINE_HOME": "MONEYLINE",
            "SPREAD_HOME": "SPREAD",
        }
    )
    df_home["HOME"] = True

    # Away teams from df_odds
    df_away = df_odds[
        [
            "GAME_DATE",
            "TEAM_NAME_TEAM_AWAY",
            "MONEYLINE_AWAY",
            "TOTAL_OVER_UNDER_LINE",
            "SPREAD_AWAY",
            "GAME_DATE_ORIGINAL",
            "average_total_over_money",
            "average_total_under_money",
            "most_common_total_over_money",
            "most_common_total_under_money",
        ]
    ].copy()

    df_away = df_away.rename(
        columns={
            "TEAM_NAME_TEAM_AWAY": "TEAM_NAME",
            "MONEYLINE_AWAY": "MONEYLINE",
            "SPREAD_AWAY": "SPREAD",
        }
    )
    df_away["HOME"] = False

    # Combined betting data for "first-pass" merge

    # 1) Prepare df_betting (same as before)
    df_betting = pd.concat([df_home, df_away], ignore_index=True)

    # Also prepare a version keyed on GAME_DATE_ORIGINAL
    df_betting_original = df_betting.drop(columns="GAME_DATE").rename(
        columns={"GAME_DATE_ORIGINAL": "GAME_DATE"}
    )

    # 2) Merge #1: on GAME_DATE
    df_team1 = df_team.merge(
        df_betting.drop(columns="GAME_DATE_ORIGINAL", errors="ignore"),
        how="left",
        on=["GAME_DATE", "TEAM_NAME", "HOME"],
        suffixes=("", "_betting"),
    )

    # 3) Merge #2: on GAME_DATE_ORIGINAL
    df_team2 = df_team.merge(
        df_betting_original,
        how="left",
        on=["GAME_DATE", "TEAM_NAME", "HOME"],
        suffixes=("", "_betting"),
    )

    # 4) Coalesce columns of interest
    df_team_final = df_team1.copy()

    for col in [
        "MONEYLINE",
        "SPREAD",
        "TOTAL_OVER_UNDER_LINE",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]:
        # If the first merge is missing data, fill from the second merge
        df_team_final[col] = df_team1[col].fillna(df_team2[col])

    mask_missing_tot = df_team_final["TOTAL_OVER_UNDER_LINE"].isna()
    df_missing_tot = df_team_final[mask_missing_tot].copy()

    # 2) Merge ignoring HOME, which can produce duplicates
    df_missing_tot_merged = df_missing_tot.merge(
        df_betting, how="left", on=["GAME_DATE", "TEAM_NAME"], suffixes=("", "_inv")
    )

    # 3) Now condense df_missing_tot_merged so that each (GAME_DATE, TEAM_NAME)
    #    appears at most once. We'll just pick the first valid TOT found:
    df_missing_tot_merged = (
        df_missing_tot_merged.dropna(
            subset=["TOTAL_OVER_UNDER_LINE_inv"]
        )  # keep rows that actually have TOT
        .drop_duplicates(subset=["GAME_DATE", "TEAM_NAME"], keep="first")
        .copy()
    )

    # 4) Rename so we only keep one TOT column
    df_missing_tot_merged["TOTAL_OVER_UNDER_LINE_new"] = df_missing_tot_merged[
        "TOTAL_OVER_UNDER_LINE"
    ].fillna(df_missing_tot_merged["TOTAL_OVER_UNDER_LINE_inv"])

    # 5) Merge that condensed info back into df_team_final
    df_team_final = df_team_final.merge(
        df_missing_tot_merged[["GAME_DATE", "TEAM_NAME", "TOTAL_OVER_UNDER_LINE_new"]],
        how="left",
        on=["GAME_DATE", "TEAM_NAME"],
    )

    # 6) Fill the original TOT where it was missing
    df_team_final["TOTAL_OVER_UNDER_LINE"] = df_team_final[
        "TOTAL_OVER_UNDER_LINE"
    ].fillna(df_team_final["TOTAL_OVER_UNDER_LINE_new"])

    # 7) Drop the helper column
    df_team_final.drop(columns=["TOTAL_OVER_UNDER_LINE_new"], inplace=True)

    # drop duplicates
    df_team_final.drop_duplicates(
        subset=["GAME_DATE", "TEAM_NAME", "HOME"], keep="first", inplace=True
    )

    return df_team_final
