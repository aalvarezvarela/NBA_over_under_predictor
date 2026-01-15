import numpy as np
import pandas as pd


def add_last_season_playoff_games(df):
    """
    Adds PLAYOFF_GAMES_LAST_SEASON computed per row,
    using only seasons strictly before the game's season.
    """

    # Precompute playoff counts per (season_year, team)
    playoff_counts = (
        df[df["SEASON_ID"].astype(str).str.startswith("4")]
        .groupby(["SEASON_YEAR", "TEAM_ID"])["GAME_ID"]
        .nunique()
        .reset_index(name="PLAYOFF_GAMES")
    )

    # Shift season year to represent "last season"
    playoff_counts["SEASON_YEAR"] += 1

    # Merge back
    df = df.merge(playoff_counts, on=["SEASON_YEAR", "TEAM_ID"], how="left")

    df["PLAYOFF_GAMES_LAST_SEASON"] = df["PLAYOFF_GAMES"].fillna(0)
    df.drop(columns=["PLAYOFF_GAMES"], inplace=True)

    return df


def add_team_record_before_game(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
    wl_col: str = "WL",
    season_type_col: str = "SEASON_TYPE",
    season_id_col: str = "SEASON_ID",
    team_id_col: str = "TEAM_ID",
) -> pd.DataFrame:
    """
    Adds pre-game team record features (wins/losses/record) with no leakage.

    For each row (team-game), computes:
      - GAME_NUMBER: nth game of the season (within season_type, season_id, team_id) [optional]
      - WINS_BEFORE_THIS_GAME
      - LOSSES_BEFORE_THIS_GAME
      - TEAM_RECORD_BEFORE_GAME = wins / (wins + losses), with 0 when denominator is 0

    Assumptions:
      - Each row represents one team in one game.
      - WL column contains 'W' or 'L' (other values will be treated as neither).
      - date_col is comparable (datetime recommended).

    Returns:
      A copy of df with added columns.
    """
    out = df.copy()

    # Ensure datetime and stable sort for cumulative computations
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, team_id_col, season_id_col])

    group_cols = [season_type_col, season_id_col, team_id_col]

    out.sort_values(by=group_cols + [date_col], ascending=True, inplace=True)

    out["GAME_NUMBER"] = out.groupby(group_cols).cumcount() + 1

    # Convert WL to win/loss indicators (non W/L become 0)
    win = (out[wl_col] == "W").astype(int)
    loss = (out[wl_col] == "L").astype(int)

    # Cumulative sums, shifted so they are "before this game"
    out["WINS_BEFORE_THIS_GAME"] = (
        win.groupby([out[c] for c in group_cols]).cumsum().shift(1)
    )
    out["LOSSES_BEFORE_THIS_GAME"] = (
        loss.groupby([out[c] for c in group_cols]).cumsum().shift(1)
    )

    # First game in group will be NaN after shift -> 0
    out["WINS_BEFORE_THIS_GAME"] = out["WINS_BEFORE_THIS_GAME"].fillna(0).astype(int)
    out["LOSSES_BEFORE_THIS_GAME"] = (
        out["LOSSES_BEFORE_THIS_GAME"].fillna(0).astype(int)
    )

    denom = out["WINS_BEFORE_THIS_GAME"] + out["LOSSES_BEFORE_THIS_GAME"]
    out["TEAM_RECORD_BEFORE_GAME"] = np.where(
        denom > 0, out["WINS_BEFORE_THIS_GAME"] / denom, 0.0
    )

    return out


def compute_rest_days_before_match(df):
    """
    Compute the number of rest days before each match for each team.

    Rest days are computed within each SEASON_YEAR to avoid counting
    off-season days between seasons.

    Args:
        df (pd.DataFrame): Team game statistics DataFrame with TEAM_ID,
                           GAME_DATE, and SEASON_YEAR columns

    Returns:
        pd.DataFrame: DataFrame with REST_DAYS_BEFORE_MATCH column added
    """
    df = df.sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"])
    df["REST_DAYS_BEFORE_MATCH"] = (
        df.groupby(["TEAM_ID", "SEASON_YEAR"])["GAME_DATE"]
        .diff()
        .dt.days.fillna(0)
        .astype(int)
    )
    df.sort_values(by="GAME_DATE", ascending=False, inplace=True)
    return df
