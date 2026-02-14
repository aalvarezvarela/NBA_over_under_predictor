from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.postgre_db.odds.update_odds.update_odds_database import update_odds_database


def _mode_or_none(s: pd.Series) -> Optional[float]:
    """Return statistical mode (most common) for numeric series; None if empty."""
    s = s.dropna()
    if s.empty:
        return None
    # If multiple modes, take the smallest for determinism (could also take first).
    modes = s.mode()
    if modes.empty:
        return None
    return float(modes.min())


def build_games_odds_df_for_postgres(
    csv_or_df: Union[str, Path, pd.DataFrame],
    *,
    team_name_map: Optional[dict[str, str]] = None,
    dedupe_keep: str = "first",
) -> pd.DataFrame:
    """
    Convert raw odds CSV (one row per team perspective) into one row per game:
    (game_date, team_home, team_away) with "most_common_*" and "average_*" metrics.

    Input columns expected:
      date, season, team, home/visitor, opponent, score, opponentScore,
      moneyLine, opponentMoneyLine, total, spread, secondHalfTotal

    Output columns:
      game_date, team_home, team_away,
      most_common_total_line, average_total_line,
      most_common_moneyline_home, average_moneyline_home,
      most_common_moneyline_away, average_moneyline_away,
      most_common_spread_home, average_spread_home,
      most_common_spread_away, average_spread_away,
      average_total_over_money, average_total_under_money,
      most_common_total_over_money, most_common_total_under_money,
      season_year

    Last four requested as None (no data).
    """
    if isinstance(csv_or_df, (str, Path)):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df.copy()

    required = [
        "date",
        "season",
        "team",
        "home/visitor",
        "opponent",
        "moneyLine",
        "opponentMoneyLine",
        "total",
        "spread",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse date (store as date string YYYY-MM-DD to be Postgres-friendly, or keep as date)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(5)
        raise ValueError(f"Some 'date' values could not be parsed. Examples:\n{bad}")

    # Normalize home/away using the same rule as your old code:
    # "vs" means the 'team' column is the home team.
    is_home = df["home/visitor"].eq("vs")

    df["team_home"] = np.where(is_home, df["team"], df["opponent"])
    df["team_away"] = np.where(is_home, df["opponent"], df["team"])

    df["moneyline_home"] = np.where(is_home, df["moneyLine"], df["opponentMoneyLine"])
    df["moneyline_away"] = np.where(is_home, df["opponentMoneyLine"], df["moneyLine"])

    # Spread: assume input `spread` is the spread for the listed `team` (common format).
    # Convert to home/away spreads so that away = -home.
    spread_home = np.where(is_home, df["spread"], -df["spread"])
    df["spread_home"] = spread_home
    df["spread_away"] = -spread_home

    df["total_line"] = df["total"]

    # Numeric coercion (important for aggregation / Postgres types)
    for c in [
        "moneyline_home",
        "moneyline_away",
        "spread_home",
        "spread_away",
        "total_line",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Team name mapping (optional but usually needed)
    mapping = team_name_map or TEAM_NAME_STANDARDIZATION
    df["team_home"] = df["team_home"].map(mapping).fillna(df["team_home"])
    df["team_away"] = df["team_away"].map(mapping).fillna(df["team_away"])

    # Optional: drop self-matches (data quality guardrail)
    df = df[df["team_home"] != df["team_away"]].copy()

    # Aggregate to one row per game in case multiple rows exist (multiple books/snapshots)
    game_key = ["date", "team_home", "team_away"]

    agg = df.groupby(game_key, as_index=False).agg(
        most_common_total_line=("total_line", _mode_or_none),
        average_total_line=(
            "total_line",
            lambda s: float(s.dropna().mean()) if s.dropna().size else None,
        ),
        most_common_moneyline_home=("moneyline_home", _mode_or_none),
        average_moneyline_home=(
            "moneyline_home",
            lambda s: float(s.dropna().mean()) if s.dropna().size else None,
        ),
        most_common_moneyline_away=("moneyline_away", _mode_or_none),
        average_moneyline_away=(
            "moneyline_away",
            lambda s: float(s.dropna().mean()) if s.dropna().size else None,
        ),
        most_common_spread_home=("spread_home", _mode_or_none),
        average_spread_home=(
            "spread_home",
            lambda s: float(s.dropna().mean()) if s.dropna().size else None,
        ),
        most_common_spread_away=("spread_away", _mode_or_none),
        average_spread_away=(
            "spread_away",
            lambda s: float(s.dropna().mean()) if s.dropna().size else None,
        ),
    )

    # Calculate season_year based on game date
    # NBA season logic: Jan-Jul games belong to previous year's season, Aug-Dec to current year
    def calculate_season_year(date):
        if pd.isna(date):
            return None
        # Convert date object to datetime to access month
        if hasattr(date, "month"):
            month = date.month
            year = date.year
        else:
            dt = pd.to_datetime(date)
            month = dt.month
            year = dt.year
        # January to July → season_year = year - 1
        # August to December → season_year = year
        return year - 1 if month in [1, 2, 3, 4, 5, 6, 7] else year

    agg["season_year"] = agg["date"].apply(calculate_season_year)

    # Last four requested as None
    agg["average_total_over_money"] = None
    agg["average_total_under_money"] = None
    agg["most_common_total_over_money"] = None
    agg["most_common_total_under_money"] = None

    # Rename date column to game_date and order columns exactly as requested
    agg = agg.rename(columns={"date": "game_date"})

    final_cols = [
        "game_date",
        "team_home",
        "team_away",
        "most_common_total_line",
        "average_total_line",
        "most_common_moneyline_home",
        "average_moneyline_home",
        "most_common_moneyline_away",
        "average_moneyline_away",
        "most_common_spread_home",
        "average_spread_home",
        "most_common_spread_away",
        "average_spread_away",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
        "season_year",
    ]
    agg = agg[final_cols].copy()

    # Deterministic sort (nice for debugging)
    agg = agg.sort_values(
        ["game_date", "team_home"], ascending=[False, True]
    ).reset_index(drop=True)

    return agg


if __name__ == "__main__":
    # Path to your historical odds CSV
    csv_path = (
        Path(__file__).resolve().parents[3] / "data" / "odds_data" / "oddsData.csv"
    )

    print(f"Loading historical odds data from: {csv_path}")

    # Build the DataFrame in the format expected by PostgreSQL
    df_odds = build_games_odds_df_for_postgres(csv_path)

    print(f"Processed {len(df_odds)} games from historical data")
    print(f"Date range: {df_odds['game_date'].min()} to {df_odds['game_date'].max()}")
    print(f"\nSample rows:\n{df_odds.head()}")

    # Upload to PostgreSQL using the existing update function
    print("\nUploading to PostgreSQL...")

    success = update_odds_database(df_odds)

    if success:
        print("✅ Historical odds data successfully uploaded to database!")
    else:
        print("❌ Failed to upload historical odds data")
