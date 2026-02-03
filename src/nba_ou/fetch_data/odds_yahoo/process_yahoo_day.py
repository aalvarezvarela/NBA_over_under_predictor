from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_matchup(matchup: str) -> Tuple[str, str]:
    """
    "Washington Wizards @ New York Knicks" -> ("Washington Wizards", "New York Knicks")
    """
    if not isinstance(matchup, str) or " @ " not in matchup:
        return (pd.NA, pd.NA)
    away, home = matchup.split(" @ ", 1)
    return away.strip(), home.strip()


def build_one_game_row_from_yahoo_group(g: pd.DataFrame) -> dict:
    """
    Yahoo CSV has 2 rows per matchup (away + home).
    Fields like pct_bets/pct_money are repeated across both rows, so we take them from the first row.
    Spread/money_line are team-specific, so we split them into away/home.
    Total_line is shared; total_side tells which row is Over vs Under (O/U). We'll split if needed.
    """
    g = g.copy()

    # Determine away/home teams from matchup string (authoritative)
    matchup = g["matchup"].iloc[0]
    team_away_name, team_home_name = _parse_matchup(matchup)

    # Identify rows for away and home using team_name matching (fallback to first/second row if mismatch)
    away_row = g.loc[g["team_name"] == team_away_name].head(1)
    home_row = g.loc[g["team_name"] == team_home_name].head(1)

    if len(away_row) == 0 or len(home_row) == 0:
        # fallback: assume row order is away then home
        away_row = g.head(1)
        home_row = g.iloc[1:2] if len(g) > 1 else g.head(0)

    # Identify O/U rows for totals (total_side)
    over_row = g.loc[g["total_side"].astype(str).str.upper() == "O"].head(1)
    under_row = g.loc[g["total_side"].astype(str).str.upper() == "U"].head(1)

    # Base shared columns
    base = g.iloc[0]

    out = {
        "season_year": base["season"],
        "game_date": pd.to_datetime(base["date"], errors="coerce").date(),
        "matchup": matchup,
        "team_away": team_away_name,
        "team_home": team_home_name,
        "team_away_abbr": away_row["team_abbr"].iloc[0] if len(away_row) else pd.NA,
        "team_home_abbr": home_row["team_abbr"].iloc[0] if len(home_row) else pd.NA,
    }

    # Team-specific markets (spread + moneyline)
    out["spread_away"] = _to_num(away_row["spread"]).iloc[0] if len(away_row) else pd.NA
    out["spread_home"] = _to_num(home_row["spread"]).iloc[0] if len(home_row) else pd.NA
    out["moneyline_away"] = (
        _to_num(away_row["money_line"]).iloc[0] if len(away_row) else pd.NA
    )
    out["moneyline_home"] = (
        _to_num(home_row["money_line"]).iloc[0] if len(home_row) else pd.NA
    )

    # Totals: line is shared; keep one, and optionally keep O/U split as well
    out["total_line"] = _to_num(pd.Series([base.get("total_line")])).iloc[0]
    out["total_line_over"] = (
        _to_num(over_row["total_line"]).iloc[0] if len(over_row) else pd.NA
    )
    out["total_line_under"] = (
        _to_num(under_row["total_line"]).iloc[0] if len(under_row) else pd.NA
    )

    # Public bet % and money % fields (these are game-level, repeated in both rows)
    # Spread
    out["spread_pct_bets_away"] = _to_num(
        pd.Series([base.get("spread_pct_bets_away")])
    ).iloc[0]
    out["spread_pct_bets_home"] = _to_num(
        pd.Series([base.get("spread_pct_bets_home")])
    ).iloc[0]
    out["spread_pct_money_away"] = _to_num(
        pd.Series([base.get("spread_pct_money_away")])
    ).iloc[0]
    out["spread_pct_money_home"] = _to_num(
        pd.Series([base.get("spread_pct_money_home")])
    ).iloc[0]

    # Total
    out["total_pct_bets_over"] = _to_num(
        pd.Series([base.get("total_pct_bets_over")])
    ).iloc[0]
    out["total_pct_bets_under"] = _to_num(
        pd.Series([base.get("total_pct_bets_under")])
    ).iloc[0]
    out["total_pct_money_over"] = _to_num(
        pd.Series([base.get("total_pct_money_over")])
    ).iloc[0]
    out["total_pct_money_under"] = _to_num(
        pd.Series([base.get("total_pct_money_under")])
    ).iloc[0]

    # Moneyline
    out["moneyline_pct_bets_away"] = _to_num(
        pd.Series([base.get("moneyline_pct_bets_away")])
    ).iloc[0]
    out["moneyline_pct_bets_home"] = _to_num(
        pd.Series([base.get("moneyline_pct_bets_home")])
    ).iloc[0]
    out["moneyline_pct_money_away"] = _to_num(
        pd.Series([base.get("moneyline_pct_money_away")])
    ).iloc[0]
    out["moneyline_pct_money_home"] = _to_num(
        pd.Series([base.get("moneyline_pct_money_home")])
    ).iloc[0]

    return out


def yahoo_one_row_per_game(input_df_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    """
    Convert Yahoo per-team rows into one row per game.
    Accepts either a pre-loaded `DataFrame` or a path/filename to a CSV.
    Assumes 2 rows per unique (season, date, matchup).
    """
    # Accept DataFrame or path-like input
    if isinstance(input_df_or_path, pd.DataFrame):
        df = input_df_or_path.copy()
        src_name = "dataframe_input"
    else:
        path = Path(input_df_or_path)
        df = pd.read_csv(path)
        src_name = path.name

    required = [
        "season",
        "date",
        "matchup",
        "team_name",
        "team_abbr",
        "spread",
        "total_side",
        "total_line",
        "money_line",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in yahoo input {src_name}: {missing}"
        )

    # Normalize types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    rows = []
    group_cols = ["season", "date", "matchup"]
    for _, g in df.groupby(group_cols, sort=False):
        rows.append(build_one_game_row_from_yahoo_group(g))

    out = pd.DataFrame(rows)

    # Nice ordering
    first_cols = [
        "season_year",
        "game_date",
        "matchup",
        "team_home",
        "team_away",
        "team_home_abbr",
        "team_away_abbr",
        "spread_home",
        "spread_away",
        "moneyline_home",
        "moneyline_away",
        "total_line",
        "total_pct_bets_over",
        "total_pct_bets_under",
        "total_pct_money_over",
        "total_pct_money_under",
        "spread_pct_bets_away",
        "spread_pct_bets_home",
        "spread_pct_money_away",
        "spread_pct_money_home",
        "moneyline_pct_bets_away",
        "moneyline_pct_bets_home",
        "moneyline_pct_money_away",
        "moneyline_pct_money_home",
    ]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[[c for c in first_cols if c in out.columns] + sorted(other_cols)]
    out.drop_duplicates(inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.drop(columns=["matchup"], inplace=True, errors="ignore")

    return out


# Example:
games = yahoo_one_row_per_game("/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/yahoo_odds/2021/csv/2022-04-10.csv")
games
