import pandas as pd
from nba_ou.config.odds_columns import resolve_main_total_line_col


def compute_total_points_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total points and related features.

    - TOTAL_POINTS: sum of PTS per GAME_ID
    - For each totals line column:
        * DIFF_FROM_<linecol>: TOTAL_POINTS - line
        * IS_OVER_<linecol>: indicator(TOTAL_POINTS > line)

    Also keeps the legacy aliases:
      - DIFF_FROM_LINE (vs configured main TOTAL_LINE_<book>)
      - IS_OVER_LINE (vs configured main TOTAL_LINE_<book>)
    """
    # Total points per game
    df["TOTAL_POINTS"] = df.groupby("GAME_ID")["PTS"].transform("sum")

    # Identify all totals line columns
    total_line_cols: list[str] = []
  

    total_line_cols.extend([c for c in df.columns if c.startswith("TOTAL_LINE_")])

    # De-dup while preserving order
    seen = set()
    total_line_cols = [c for c in total_line_cols if not (c in seen or seen.add(c))]

    # Compute features for each line column
    for line_col in total_line_cols:
        # Make a stable suffix for new column names
        suffix = line_col.replace("TOTAL_", "")  # e.g. OVER_UNDER_LINE or LINE_betmgm

        diff_col = f"DIFF_FROM_{suffix}"

        line_vals = pd.to_numeric(df[line_col], errors="coerce")

        df[diff_col] = df["TOTAL_POINTS"] - line_vals    

    # Backward-compatible aliases based on configured main book total line
    main_total_line = resolve_main_total_line_col(df)
    if main_total_line is not None:
        df["DIFF_FROM_LINE"] = df["TOTAL_POINTS"] - df[main_total_line]
        df["IS_OVER_LINE"] = (df["TOTAL_POINTS"] > df[main_total_line]).astype(
            "Int64"
        )  # nullable int

    # Dates
    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"], format="%Y-%m-%d", errors="coerce"
    )

    return df
