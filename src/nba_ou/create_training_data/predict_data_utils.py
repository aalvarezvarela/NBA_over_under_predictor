import pandas as pd

from nba_ou.utils.filter_by_date_range import filter_by_date_range


def normalize_game_ids(game_ids) -> list[str]:
    """Normalize GAME_ID values into unique, non-empty strings."""
    if game_ids is None:
        return []

    normalized = []
    seen = set()
    for game_id in game_ids:
        if pd.isna(game_id):
            continue
        game_id_str = str(game_id).strip()
        if not game_id_str or game_id_str in seen:
            continue
        seen.add(game_id_str)
        normalized.append(game_id_str)
    return normalized


def extract_home_away_pairs_from_scheduled_games(
    scheduled_games: pd.DataFrame | None,
) -> list[tuple[str, str]]:
    """Extract unique (home_team_id, away_team_id) tuples from scheduled games."""
    if scheduled_games is None or scheduled_games.empty:
        return []

    required_cols = {"HOME_TEAM_ID", "VISITOR_TEAM_ID"}
    if not required_cols.issubset(set(scheduled_games.columns)):
        missing = sorted(required_cols - set(scheduled_games.columns))
        print(
            f"Could not derive extra matchup game IDs; missing scheduled columns: {missing}"
        )
        return []

    pairs_df = (
        scheduled_games[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]]
        .dropna(subset=["HOME_TEAM_ID", "VISITOR_TEAM_ID"])
        .drop_duplicates()
    )

    return [
        (str(home_team_id), str(away_team_id))
        for home_team_id, away_team_id in pairs_df.itertuples(index=False, name=None)
    ]


def filter_by_date_range_with_extra_game_ids(
    df: pd.DataFrame,
    older_limit_to_include,
    recent_limit_to_include,
    extra_game_ids=None,
) -> pd.DataFrame:
    """
    Filter by date and keep explicit extra GAME_IDs, capped to most recent date.
    """
    filtered_df = filter_by_date_range(
        df, older_limit_to_include, recent_limit_to_include
    )
    normalized_extra_game_ids = normalize_game_ids(extra_game_ids)
    if not normalized_extra_game_ids or "GAME_ID" not in df.columns:
        return filtered_df

    extra_mask = df["GAME_ID"].astype(str).isin(set(normalized_extra_game_ids))
    if not extra_mask.any():
        return filtered_df

    extra_rows = df.loc[extra_mask].copy()
    # Extra matchup rows may bypass lower bound, but must respect the recent upper bound.
    extra_rows = filter_by_date_range(
        extra_rows, None, most_recent_date_to_include=recent_limit_to_include
    )

    filtered_df = pd.concat([filtered_df, extra_rows], ignore_index=True)
    filtered_df = filtered_df.drop_duplicates(keep="first")
    return filtered_df
