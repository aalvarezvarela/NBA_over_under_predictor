# Column name constants for training data selection
TEAM_INFO_COLUMNS = [
    "TEAM_ID",
    "TEAM_CITY",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "MATCHUP",
    "GAME_NUMBER",
]

STATIC_COLUMNS = [
    "SEASON_ID",
    "IS_OVERTIME",
    "GAME_ID",
    "GAME_DATE",
    "SEASON_TYPE",
    "IS_PLAYOFF_GAME",
    "PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY",
    "PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME",
    "SEASON_YEAR",
]

ODDS_COLUMNS = [
    "TOTAL_OVER_UNDER_LINE",
    "SPREAD",
    "MONEYLINE_TEAM_HOME",
    "MONEYLINE_TEAM_AWAY",
]

TARGET_COLUMN = "TOTAL_POINTS"

FORBIDDEN_COLUMNS = ["DIFFERENCE_FROM_LINE", "DIFF_FROM_LINE", "TOTAL_PF", "IS_OVER_LINE"]


def select_training_columns(df_merged, original_columns):
    """
    Select and organize columns for training dataset.

    Args:
        df_merged (pd.DataFrame): Merged home/away DataFrame with all features
        original_columns (list): List of original column names to check against

    Returns:
        pd.DataFrame: DataFrame with selected columns for training

    Raises:
        ValueError: If any disallowed original columns are present in the final training data
    """
    # Generate new list with _HOME and _AWAY appended
    columns_info_before = [f"{col}_TEAM_HOME" for col in TEAM_INFO_COLUMNS] + [
        f"{col}_TEAM_AWAY" for col in TEAM_INFO_COLUMNS
    ]

    columns_info_before.extend(STATIC_COLUMNS)

    # Insert columns that have BEFORE in the name
    columns_info_before.extend([col for col in df_merged.columns if "BEFORE" in col])

    columns_info_before.extend(ODDS_COLUMNS)

    df_training = df_merged[columns_info_before + [TARGET_COLUMN]].copy()
    # Drop any forbidden columns if they exist
    columns_to_drop = [col for col in FORBIDDEN_COLUMNS if col in df_training.columns]
    if columns_to_drop:
        df_training = df_training.drop(columns=columns_to_drop)

    # Safety check: Ensure no disallowed original columns are present
    # Allowed columns include: static columns, target, odds, and team info with suffixes
    allowed_columns = set(STATIC_COLUMNS + [TARGET_COLUMN] + ODDS_COLUMNS)

    # Add team info columns with HOME/AWAY suffixes
    for col in TEAM_INFO_COLUMNS:
        allowed_columns.add(f"{col}_TEAM_HOME")
        allowed_columns.add(f"{col}_TEAM_AWAY")

    disallowed_columns = []

    for col in df_training.columns:
        # Check if this column is in original columns but not in allowed list
        # Skip columns with BEFORE (they are temporal features)
        if (
            "BEFORE" not in col
            and col not in allowed_columns
            and col in original_columns
        ):
            disallowed_columns.append(col)

    if disallowed_columns:
        raise ValueError(
            f"Disallowed original columns found in training data: {disallowed_columns}. "
            "These columns should have '_BEFORE' suffix or be excluded."
        )

    return df_training
