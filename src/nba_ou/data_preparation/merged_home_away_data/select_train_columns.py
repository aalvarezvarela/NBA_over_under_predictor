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
    # Bet365 odds (already merged individually per team)
    "TOTAL_OVER_UNDER_LINE",
    "SPREAD",
    "MONEYLINE_TEAM_HOME",
    "MONEYLINE_TEAM_AWAY",
    # Yahoo odds columns
    "spread_home",
    "spread_away",
    "moneyline_home",
    "moneyline_away",
    # "total_line", # This column is dropped after filling BetMGM missing values, so we won't have it in the final merged df
    # Yahoo percentage columns (betting behavior data)
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
    # Consensus data from sportsbooks
    "total_consensus_pct_over",
    "total_consensus_pct_under",
    "spread_consensus_pct_away",
    "spread_consensus_pct_home",
    "spread_consensus_opener_line_away",
    "spread_consensus_opener_line_home",
    "spread_consensus_opener_price_away",
    "spread_consensus_opener_price_home",
    "total_consensus_opener_line_over",
    "total_consensus_opener_price_over",
    "total_consensus_opener_line_under",
    "total_consensus_opener_price_under",
    "ml_consensus_opener_price_away",
    "ml_consensus_opener_price_home",
]

# Add all sportsbook-specific columns dynamically
# Common sportsbooks in the data
SPORTSBOOKS = [
    "pinnacle",
    "betmgm",
    "bet365",
    "caesars",
    "fanduel",
    "fanatics",
    "draftkings",
    "mybookie",
    "bovada",
    "betonline",
    "intertops",
    "heritage",
    "bookmaker",
    "lowvig",
    "betcris",
    "justbet",
    "sportsbetting",
    "gtbets",
    "consensus",
    "fanatics_sportsbook",
]

# Add total line and price columns for each book
for book in SPORTSBOOKS:
    if book != "consensus":  # consensus is handled separately above
        ODDS_COLUMNS.extend(
            [
                f"total_{book}_line_over",
                f"total_{book}_line_under",
                f"total_{book}_price_over",
                f"total_{book}_price_under",
            ]
        )

# Add spread columns for each book
for book in SPORTSBOOKS:
    if book not in [
        "consensus",
        "consensus_opener",
    ]:  # consensus variants handled separately
        ODDS_COLUMNS.extend(
            [
                f"spread_{book}_line_away",
                f"spread_{book}_line_home",
                f"spread_{book}_price_away",
                f"spread_{book}_price_home",
            ]
        )

# Add moneyline columns for each book
for book in SPORTSBOOKS:
    ODDS_COLUMNS.extend(
        [
            f"ml_{book}_price_away",
            f"ml_{book}_price_home",
        ]
    )

TARGET_COLUMN = "TOTAL_POINTS"

FORBIDDEN_COLUMNS = [
    "DIFFERENCE_FROM_LINE",
    "DIFF_FROM_LINE",
    "TOTAL_PF",
    "IS_OVER_LINE",
]


def select_training_columns(df_merged, original_columns, debug=True):
    """
    Select and organize columns for training dataset.

    Args:
        df_merged (pd.DataFrame): Merged home/away DataFrame with all features
        original_columns (list): List of original column names to check against
        debug (bool): If True, print information about deleted columns

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

    # Add odds columns that exist in the dataframe (permit but don't require)
    odds_cols_present = [col for col in ODDS_COLUMNS if col in df_merged.columns]
    columns_info_before.extend(odds_cols_present)

    # Add any columns that start with "odds_" prefix
    columns_info_before.extend(
        [col for col in df_merged.columns if col.startswith("odds_")]
    )

    # Add any columns that start with "TOTAL_LINE_" prefix
    columns_info_before.extend(
        [col for col in df_merged.columns if col.startswith("TOTAL_LINE_")]
    )

    # Filter to only include columns that actually exist in df_merged
    columns_to_select = [col for col in columns_info_before if col in df_merged.columns]

    # Add target column if it exists
    if TARGET_COLUMN in df_merged.columns:
        columns_to_select.append(TARGET_COLUMN)

    if debug:
        excluded_columns = [
            col for col in df_merged.columns if col not in columns_to_select
        ]
        if excluded_columns:
            print(f"Debug: {len(excluded_columns)} columns not selected for training:")
            for col in excluded_columns:
                print(f"  - {col}")

    df_training = df_merged[columns_to_select].copy()
    # Drop any forbidden columns if they exist
    columns_to_drop = [col for col in FORBIDDEN_COLUMNS if col in df_training.columns]
    # add to columns to discard anything that contains "DIFF_FROM" (unless it also contains "BEFORE")
    columns_to_drop.extend(
        [
            col
            for col in df_training.columns
            if "DIFF_FROM" in col and "_BEFORE" not in col
        ]
    )

    if columns_to_drop:
        if debug:
            print(f"Debug: Dropping {len(columns_to_drop)} forbidden columns:")
            for col in columns_to_drop:
                print(f"  - {col}")
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
        # Skip columns that start with "odds_" (they are odds features)
        if (
            "_BEFORE" not in col
            and not col.startswith("odds_")
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
