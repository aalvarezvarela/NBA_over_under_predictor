import pandas as pd
from nba_ou.postgre_db.odds.merge_odds_data import (
    merge_yahoo_sportsbook_odds,
)


def merge_and_validate_scheduled_odds(
    df_odds: pd.DataFrame,
    df_odds_yahoo: pd.DataFrame,
    df_odds_sportsbook: pd.DataFrame,
    strict_mode: int = 0,
) -> pd.DataFrame:
    """Merge and validate scheduled odds with historical odds data.

    This function merges Yahoo and Sportsbook odds for scheduled games, validates
    column consistency, and optionally checks for null values in strict mode.

    Args:
        df_odds (pd.DataFrame): Historical odds data from database
        df_odds_yahoo (pd.DataFrame): Yahoo odds for scheduled games
        df_odds_sportsbook (pd.DataFrame): Sportsbook odds for scheduled games
        strict_mode (int, optional): Maximum number of columns allowed to have NaN/None values.
            Use 0 for no columns with nulls allowed, -1 or any negative value to disable the check. Default is 0.

    Returns:
        pd.DataFrame: Combined odds dataframe with historical and scheduled games

    Raises:
        ValueError: If column validation fails or if number of columns with nulls exceeds strict_mode threshold
    """
    # Merge Yahoo and Sportsbook scheduled odds
    df_odds_predict = merge_yahoo_sportsbook_odds(df_odds_yahoo, df_odds_sportsbook)

    # Validate columns
    df_odds_cols = set(df_odds.columns)
    df_odds_predict_cols = set(df_odds_predict.columns)

    # Check if df_odds has columns that df_odds_predict doesn't have
    missing_in_predict = df_odds_cols - df_odds_predict_cols
    if missing_in_predict:
        raise ValueError(
            f"df_odds_predict is missing columns that exist in df_odds: {missing_in_predict}"
        )

    # Drop extra columns from df_odds_predict that are not in df_odds
    extra_in_predict = df_odds_predict_cols - df_odds_cols
    if extra_in_predict:
        print(f"Dropping extra columns from df_odds_predict: {extra_in_predict}")
        df_odds_predict = df_odds_predict.drop(columns=list(extra_in_predict))

    # Strict mode: check for NaN or None values
    if strict_mode >= 0:
        null_counts = df_odds_predict.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        num_cols_with_nulls = len(cols_with_nulls)

        if num_cols_with_nulls > strict_mode:
            raise ValueError(
                f"Strict mode: Found {num_cols_with_nulls} columns with NaN/None values, but only {strict_mode} allowed:\n{cols_with_nulls}"
            )

    # Concatenate and sort
    df_odds_combined = pd.concat([df_odds, df_odds_predict], ignore_index=True)
    df_odds_combined.sort_values(by="game_date", inplace=True, ascending=False)
    df_odds_combined.reset_index(drop=True, inplace=True)

    return df_odds_combined
