import pandas as pd


def apply_final_transformations(df_training):
    """
    Apply final transformations to the training dataset.

    Args:
        df_training (pd.DataFrame): DataFrame with derived features

    Returns:
        pd.DataFrame: Final training-ready DataFrame
    """
    df_training.loc[df_training["MATCHUP_TEAM_HOME"] == 0, "MATCHUP_TEAM_HOME"] = (
        df_training["TEAM_ABBREVIATION_TEAM_HOME"]
        + " vs. "
        + df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
    )

    df_training.loc[df_training["MATCHUP_TEAM_AWAY"] == 0, "MATCHUP_TEAM_AWAY"] = (
        df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
        + " @ "
        + df_training["TEAM_ABBREVIATION_TEAM_HOME"]
    )

    df_training.drop(columns=["IS_OVERTIME"], inplace=True)

    # Move TOTAL_OVER_UNDER_LINE to first column
    first_col = "TOTAL_OVER_UNDER_LINE"
    df_training = df_training[
        [first_col] + [col for col in df_training.columns if col != first_col]
    ]

    return df_training


def add_derived_features_after_computed_stats(df_training):
    """
    Add derived features to the training dataset.

    Args:
        df_training (pd.DataFrame): DataFrame with selected training columns

    Returns:
        pd.DataFrame: DataFrame with additional derived features
    """
    df_training["TOTAL_PTS_SEASON_BEFORE_AVG"] = (
        df_training["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
        + df_training["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    )

    df_training["TOTAL_PTS_LAST_GAMES_AVG"] = (
        df_training["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        + df_training["PTS_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )

    df_training["BACK_TO_BACK"] = (
        df_training["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"] == 1
    ) & (df_training["REST_DAYS_BEFORE_MATCH_TEAM_HOME"] == 1)

    df_training["DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE_MATCH"] = (
        df_training["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
        - df_training["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
    )
    df_training["DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE_MATCH"] = (
        df_training["OFF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_AWAY"]
        - df_training["DEF_RATING_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME"]
    )
    df_training = apply_final_transformations(df_training)

    return df_training
