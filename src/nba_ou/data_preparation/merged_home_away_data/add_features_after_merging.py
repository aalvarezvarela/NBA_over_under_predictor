import numpy as np
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


def add_high_value_features_for_team_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of high-value, numeric-only engineered features for predicting
    TEAM points (works well with XGBoost).

    This function is defensive:
      - It only creates features when the required source columns exist.
      - It never uses player NAME columns.
      - It avoids target leakage (does not use same-game points).
      - It ONLY adds features that are not trivially redundant with existing columns.

    Returns a COPY of df with new columns appended.
    """
    out = df.copy()

    def _has(cols):
        return all(c in out.columns for c in cols)

    def _safe_div(num, den):
        den = den.replace(0, np.nan)
        return num / den

    def _add(name, series):
        # Add only if it does not already exist (avoid overwriting any preexisting column).
        if name in out.columns:
            return
        out[name] = pd.to_numeric(series, errors="coerce")

    # ---------------------------------------------------------------------
    # 1) Market-derived: implied team totals (strong, non-leaky)
    # ---------------------------------------------------------------------
    # Uses game-level TOTAL line and SPREAD (assumes SPREAD is from HOME perspective: home - away).
    # If your SPREAD is opposite, flip the sign.
    if _has(["TOTAL_OVER_UNDER_LINE", "SPREAD"]):
        _add(
            "IMPLIED_PTS_HOME",
            (out["TOTAL_OVER_UNDER_LINE"] / 2.0) - (out["SPREAD"] / 2.0),
        )
        _add(
            "IMPLIED_PTS_AWAY",
            (out["TOTAL_OVER_UNDER_LINE"] / 2.0) + (out["SPREAD"] / 2.0),
        )
        # NOTE: Do NOT add implied sum/diff checks: they are algebraic restatements of
        # TOTAL_OVER_UNDER_LINE and SPREAD, hence redundant.

    # ---------------------------------------------------------------------
    # 2) Offense-vs-defense interaction (often better than raw ratings)
    # ---------------------------------------------------------------------
    # Possession proxy: average of both teams' pace.
    if _has(
        [
            "PACE_PER40_SEASON_BEFORE_AVG_TEAM_HOME",
            "PACE_PER40_SEASON_BEFORE_AVG_TEAM_AWAY",
        ]
    ):
        exp_poss = (
            out["PACE_PER40_SEASON_BEFORE_AVG_TEAM_HOME"]
            + out["PACE_PER40_SEASON_BEFORE_AVG_TEAM_AWAY"]
        ) / 2.0
        _add("EXPECTED_POSS_FROM_PACE", exp_poss)

        # Expected points proxy: possessions * offensive rating.
        if _has(["OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"]):
            _add(
                "EXPECTED_PTS_HOME_FROM_OFFR_PACE",
                exp_poss * (out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"] / 100.0),
            )
        if _has(["OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"]):
            _add(
                "EXPECTED_PTS_AWAY_FROM_OFFR_PACE",
                exp_poss * (out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"] / 100.0),
            )

    # Off-Def mismatch features (home offense vs away defense, and vice versa)
    if _has(
        [
            "OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME",
            "DEF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
        ]
    ):
        _add(
            "OFFDEF_MISMATCH_HOME_OFF_MINUS_AWAY_DEF",
            out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"]
            - out["DEF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"],
        )
    if _has(
        [
            "OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
            "DEF_RATING_SEASON_BEFORE_AVG_TEAM_HOME",
        ]
    ):
        _add(
            "OFFDEF_MISMATCH_AWAY_OFF_MINUS_HOME_DEF",
            out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"]
            - out["DEF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"],
        )

    # ---------------------------------------------------------------------
    # 3) Form and volatility: standardized recent scoring vs season baseline
    # ---------------------------------------------------------------------
    # z = (recent_avg - season_avg) / season_std
    if _has(
        [
            "PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME",
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME",
            "PTS_SEASON_BEFORE_STD_TEAM_HOME",
        ]
    ):
        z_home = _safe_div(
            out["PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"]
            - out["PTS_SEASON_BEFORE_AVG_TEAM_HOME"],
            out["PTS_SEASON_BEFORE_STD_TEAM_HOME"],
        )
        _add("PTS_FORM_Z_HOME_LAST5_VS_SEASON", z_home)

    if _has(
        [
            "PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY",
            "PTS_SEASON_BEFORE_AVG_TEAM_AWAY",
            "PTS_SEASON_BEFORE_STD_TEAM_AWAY",
        ]
    ):
        z_away = _safe_div(
            out["PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY"]
            - out["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"],
            out["PTS_SEASON_BEFORE_STD_TEAM_AWAY"],
        )
        _add("PTS_FORM_Z_AWAY_LAST5_VS_SEASON", z_away)

    # Trend slopes already exist for last 5 games; compress them into combined deltas
    if _has(
        [
            "PTS_TEAM_HOME_TREND_SLOPE_LAST_5_GAMES_BEFORE",
            "PTS_TEAM_AWAY_TREND_SLOPE_LAST_5_GAMES_BEFORE",
        ]
    ):
        _add(
            "PTS_TREND_SLOPE_DIFF_HOME_MINUS_AWAY",
            out["PTS_TEAM_HOME_TREND_SLOPE_LAST_5_GAMES_BEFORE"]
            - out["PTS_TEAM_AWAY_TREND_SLOPE_LAST_5_GAMES_BEFORE"],
        )
        _add(
            "PTS_TREND_SLOPE_SUM_HOME_PLUS_AWAY",
            out["PTS_TEAM_HOME_TREND_SLOPE_LAST_5_GAMES_BEFORE"]
            + out["PTS_TEAM_AWAY_TREND_SLOPE_LAST_5_GAMES_BEFORE"],
        )

    # ---------------------------------------------------------------------
    # 4) Fatigue and travel compression
    # ---------------------------------------------------------------------
    # NOTE: Do NOT add IS_BACK_TO_BACK because BACK_TO_BACK already exists (redundant).

    # Travel intensity: concentration of travel in last 2 vs 14 days
    if _has(
        ["TOTAL_KM_IN_LAST_2_DAYS_HOME_TEAM", "TOTAL_KM_IN_LAST_14_DAYS_HOME_TEAM"]
    ):
        ratio = _safe_div(
            out["TOTAL_KM_IN_LAST_2_DAYS_HOME_TEAM"],
            out["TOTAL_KM_IN_LAST_14_DAYS_HOME_TEAM"],
        )
        _add("TRAVEL_RECENCY_RATIO_HOME_2D_OVER_14D", ratio)

    if _has(
        ["TOTAL_KM_IN_LAST_2_DAYS_AWAY_TEAM", "TOTAL_KM_IN_LAST_14_DAYS_AWAY_TEAM"]
    ):
        ratio = _safe_div(
            out["TOTAL_KM_IN_LAST_2_DAYS_AWAY_TEAM"],
            out["TOTAL_KM_IN_LAST_14_DAYS_AWAY_TEAM"],
        )
        _add("TRAVEL_RECENCY_RATIO_AWAY_2D_OVER_14D", ratio)

    # Rest mismatch
    if _has(["REST_DAYS_BEFORE_MATCH_TEAM_HOME", "REST_DAYS_BEFORE_MATCH_TEAM_AWAY"]):
        _add(
            "REST_DAYS_DIFF_HOME_MINUS_AWAY",
            out["REST_DAYS_BEFORE_MATCH_TEAM_HOME"]
            - out["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"],
        )

    # ---------------------------------------------------------------------
    # 5) Injury impact compression (numeric-only)
    # ---------------------------------------------------------------------
    if _has(
        ["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME", "PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
    ):
        _add(
            "INJURY_PTS_SHARE_HOME",
            _safe_div(
                out["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"],
                out["PTS_SEASON_BEFORE_AVG_TEAM_HOME"],
            ),
        )

    if _has(
        ["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY", "PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    ):
        _add(
            "INJURY_PTS_SHARE_AWAY",
            _safe_div(
                out["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY"],
                out["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"],
            ),
        )

    if _has(
        ["STAR_PTS_PERCENTAGE_BEFORE_TEAM_HOME", "STAR_PTS_PERCENTAGE_BEFORE_TEAM_AWAY"]
    ):
        _add(
            "STAR_PTS_PCT_DIFF_HOME_MINUS_AWAY",
            out["STAR_PTS_PERCENTAGE_BEFORE_TEAM_HOME"]
            - out["STAR_PTS_PERCENTAGE_BEFORE_TEAM_AWAY"],
        )

    # ---------------------------------------------------------------------
    # 6) Efficiency decomposition
    # ---------------------------------------------------------------------
    if _has(["POSS_SEASON_BEFORE_AVG_TEAM_HOME", "TS_PCT_SEASON_BEFORE_AVG_TEAM_HOME"]):
        _add(
            "POSS_X_TSPCT_HOME",
            out["POSS_SEASON_BEFORE_AVG_TEAM_HOME"]
            * out["TS_PCT_SEASON_BEFORE_AVG_TEAM_HOME"],
        )
    if _has(["POSS_SEASON_BEFORE_AVG_TEAM_AWAY", "TS_PCT_SEASON_BEFORE_AVG_TEAM_AWAY"]):
        _add(
            "POSS_X_TSPCT_AWAY",
            out["POSS_SEASON_BEFORE_AVG_TEAM_AWAY"]
            * out["TS_PCT_SEASON_BEFORE_AVG_TEAM_AWAY"],
        )

    # ---------------------------------------------------------------------
    # Final: replace inf with nan, keep NaNs (XGBoost can handle missing)
    # ---------------------------------------------------------------------
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out
