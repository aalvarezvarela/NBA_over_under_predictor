from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nba_ou.data_preparation.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from nba_ou.postgre_db.predictions.create.create_nba_predictions_db import (
    upload_predictions_to_postgre,
)
from nba_ou.utils.s3_models import (
    get_first_joblib_from_prefix,
    load_joblib_from_bytes,
    read_s3_object_bytes,
)

PredictionTarget = Literal["PRED_LINE_ERROR", "TOTAL_POINTS"]

PREDICTION_TARGET_LINE_ERROR: PredictionTarget = "PRED_LINE_ERROR"
PREDICTION_TARGET_TOTAL_POINTS: PredictionTarget = "TOTAL_POINTS"
DEFAULT_TOTAL_POINTS_PICK_LINE_COL = "total_bet365_line_over"
PREDICTION_VALUE_TYPE_TOTAL_POINTS = "TOTAL_POINTS"
PREDICTION_VALUE_TYPE_DIFF_FROM_LINE = "DIFF_FROM_LINE"


def _add_na_tracking_columns(
    df: pd.DataFrame,
    *,
    count_col: str = "NA_COLUMNS_COUNT",
    names_col: str = "NA_COLUMNS_NAMES",
) -> pd.DataFrame:
    """Add per-row NA count and NA column-name list.

    Excludes TOTAL_POINTS and columns containing 'MATCHUP' (case insensitive) from tracking.
    """
    out = df.copy()

    # Filter columns: exclude TOTAL_POINTS and any column containing 'MATCHUP'
    cols_to_check = [
        col
        for col in out.columns
        if col != "TOTAL_POINTS" and "matchup" not in col.lower()
    ]

    na_mask = out[cols_to_check].isna()
    col_names = na_mask.columns.to_numpy()
    out[count_col] = na_mask.sum(axis=1).astype(int)
    out[names_col] = [
        ",".join(col_names[row_mask]) if row_mask.any() else None
        for row_mask in na_mask.to_numpy()
    ]
    return out


def _resolve_column_name(df: pd.DataFrame, desired_column: str) -> str | None:
    """Resolve column name with case-insensitive fallback."""
    if desired_column in df.columns:
        return desired_column

    desired_lower = desired_column.lower()
    for col in df.columns:
        if col.lower() == desired_lower:
            return col
    return None


def _prepare_required_features_for_prediction(
    df: pd.DataFrame,
    required_features: list[str],
    *,
    mandatory_bet365_col: str = DEFAULT_TOTAL_POINTS_PICK_LINE_COL,
) -> tuple[list[str], list[str]]:
    """Ensure required model features exist using bet365 fallback when possible."""
    normalized_required = [str(feat) for feat in required_features]

    bet365_col = _resolve_column_name(df, mandatory_bet365_col)
    if bet365_col is None:
        raise ValueError(
            f"Column '{mandatory_bet365_col}' is mandatory for prediction."
        )

    created_as_nan: list[str] = []

    for feat in normalized_required:
        if feat in df.columns:
            continue

        source_col = None
        feat_lower = feat.lower()

        if "consensus_opener" in feat_lower:
            bet365_equivalent = feat_lower.replace("consensus_opener", "bet365")
            source_col = _resolve_column_name(df, bet365_equivalent)
            if source_col is None and "total_consensus_opener_line_over" in feat_lower:
                source_col = bet365_col

        if source_col is not None:
            df[feat] = pd.to_numeric(df[source_col], errors="coerce")
            continue

        df[feat] = np.nan
        created_as_nan.append(feat)

    return normalized_required, created_as_nan


def load_and_predict_model_for_nba_games(
    df: pd.DataFrame,
    regressor,
    model_name: str,
    model_type: str,
    model_version: str,
    prediction_datetime: datetime | None = None,
    prediction_target: PredictionTarget = PREDICTION_TARGET_LINE_ERROR,
    total_points_pick_line_col: str = DEFAULT_TOTAL_POINTS_PICK_LINE_COL,
) -> pd.DataFrame:
    """
    Predict NBA games using a trained regressor for one of two targets:
    - PRED_LINE_ERROR: model predicts line error directly
    - TOTAL_POINTS: model predicts total points directly

    PRED_PICK behavior:
    - PRED_LINE_ERROR mode: OVER if prediction > 0, UNDER if < 0, PUSH if == 0
    - TOTAL_POINTS mode: compares predicted total points against the configured line column

    Args:
        df: Input DataFrame containing game data with features
        regressor: Trained regressor model.
        model_name: Name of the model being used for predictions
        model_type: Type of the model (e.g., "regression", "classification")
        model_version: Version identifier for the model
        prediction_datetime: Timestamp when predictions are made. If None, uses current time in Europe/Madrid timezone.
        prediction_target: Which target this model predicts.
        total_points_pick_line_col: Line column used to derive OVER/UNDER/PUSH when
            prediction_target is TOTAL_POINTS.

    Returns:
        DataFrame containing predictions with key information
    """

    # Step 1: Clean the DataFrame for prediction
    df_predictable = clean_dataframe_for_training(
        df,
        nan_threshold=100,
        drop_all_na_rows=False,
        keep_columns=[
            "GAME_ID",
            "SEASON_TYPE",
            "GAME_DATE",
            "GAME_TIME",
            "TEAM_NAME_TEAM_HOME",
            "TEAM_NAME_TEAM_AWAY",
            "MATCHUP_TEAM_HOME",
        ],
        keep_all_cols=True,
        verbose=1,
        strict_mode=2,
    )

    df_predictable = _add_na_tracking_columns(df_predictable)

    # Load the model using joblib
    model = regressor

    # Extract required features from model
    required_features_raw = model.feature_names_in_
    required_features = [str(feat) for feat in required_features_raw]

    # Handle IS_TRAINING_DATA if it's missing - create it with all False values
    if (
        "IS_TRAINING_DATA" in required_features
        and "IS_TRAINING_DATA" not in df_predictable
    ):
        print(
            "⚠️  Warning: 'IS_TRAINING_DATA' column is missing. Creating it with default False values for prediction."
        )
        df_predictable["IS_TRAINING_DATA"] = False

    required_features, created_nan_features = _prepare_required_features_for_prediction(
        df_predictable,
        required_features,
        mandatory_bet365_col=DEFAULT_TOTAL_POINTS_PICK_LINE_COL,
    )
    if created_nan_features:
        print(
            "⚠️  Warning: Missing model features were auto-created as NaN "
            f"({len(created_nan_features)} columns)."
        )
    X = df_predictable[required_features]

    # Make predictions
    raw_predictions = model.predict(X)
    prediction_values = pd.to_numeric(
        pd.Series(raw_predictions, index=df_predictable.index), errors="coerce"
    )
    pick_line: pd.Series | None = None

    if prediction_target == PREDICTION_TARGET_LINE_ERROR:
        df_predictable["PRED_LINE_ERROR"] = prediction_values
        line_col = _resolve_column_name(df_predictable, "TOTAL_OVER_UNDER_LINE")
        if line_col:
            df_predictable["PRED_TOTAL_POINTS"] = pd.to_numeric(
                df_predictable[line_col], errors="coerce"
            ) + df_predictable["PRED_LINE_ERROR"]
        else:
            df_predictable["PRED_TOTAL_POINTS"] = np.nan

    elif prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        df_predictable["PRED_TOTAL_POINTS"] = prediction_values

        pick_line_col = _resolve_column_name(df_predictable, total_points_pick_line_col)
        if pick_line_col is None:
            raise ValueError(
                f"Column '{total_points_pick_line_col}' is required to compute "
                f"PRED_LINE_ERROR/PRED_PICK when "
                f"prediction_target={PREDICTION_TARGET_TOTAL_POINTS}."
            )
        pick_line = pd.to_numeric(df_predictable[pick_line_col], errors="coerce")
        df_predictable["PRED_LINE_ERROR"] = df_predictable["PRED_TOTAL_POINTS"] - pick_line

    else:
        raise ValueError(
            "prediction_target must be one of: "
            f"{PREDICTION_TARGET_LINE_ERROR}, {PREDICTION_TARGET_TOTAL_POINTS}"
        )

    if prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        if pick_line is None:
            pick_line_col = _resolve_column_name(df_predictable, total_points_pick_line_col)
            if pick_line_col is None:
                raise ValueError(
                    f"Column '{total_points_pick_line_col}' is required to compute PRED_PICK "
                    f"when prediction_target={PREDICTION_TARGET_TOTAL_POINTS}."
                )
            pick_line = pd.to_numeric(df_predictable[pick_line_col], errors="coerce")
        pred_total = pd.to_numeric(df_predictable["PRED_TOTAL_POINTS"], errors="coerce")
        df_predictable["PRED_PICK"] = np.select(
            [
                pred_total > pick_line,
                pred_total < pick_line,
                (pred_total == pick_line) & pred_total.notna() & pick_line.notna(),
            ],
            ["OVER", "UNDER", "PUSH"],
            default=None,
        )
    else:
        pred_line_error = pd.to_numeric(df_predictable["PRED_LINE_ERROR"], errors="coerce")
        df_predictable["PRED_PICK"] = np.select(
            [
                pred_line_error > 0,
                pred_line_error < 0,
                pred_line_error == 0,
            ],
            ["OVER", "UNDER", "PUSH"],
            default=None,
        )

    bet365_line_col = _resolve_column_name(
        df_predictable, DEFAULT_TOTAL_POINTS_PICK_LINE_COL
    )
    if bet365_line_col is None:
        raise ValueError(
            f"Column '{DEFAULT_TOTAL_POINTS_PICK_LINE_COL}' is mandatory for prediction."
        )
    df_predictable["TOTAL_BET365_LINE_AT_PREDICTION"] = pd.to_numeric(
        df_predictable[bet365_line_col], errors="coerce"
    )
    df_predictable["PREDICTION_VALUE_TYPE"] = (
        PREDICTION_VALUE_TYPE_TOTAL_POINTS
        if prediction_target == PREDICTION_TARGET_TOTAL_POINTS
        else PREDICTION_VALUE_TYPE_DIFF_FROM_LINE
    )

    df_predictable.rename(columns={"MATCHUP_TEAM_HOME": "MATCHUP"}, inplace=True)
    df_predictable["GAME_DATE"] = (
        df_predictable["GAME_DATE"].astype(str).str.split("T").str[0]
    )

    # Sheet 1: Summary DataFrame
    summary_columns = [
        "GAME_ID",
        "SEASON_TYPE",
        "GAME_DATE",
        "GAME_TIME",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "PREDICTION_VALUE_TYPE",
        "TOTAL_OVER_UNDER_LINE",
        "TOTAL_BET365_LINE_AT_PREDICTION",
        "PRED_LINE_ERROR",
        "PRED_TOTAL_POINTS",
        "PRED_PICK",
        "NA_COLUMNS_COUNT",
        "NA_COLUMNS_NAMES",
    ]

    df_summary = df_predictable[summary_columns].copy()

    # Add prediction timestamp and time to match
    # Use provided prediction_datetime or create timezone-aware timestamp in Madrid time
    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))

    df_summary["PREDICTION_DATETIME"] = prediction_datetime
    df_summary["PREDICTION_DATE"] = prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Add score columns (will be None for predictions, filled in later with actual results)
    df_summary["HOME_PTS"] = None
    df_summary["AWAY_PTS"] = None

    # Calculate time to match in minutes
    def ensure_timezone_aware(dt_value):
        """Convert datetime to timezone-aware (US/Pacific) if naive."""
        if pd.isna(dt_value):
            return pd.NaT

        # Convert to datetime if it's not already
        if not isinstance(dt_value, (pd.Timestamp, datetime)):
            try:
                dt_value = pd.to_datetime(dt_value)
            except Exception:
                return pd.NaT

        # Check if timezone-aware
        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo is None:
            # Naive datetime, localize to US/Pacific
            return dt_value.tz_localize("US/Pacific")
        elif hasattr(dt_value, "tzinfo") and dt_value.tzinfo is not None:
            # Already timezone-aware
            return dt_value
        else:
            # Fallback: try to convert and localize
            return pd.to_datetime(dt_value).tz_localize("US/Pacific")

    game_time_aware = df_summary["GAME_TIME"].apply(ensure_timezone_aware)

    df_summary["TIME_TO_MATCH_MINUTES"] = (
        game_time_aware - df_summary["PREDICTION_DATETIME"]
    ).dt.total_seconds() / 60
    df_summary["TIME_TO_MATCH_MINUTES"] = (
        df_summary["TIME_TO_MATCH_MINUTES"].round(0).astype(int)
    )

    # Add model information from parameters
    df_summary["MODEL_NAME"] = model_name
    df_summary["MODEL_TYPE"] = model_type
    df_summary["MODEL_VERSION"] = model_version

    # Drop rows with NaN in the model's predicted target before saving to database
    if prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        df_summary_clean = df_summary.dropna(subset=["PRED_TOTAL_POINTS"])
    else:
        df_summary_clean = df_summary.dropna(subset=["PRED_LINE_ERROR"])
    upload_predictions_to_postgre(df_summary_clean)

    return df_summary_clean


def load_s3_model_and_predict(
    *,
    s3_client,
    bucket: str,
    prefix: str,
    df: pd.DataFrame,
    model_id: str,
    prediction_datetime: datetime | None = None,
    prediction_target: PredictionTarget = PREDICTION_TARGET_LINE_ERROR,
    total_points_pick_line_col: str = DEFAULT_TOTAL_POINTS_PICK_LINE_COL,
) -> pd.DataFrame:
    """
    Load the first .joblib model from an S3 prefix and generate predictions.

    This helper function reduces code duplication by combining:
    1. Finding the first .joblib file in an S3 prefix
    2. Loading the model from S3
    3. Generating predictions

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix to search for model
        df: DataFrame containing game data to predict
        prediction_datetime: Timestamp for predictions (defaults to now in Europe/Madrid)
        prediction_target: Which target this model predicts.
        total_points_pick_line_col: Line column used for PRED_PICK when
            prediction_target is TOTAL_POINTS.

    Returns:
        DataFrame containing predictions

    Raises:
        FileNotFoundError: If no .joblib file is found in the prefix
    """
    # Find the first .joblib file in the prefix
    model_key = get_first_joblib_from_prefix(
        s3_client=s3_client,
        bucket=bucket,
        prefix=prefix,
    )

    if not model_key:
        raise FileNotFoundError(f"No .joblib file found in S3 prefix: {prefix}")

    # Load the model from S3
    model_bytes = read_s3_object_bytes(
        s3_client=s3_client,
        bucket=bucket,
        key=model_key,
    )
    regressor = load_joblib_from_bytes(model_bytes)

    # Extract model metadata from the S3 key
    model_name = Path(model_key).stem
    model_type = model_id
    # Extract version from model name (assumes format like: model_name_DD_MM_YY)
    # e.g., "recent_data_xgb_line_error_14_02_26" -> "14_02_26"
    name_parts = model_name.split("_")
    if len(name_parts) >= 3 and all(part.isdigit() for part in name_parts[-3:]):
        model_version = "_".join(name_parts[-3:])  # Last 3 parts as date
    else:
        model_version = "1.0"  # Fallback if no date pattern found

    # Generate predictions
    return load_and_predict_model_for_nba_games(
        df=df,
        regressor=regressor,
        model_name=model_name,
        model_type=model_type,
        model_version=model_version,
        prediction_datetime=prediction_datetime,
        prediction_target=prediction_target,
        total_points_pick_line_col=total_points_pick_line_col,
    )
