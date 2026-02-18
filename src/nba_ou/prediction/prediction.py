from datetime import datetime
from pathlib import Path
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


def load_and_predict_model_for_nba_games(
    df: pd.DataFrame,
    regressor,
    model_name: str,
    model_type: str,
    model_version: str,
    prediction_datetime: datetime | None = None,
) -> list[pd.DataFrame]:
    """
    Predicts the line error for NBA games using a trained regression model.

    This function processes NBA game data and predicts the difference between
    the actual total score and the over/under line (LINE_ERROR). From this prediction:
    - Positive error → OVER prediction
    - Negative error → UNDER prediction
    - Zero error → PUSH prediction

    Args:
        df: Input DataFrame containing game data with features
        regressor: Trained regressor model that predicts LINE_ERROR.
        model_name: Name of the model being used for predictions
        model_type: Type of the model (e.g., "regression", "classification")
        model_version: Version identifier for the model
        prediction_datetime: Timestamp when predictions are made. If None, uses current time in Europe/Madrid timezone.

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
        verbose=False,
        strict_mode=True,
    )

    # Load the model using joblib
    model = regressor

    # Extract required features from model
    required_features = model.feature_names_in_

    # Ensure all required features are present in the DataFrame
    missing_features = [
        feat for feat in required_features if feat not in df_predictable.columns
    ]
    if missing_features:
        raise ValueError(
            f"Missing required features for prediction: {missing_features}"
        )
    X = df_predictable[required_features]

    # Make predictions - predict LINE_ERROR
    pred_line_error = model.predict(X)

    df_predictable["PRED_LINE_ERROR"] = pred_line_error

    df_predictable["PRED_PICK"] = np.where(
        df_predictable["PRED_LINE_ERROR"] > 0,
        "OVER",
        np.where(df_predictable["PRED_LINE_ERROR"] < 0, "UNDER", "PUSH"),
    )

    # Calculate predicted total points from line error
    df_predictable["PRED_TOTAL_POINTS"] = (
        df_predictable["TOTAL_OVER_UNDER_LINE"] + df_predictable["PRED_LINE_ERROR"]
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
        "TOTAL_OVER_UNDER_LINE",
        "PRED_LINE_ERROR",
        "PRED_TOTAL_POINTS",
        "PRED_PICK",
    ]

    df_summary = df_predictable[summary_columns].copy()

    # Add prediction timestamp and time to match
    # Use provided prediction_datetime or create timezone-aware timestamp in Madrid time
    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))

    df_summary["PREDICTION_DATE"] = prediction_datetime

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
        game_time_aware - df_summary["PREDICTION_DATE"]
    ).dt.total_seconds() / 60
    df_summary["TIME_TO_MATCH_MINUTES"] = (
        df_summary["TIME_TO_MATCH_MINUTES"].round(0).astype(int)
    )

    # Add model information from parameters
    df_summary["MODEL_NAME"] = model_name
    df_summary["MODEL_TYPE"] = model_type
    df_summary["MODEL_VERSION"] = model_version

    # Drop rows with NaN in PRED_LINE_ERROR before saving to database
    df_summary_clean = df_summary.dropna(subset=["PRED_LINE_ERROR"])
    upload_predictions_to_postgre(df_summary_clean)

    return df_summary_clean


def load_s3_model_and_predict(
    *,
    s3_client,
    bucket: str,
    prefix: str,
    df: pd.DataFrame,
    prediction_datetime: datetime | None = None,
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
    model_type = type(regressor).__name__
    model_version = "1.0"

    # Generate predictions
    return load_and_predict_model_for_nba_games(
        df=df,
        regressor=regressor,
        model_name=model_name,
        model_type=model_type,
        model_version=model_version,
        prediction_time=prediction_datetime,
    )
