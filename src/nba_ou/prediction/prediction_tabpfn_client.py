import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nba_ou.config.settings import SETTINGS
from nba_ou.data_preparation.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from nba_ou.postgre_db.predictions.create.create_nba_predictions_db import (
    upload_predictions_to_postgre,
)
from nba_ou.utils.s3_models import (
    list_s3_objects,
    load_parquet_from_bytes,
    make_s3_client,
    read_s3_object_bytes,
)


def _add_na_tracking_columns(
    df: pd.DataFrame,
    *,
    count_col: str = "NA_COLUMNS_COUNT",
    names_col: str = "NA_COLUMNS_NAMES",
) -> pd.DataFrame:
    """Add per-row NA count and NA column-name list."""
    out = df.copy()
    na_mask = out.isna()
    col_names = na_mask.columns.to_numpy()
    out[count_col] = na_mask.sum(axis=1).astype(int)
    out[names_col] = [
        ",".join(col_names[row_mask]) if row_mask.any() else None
        for row_mask in na_mask.to_numpy()
    ]
    return out


def _select_latest_historical_train_key(s3_client, bucket: str, prefix: str) -> str:
    """Select the latest historical parquet by YYYYMMDD suffix in key name."""
    objects = list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix)
    parquet_objects = [
        obj for obj in objects if obj.get("Key", "").endswith(".parquet")
    ]
    if not parquet_objects:
        raise FileNotFoundError(f"No parquet files found in s3://{bucket}/{prefix}")

    pattern = re.compile(r"historical_training_data_until_(\d{8})\.parquet$")

    with_date: list[tuple[pd.Timestamp, str]] = []
    without_date: list[dict] = []
    for obj in parquet_objects:
        key = obj.get("Key", "")
        match = pattern.search(key)
        if match:
            parsed = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
            if pd.notna(parsed):
                with_date.append((parsed, key))
                continue
        without_date.append(obj)

    if with_date:
        with_date.sort(key=lambda x: x[0], reverse=True)
        return with_date[0][1]

    without_date.sort(key=lambda x: x.get("LastModified"), reverse=True)
    return without_date[0]["Key"]


def _init_tabpfn_client() -> type:
    """Initialize tabpfn client auth and return TabPFNRegressor class."""
    try:
        import tabpfn_client
    except ImportError as exc:
        raise ImportError(
            "tabpfn_client is not installed. Install it with: pip install tabpfn-client"
        ) from exc

    token = os.getenv("TABPFN_ACCESS_TOKEN")

    # If we have a token, set it directly
    if token:
        set_token_fn = getattr(tabpfn_client, "set_access_token", None)
        if callable(set_token_fn):
            set_token_fn(token)
        else:
            raise RuntimeError(
                "tabpfn_client.set_access_token not found. "
                "Ensure tabpfn-client is properly installed."
            )
    else:
        # No token provided - try init() for interactive auth
        # This will fail in non-interactive environments (e.g., GitHub Actions)
        init_fn = getattr(tabpfn_client, "init", None)
        if callable(init_fn):
            init_fn()
        else:
            raise RuntimeError(
                "TABPFN_ACCESS_TOKEN environment variable not set and "
                "tabpfn_client.init() not available for interactive auth."
            )

    return tabpfn_client.TabPFNRegressor


def load_and_predict_tabpfn_client_for_nba_games(
    df: pd.DataFrame,
    prediction_date: str | datetime | pd.Timestamp,
    prediction_datetime: datetime | None = None,
    historical_train_s3_key: str | None = None,
    historical_train_prefix: str = "train_data/",
    model_name: str = "tabpfn_client_regressor",
    model_version: str = "2.5",
) -> pd.DataFrame:
    """
    Predict NBA totals using TabPFN client regressor and upload to PostgreSQL.

    Flow:
    1) Load historical parquet from S3
    2) Merge with incoming prediction dataframe
    3) Drop duplicates by GAME_ID keeping historical parquet row
    4) Clean merged dataframe
    5) Fit TabPFN client regressor and predict incoming games
    6) Upload prediction summary to PostgreSQL
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    if "GAME_ID" not in df.columns:
        raise ValueError("Input dataframe must include GAME_ID")

    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))

    incoming_df = df.copy()
    incoming_df["GAME_ID"] = incoming_df["GAME_ID"].astype(str)

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket

    selected_key = (
        historical_train_s3_key
        or os.getenv("TABPFN_HISTORICAL_TRAIN_S3_KEY")
        or _select_latest_historical_train_key(
            s3_client=s3, bucket=bucket, prefix=historical_train_prefix
        )
    )

    historical_bytes = read_s3_object_bytes(
        s3_client=s3,
        bucket=bucket,
        key=selected_key,
    )
    historical_df = load_parquet_from_bytes(historical_bytes)

    if "GAME_ID" not in historical_df.columns:
        raise ValueError(
            f"Historical parquet does not contain GAME_ID: s3://{bucket}/{selected_key}"
        )

    historical_df["GAME_ID"] = historical_df["GAME_ID"].astype(str)

    merged_df = pd.concat([historical_df, incoming_df], ignore_index=True, sort=False)
    merged_df = merged_df.drop_duplicates(subset=["GAME_ID"], keep="first")

    cleaned_df = clean_dataframe_for_training(
        merged_df,
        nan_threshold=50,
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
    )
    cleaned_df = _add_na_tracking_columns(cleaned_df)

    if "TOTAL_OVER_UNDER_LINE" not in cleaned_df.columns:
        raise ValueError("TOTAL_OVER_UNDER_LINE is required for prediction")

    cleaned_df["LINE_ERROR"] = pd.to_numeric(
        cleaned_df.get("TOTAL_POINTS"), errors="coerce"
    ) - pd.to_numeric(cleaned_df["TOTAL_OVER_UNDER_LINE"], errors="coerce")

    prediction_day = pd.to_datetime(prediction_date, errors="coerce")
    if pd.isna(prediction_day):
        raise ValueError(f"Invalid prediction_date: {prediction_date}")
    prediction_day = prediction_day.date()

    game_dates = pd.to_datetime(cleaned_df["GAME_DATE"], errors="coerce").dt.date
    predict_mask = game_dates == prediction_day

    df_predictable = cleaned_df[predict_mask].copy()

    if df_predictable.empty:
        raise ValueError(
            "No games found for TabPFN prediction after merge/cleaning for "
            f"prediction_date={prediction_day}"
        )

    train_mask = cleaned_df["LINE_ERROR"].notna() & (~predict_mask)
    df_train = cleaned_df[train_mask].copy()
    if df_train.empty:
        raise ValueError("No training rows available for TabPFN fit after cleaning")

    drop_feature_cols = {
        "LINE_ERROR",
        "TOTAL_POINTS",
        "GAME_ID",
        "GAME_DATE",
        "GAME_TIME",
        "SEASON_TYPE",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "MATCHUP",
        "MATCHUP_TEAM_HOME",
        "PRED_LINE_ERROR",
        "PRED_PICK",
        "PRED_TOTAL_POINTS",
    }

    feature_cols = [c for c in df_train.columns if c not in drop_feature_cols]
    if not feature_cols:
        raise ValueError("No feature columns available for TabPFN")

    X_train = df_train[feature_cols].copy()
    y_train = pd.to_numeric(df_train["LINE_ERROR"], errors="coerce")
    valid_train = y_train.notna()
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    X_pred = df_predictable.copy()
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = np.nan
    X_pred = X_pred[feature_cols]

    TabPFNRegressor = _init_tabpfn_client()
    regressor = TabPFNRegressor()

    # print("Remove This line after testing TabPFN client fit/predict")
    print("Training TabPFN client regressor on historical data")
    regressor.fit(X_train, y_train)  # Temporary limit for testing
    print("Predicting with TabPFN client regressor for incoming games")
    pred_line_error = regressor.predict(X_pred)

    df_predictable["PRED_LINE_ERROR"] = pred_line_error
    df_predictable["PRED_PICK"] = np.where(
        df_predictable["PRED_LINE_ERROR"] > 0,
        "OVER",
        np.where(df_predictable["PRED_LINE_ERROR"] < 0, "UNDER", "PUSH"),
    )
    df_predictable["PRED_TOTAL_POINTS"] = pd.to_numeric(
        df_predictable["TOTAL_OVER_UNDER_LINE"], errors="coerce"
    ) + pd.to_numeric(df_predictable["PRED_LINE_ERROR"], errors="coerce")

    df_predictable.rename(columns={"MATCHUP_TEAM_HOME": "MATCHUP"}, inplace=True)
    if "GAME_DATE" in df_predictable.columns:
        df_predictable["GAME_DATE"] = (
            df_predictable["GAME_DATE"].astype(str).str.split("T").str[0]
        )

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
        "NA_COLUMNS_COUNT",
        "NA_COLUMNS_NAMES",
    ]
    missing_summary = [c for c in summary_columns if c not in df_predictable.columns]
    if missing_summary:
        raise ValueError(f"Missing summary columns for upload: {missing_summary}")

    df_summary = df_predictable[summary_columns].copy()
    df_summary["PREDICTION_DATETIME"] = prediction_datetime
    df_summary["PREDICTION_DATE"] = pd.to_datetime(prediction_datetime).date()

    def ensure_timezone_aware(dt_value):
        if pd.isna(dt_value):
            return pd.NaT

        if not isinstance(dt_value, (pd.Timestamp, datetime)):
            try:
                dt_value = pd.to_datetime(dt_value)
            except Exception:
                return pd.NaT

        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo is None:
            return dt_value.tz_localize("US/Pacific")
        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo is not None:
            return dt_value
        return pd.to_datetime(dt_value).tz_localize("US/Pacific")

    game_time_aware = df_summary["GAME_TIME"].apply(ensure_timezone_aware)
    df_summary["TIME_TO_MATCH_MINUTES"] = (
        game_time_aware - df_summary["PREDICTION_DATETIME"]
    ).dt.total_seconds() / 60
    df_summary["TIME_TO_MATCH_MINUTES"] = (
        pd.to_numeric(df_summary["TIME_TO_MATCH_MINUTES"], errors="coerce")
        .fillna(0)
        .round(0)
        .astype(int)
    )

    df_summary["MODEL_NAME"] = model_name
    df_summary["MODEL_TYPE"] = "TabPFNRegressor"
    df_summary["MODEL_VERSION"] = f"{model_version}|train={selected_key}"

    df_summary_clean = df_summary.dropna(subset=["PRED_LINE_ERROR"])
    upload_predictions_to_postgre(df_summary_clean)

    return df_summary_clean
