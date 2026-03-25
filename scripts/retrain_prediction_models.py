#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.modeling.retraining import (
    retrain_model_to_staging_with_inferred_settings,
)
from nba_ou.utils.s3_models import make_s3_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "train_data"
TODAY_TIMEZONE = ZoneInfo("Europe/Madrid")
XGB_STATIC_PARAMS = {
    "booster": "gbtree",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "random_state": 16,
    "n_jobs": -1,
    "verbosity": 0,
}
DATE_COLUMN = "GAME_DATE"
MINIMUM_LINE_VALUE = 100.0


def _today_limit_date() -> str:
    return datetime.now(TODAY_TIMEZONE).strftime("%Y-%m-%d")


def _create_training_dataframe_for_today(limit_date: str) -> tuple[pd.DataFrame, Path]:
    df_train = create_df_to_predict(
        todays_prediction=False,
        recent_limit_to_include=limit_date,
        older_season_limit=None,
    )

    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        TRAIN_DATA_DIR
        / f"all_odds_training_data_until_{pd.to_datetime(limit_date).strftime('%Y%m%d')}.csv"
    )
    df_train.to_csv(output_path, index=False)
    return df_train, output_path


def main() -> None:
    limit_date = _today_limit_date()
    configured_prefixes = SETTINGS.prediction_model_prefixes
    if not configured_prefixes:
        raise ValueError(
            "No prediction model prefixes configured in [PredictionModels] "
            "S3_MODEL_PREFIXES."
        )

    print(f"Creating training dataframe up to {limit_date}")
    df_train, output_path = _create_training_dataframe_for_today(limit_date)
    print(f"Training dataframe saved to {output_path}")
    print(f"Training dataframe rows: {len(df_train)}")

    s3_client = make_s3_client(
        profile=SETTINGS.s3_aws_profile,
        region=SETTINGS.s3_aws_region,
    )
    bucket = SETTINGS.s3_bucket

    for prefix in configured_prefixes:
        print(f"Retraining {prefix}")
        bundle = retrain_model_to_staging_with_inferred_settings(
            df_train,
            production_prefix=prefix,
            date_column=DATE_COLUMN,
            minimum_line_value=MINIMUM_LINE_VALUE,
            xgb_static_params=XGB_STATIC_PARAMS,
            s3_client=s3_client,
            bucket=bucket,
        )
        print(
            "Staged retrained bundle: "
            f"prefix={prefix}, staging_prefix={bundle.staging_prefix}, "
            f"model_key={bundle.model_key}, meta_key={bundle.meta_key}, "
            f"train_games={bundle.train_games}"
        )


if __name__ == "__main__":
    main()
