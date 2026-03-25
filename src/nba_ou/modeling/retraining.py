from pathlib import Path

import pandas as pd
from nba_ou.modeling.retraining_utils import (
    RetrainedModelBundle,
    build_retraining_settings_from_artifacts,
    build_updated_retraining_metadata,
    load_production_artifacts_from_s3,
    prepare_retraining_dataframe_from_raw,
    resolve_train_games_to_use,
    retrain_model,
    save_retrained_bundle_to_staging,
    select_rolling_training_window,
)


def retrain_model_to_staging_with_inferred_settings(
    raw_df: pd.DataFrame,
    *,
    production_prefix: str,
    date_column: str,
    minimum_line_value: float | None,
    xgb_static_params: dict,
    s3_client=None,
    bucket: str | None = None,
) -> RetrainedModelBundle:
    """Retrain one configured production model using settings inferred from metadata."""
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if s3_client is None:
        s3_client = make_s3_client(
            profile=SETTINGS.s3_aws_profile,
            region=SETTINGS.s3_aws_region,
        )
    artifacts = load_production_artifacts_from_s3(
        production_prefix=production_prefix,
        s3_client=s3_client,
        bucket=bucket,
    )
    if (
        artifacts.metadata.schema_info is None
        or not artifacts.metadata.schema_info.feature_names
    ):
        raise ValueError("Production metadata is missing schema.feature_names.")

    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column=date_column,
        minimum_line_value=minimum_line_value,
        xgb_static_params=xgb_static_params,
    )
    prepared_df = prepare_retraining_dataframe_from_raw(raw_df, settings=settings)
    training_window_df = select_rolling_training_window(
        prepared_df,
        date_col=settings.date_column,
        train_games=resolve_train_games_to_use(settings=settings),
    )
    model = retrain_model(training_window_df, settings=settings)
    metadata = build_updated_retraining_metadata(
        training_window_df,
        settings=settings,
    )
    staging_prefix, model_key, meta_key = save_retrained_bundle_to_staging(
        model,
        settings=settings,
        metadata=metadata,
        production_prefix=production_prefix,
        s3_client=s3_client,
        bucket=artifacts.bucket,
    )

    return RetrainedModelBundle(
        model=model,
        metadata=metadata,
        bucket=artifacts.bucket,
        staging_prefix=staging_prefix,
        model_key=model_key,
        meta_key=meta_key,
        target_column=settings.target_column,
        train_games=int(len(training_window_df)),
        feature_count=len(settings.feature_names),
    )


def retrain_production_model_to_staging(
    raw_df: pd.DataFrame,
    *,
    production_prefix: str = "models/total_points_full_dataset/production/",
    s3_client=None,
    bucket: str | None = None,
) -> RetrainedModelBundle:
    """Retrain the current production model on the latest rolling window and stage it."""
    date_column = "GAME_DATE"
    minimum_line_value = 100.0
    xgb_static_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": 16,
        "n_jobs": -1,
        "verbosity": 0,
    }

    return retrain_model_to_staging_with_inferred_settings(
        raw_df,
        production_prefix=production_prefix,
        date_column=date_column,
        minimum_line_value=minimum_line_value,
        xgb_static_params=xgb_static_params,
        s3_client=s3_client,
        bucket=bucket,
    )


if __name__ == "__main__":
    from nba_ou.config.settings import SETTINGS

    training_data_path = Path(
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data/"
        "all_odds_training_data_until_20260318.csv"
    )

    df_train = pd.read_csv(training_data_path)
    configured_prefixes = SETTINGS.prediction_model_prefixes
    if not configured_prefixes:
        raise ValueError(
            "No prediction model prefixes configured in [PredictionModels] "
            "S3_MODEL_PREFIXES."
        )

    for production_prefix in configured_prefixes:
        retrained_bundle = retrain_production_model_to_staging(
            df_train,
            production_prefix=production_prefix,
        )

        print(f"Production prefix: {production_prefix}")
        print(f"Staging prefix: {retrained_bundle.staging_prefix}")
        print(f"Model key: {retrained_bundle.model_key}")
        print(f"Meta key: {retrained_bundle.meta_key}")
        print(f"Target column: {retrained_bundle.target_column}")
        print(f"Train games used: {retrained_bundle.train_games}")
        print(f"Feature count: {retrained_bundle.feature_count}")
        print()
