"""Upload prediction pipeline snapshots to S3.

Folder layout
─────────────
prediction_snapshots/
  {date}/
    {timestamp}/
      input_features/
        df_to_predict.parquet          ← today-only rows used by all models
      models/
        {model_slug}/
          predictions.parquet          ← output of one model
        tabpfn_client/
          predictions.parquet
"""

from __future__ import annotations

import io
import re
from datetime import datetime

import pandas as pd

from nba_ou.utils.s3_models import upload_bytes_to_s3

DEFAULT_SNAPSHOT_PREFIX = "prediction_snapshots"


def _slugify_model_prefix(model_prefix: str) -> str:
    """Convert an S3 model prefix like 'models/total_points_full_dataset/production/'
    into a compact directory slug like 'total_points_full_dataset_production'."""
    slug = model_prefix.strip("/").replace("/", "_")
    slug = re.sub(r"^models_", "", slug)
    slug = re.sub(r"[^a-z0-9_]+", "_", slug.lower()).strip("_")
    return slug or "unknown_model"


def _format_snapshot_timestamp(dt: datetime) -> str:
    """Format a datetime into a directory-safe timestamp string."""
    return dt.strftime("%Y-%m-%dT%H_%M_%S")


def _dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


def build_snapshot_base_key(
    pipeline_start_time: datetime,
    date_to_predict: str,
    *,
    prefix: str = DEFAULT_SNAPSHOT_PREFIX,
) -> str:
    """Return the S3 key prefix for a particular pipeline run.

    Example: ``prediction_snapshots/2026-04-10/2026-04-10T14_30_00/``
    """
    ts = _format_snapshot_timestamp(pipeline_start_time)
    return f"{prefix}/{date_to_predict}/{ts}/"


def upload_input_features_snapshot(
    *,
    s3_client,
    bucket: str,
    pipeline_start_time: datetime,
    date_to_predict: str,
    df_to_predict: pd.DataFrame,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
) -> str:
    """Upload the input feature DataFrame for today's games to S3.

    Returns the S3 key of the uploaded file.
    """
    base = build_snapshot_base_key(
        pipeline_start_time, date_to_predict, prefix=snapshot_prefix
    )
    key = f"{base}input_features/df_to_predict.parquet"
    data = _dataframe_to_parquet_bytes(df_to_predict)
    upload_bytes_to_s3(s3_client=s3_client, bucket=bucket, key=key, data=data)
    return key


def upload_model_predictions_snapshot(
    *,
    s3_client,
    bucket: str,
    pipeline_start_time: datetime,
    date_to_predict: str,
    model_prefix: str,
    predictions_df: pd.DataFrame,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
) -> str:
    """Upload one model's prediction output to the snapshot folder.

    Returns the S3 key of the uploaded file.
    """
    base = build_snapshot_base_key(
        pipeline_start_time, date_to_predict, prefix=snapshot_prefix
    )
    slug = _slugify_model_prefix(model_prefix)
    key = f"{base}models/{slug}/predictions.parquet"
    data = _dataframe_to_parquet_bytes(predictions_df)
    upload_bytes_to_s3(s3_client=s3_client, bucket=bucket, key=key, data=data)
    return key
