import os
import sys
from pathlib import Path

from nba_ou.config.settings import SETTINGS
from nba_ou.utils.s3_models import (
    get_first_joblib_from_prefix,
    load_joblib_from_bytes,
    make_s3_client,
    read_s3_object_bytes,
)


def _env_or_default(env_name: str, default: str) -> str:
    value = os.getenv(env_name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def main() -> int:
    region = _env_or_default("S3_AWS_REGION", SETTINGS.s3_aws_region)
    bucket = _env_or_default("S3_MODEL_BUCKET", SETTINGS.s3_bucket)
    prefix = _env_or_default(
        "S3_MODEL_PREFIX", SETTINGS.s3_regressor_full_dataset_prefix
    )
    profile = SETTINGS.s3_aws_profile

    print("Starting S3 model smoke test")
    print(f"  region: {region}")
    print(f"  bucket: {bucket}")
    print(f"  prefix: {prefix}")
    print(f"  profile: {profile or '<none>'}")

    s3 = make_s3_client(profile=profile, region=region)

    # Find the first .joblib file in the prefix
    key = get_first_joblib_from_prefix(
        s3_client=s3,
        bucket=bucket,
        prefix=prefix,
    )

    if not key:
        print(f"ERROR: No .joblib file found in prefix: {prefix}")
        return 1

    print(f"  Found model: {key}")

    model_bytes = read_s3_object_bytes(
        s3_client=s3,
        bucket=bucket,
        key=key,
    )
    model = load_joblib_from_bytes(model_bytes)

    model_type = f"{type(model).__module__}.{type(model).__name__}"
    print(f"Smoke test passed. Loaded model type: {model_type}")
    if hasattr(model, "n_features_in_"):
        print(f"n_features_in_: {model.n_features_in_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
