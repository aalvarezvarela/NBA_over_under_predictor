import os

from nba_ou.config.settings import SETTINGS
from nba_ou.utils.s3_models import (
    get_latest_model_bundle_from_prefix,
    make_s3_client,
    read_s3_json_object,
    read_s3_object_bytes,
)
from xgboost import XGBRegressor


def _env_or_default(env_name: str, default: str) -> str:
    value = os.getenv(env_name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def main() -> int:
    region = _env_or_default("S3_AWS_REGION", SETTINGS.s3_aws_region)
    bucket = _env_or_default("S3_MODEL_BUCKET", SETTINGS.s3_bucket)
    default_prefix = (
        SETTINGS.prediction_model_prefixes[0]
        if SETTINGS.prediction_model_prefixes
        else ""
    )
    prefix = _env_or_default("S3_MODEL_PREFIX", default_prefix)
    profile = SETTINGS.s3_aws_profile

    print("Starting S3 model smoke test")
    print(f"  region: {region}")
    print(f"  bucket: {bucket}")
    print(f"  prefix: {prefix}")
    print(f"  profile: {profile or '<none>'}")

    s3 = make_s3_client(profile=profile, region=region)

    bundle = get_latest_model_bundle_from_prefix(
        s3_client=s3,
        bucket=bucket,
        prefix=prefix,
    )

    if bundle is None:
        print(f"ERROR: No .json model bundle found in prefix: {prefix}")
        return 1

    print(f"  Found model: {bundle.model_key}")
    print(f"  Found metadata: {bundle.meta_key}")

    model_bytes = read_s3_object_bytes(
        s3_client=s3,
        bucket=bucket,
        key=bundle.model_key,
    )
    metadata = read_s3_json_object(
        s3_client=s3,
        bucket=bucket,
        key=bundle.meta_key,
    )

    model = XGBRegressor()
    model.load_model(bytearray(model_bytes))

    model_type = f"{type(model).__module__}.{type(model).__name__}"
    print(f"Smoke test passed. Loaded model type: {model_type}")
    print(f"Model name: {metadata.get('model', {}).get('name')}")
    if hasattr(model, "n_features_in_"):
        print(f"n_features_in_: {model.n_features_in_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
