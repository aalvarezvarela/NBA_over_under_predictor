from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nba_ou.config.settings import SETTINGS
from nba_ou.utils.s3_models import list_s3_objects, make_s3_client


@dataclass(frozen=True)
class ModelBundle:
    model_key: str
    meta_key: str

    @property
    def filenames(self) -> tuple[str, str]:
        return (Path(self.model_key).name, Path(self.meta_key).name)


def derive_staging_prefix(production_prefix: str) -> str:
    marker = "/production/"
    if marker not in production_prefix:
        raise ValueError(
            f"Configured production prefix must contain '{marker}': {production_prefix}"
        )
    return production_prefix.replace(marker, "/staging/", 1)


def derive_temp_prefix(production_prefix: str) -> str:
    marker = "/production/"
    if marker not in production_prefix:
        raise ValueError(
            f"Configured production prefix must contain '{marker}': {production_prefix}"
        )
    suffix = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return production_prefix.replace(marker, f"/_promotion_tmp/{suffix}/", 1)


def _prefix_objects(*, s3_client, bucket: str, prefix: str) -> list[dict]:
    return [
        obj
        for obj in list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix)
        if obj["Key"] != prefix and not obj["Key"].endswith("/")
    ]


def extract_single_bundle(
    *,
    s3_client,
    bucket: str,
    prefix: str,
) -> ModelBundle | None:
    objects = _prefix_objects(s3_client=s3_client, bucket=bucket, prefix=prefix)
    if not objects:
        return None

    keys = sorted(obj["Key"] for obj in objects)
    if len(keys) != 2:
        raise ValueError(
            f"Expected exactly 2 objects under {prefix}, found {len(keys)}: {keys}"
        )

    model_keys = [
        key for key in keys if key.endswith(".json") and not key.endswith(".meta.json")
    ]
    meta_keys = [key for key in keys if key.endswith(".meta.json")]

    if len(model_keys) != 1 or len(meta_keys) != 1:
        raise ValueError(
            f"Expected one model .json and one .meta.json under {prefix}, found: {keys}"
        )

    model_key = model_keys[0]
    meta_key = meta_keys[0]
    expected_meta = f"{model_key[:-len('.json')]}.meta.json"
    if meta_key != expected_meta:
        raise ValueError(
            f"Metadata/model mismatch under {prefix}: model={model_key}, meta={meta_key}"
        )

    return ModelBundle(model_key=model_key, meta_key=meta_key)


def _copy_object(*, s3_client, bucket: str, source_key: str, target_key: str) -> None:
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=target_key,
    )


def _delete_object(*, s3_client, bucket: str, key: str) -> None:
    s3_client.delete_object(Bucket=bucket, Key=key)


def move_bundle(
    *,
    s3_client,
    bucket: str,
    bundle: ModelBundle,
    destination_prefix: str,
    dry_run: bool,
) -> ModelBundle:
    target_model_key = f"{destination_prefix.rstrip('/')}/{Path(bundle.model_key).name}"
    target_meta_key = f"{destination_prefix.rstrip('/')}/{Path(bundle.meta_key).name}"

    if dry_run:
        print(f"    DRY RUN copy {bundle.model_key} -> {target_model_key}")
        print(f"    DRY RUN copy {bundle.meta_key} -> {target_meta_key}")
        print(f"    DRY RUN delete {bundle.model_key}")
        print(f"    DRY RUN delete {bundle.meta_key}")
    else:
        _copy_object(
            s3_client=s3_client,
            bucket=bucket,
            source_key=bundle.model_key,
            target_key=target_model_key,
        )
        _copy_object(
            s3_client=s3_client,
            bucket=bucket,
            source_key=bundle.meta_key,
            target_key=target_meta_key,
        )
        _delete_object(s3_client=s3_client, bucket=bucket, key=bundle.model_key)
        _delete_object(s3_client=s3_client, bucket=bucket, key=bundle.meta_key)

    return ModelBundle(model_key=target_model_key, meta_key=target_meta_key)


def promote_prediction_models(*, dry_run: bool = True) -> None:
    prefixes = SETTINGS.prediction_model_prefixes
    if not prefixes:
        raise ValueError(
            "No prediction model prefixes configured in [PredictionModels] S3_MODEL_PREFIXES."
        )

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket

    print(f"{'DRY RUN' if dry_run else 'EXECUTE'} promotion for bucket: {bucket}")

    for production_prefix in prefixes:
        staging_prefix = derive_staging_prefix(production_prefix)
        temp_prefix = derive_temp_prefix(production_prefix)

        print(f"\nModel family: {production_prefix}")
        print(f"  staging: {staging_prefix}")

        production_bundle = extract_single_bundle(
            s3_client=s3,
            bucket=bucket,
            prefix=production_prefix,
        )
        staging_bundle = extract_single_bundle(
            s3_client=s3,
            bucket=bucket,
            prefix=staging_prefix,
        )

        if staging_bundle is None:
            print("  No staging bundle found. Skipping.")
            continue

        print(f"  Staging bundle: {staging_bundle.filenames}")
        if production_bundle is None:
            print("  Production bundle: empty")
        else:
            print(f"  Production bundle: {production_bundle.filenames}")

        temp_bundle = None
        if production_bundle is not None:
            print(f"  Moving current production bundle to temp: {temp_prefix}")
            temp_bundle = move_bundle(
                s3_client=s3,
                bucket=bucket,
                bundle=production_bundle,
                destination_prefix=temp_prefix,
                dry_run=dry_run,
            )

        print("  Promoting staging bundle to production")
        _ = move_bundle(
            s3_client=s3,
            bucket=bucket,
            bundle=staging_bundle,
            destination_prefix=production_prefix,
            dry_run=dry_run,
        )

        if temp_bundle is not None:
            print("  Moving previous production bundle to staging")
            _ = move_bundle(
                s3_client=s3,
                bucket=bucket,
                bundle=temp_bundle,
                destination_prefix=staging_prefix,
                dry_run=dry_run,
            )

    print("\nPromotion flow completed.")
