import io
from dataclasses import dataclass

import boto3
import joblib
import pandas as pd


@dataclass(frozen=True)
class S3ModelLocation:
    bucket: str
    key: str


def make_s3_client(*, profile: str | None, region: str):
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    return session.client("s3")


def read_s3_object_bytes(*, s3_client, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def upload_file_to_s3(*, s3_client, bucket: str, key: str, file_path: str) -> None:
    """Upload a file to S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key (path within bucket)
        file_path: Local file path to upload
    """
    s3_client.upload_file(file_path, bucket, key)


def upload_bytes_to_s3(*, s3_client, bucket: str, key: str, data: bytes) -> None:
    """Upload bytes directly to S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key (path within bucket)
        data: Bytes data to upload
    """
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)


def load_joblib_from_bytes(b: bytes):
    return joblib.load(io.BytesIO(b))


def list_s3_objects(*, s3_client, bucket: str, prefix: str) -> list[dict]:
    """
    List objects under an S3 prefix.

    Returns:
        list[dict]: Raw S3 object dictionaries (contains Key, LastModified, Size, ...)
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects: list[dict] = []
    for page in pages:
        objects.extend(page.get("Contents", []))
    return objects


def load_parquet_from_bytes(b: bytes) -> pd.DataFrame:
    """Load a parquet DataFrame from bytes."""
    return pd.read_parquet(io.BytesIO(b))


def get_first_joblib_from_prefix(*, s3_client, bucket: str, prefix: str) -> str | None:
    """
    Find the first .joblib file under an S3 prefix.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix to search under

    Returns:
        Full S3 key of the first .joblib file found, or None if none exist
    """
    objects = list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix)

    # Filter for .joblib files and sort by key name
    joblib_files = [obj["Key"] for obj in objects if obj["Key"].endswith(".joblib")]

    if not joblib_files:
        return None

    # Return the first one (sorted alphabetically)
    return sorted(joblib_files)[0]
