import io
from dataclasses import dataclass

import boto3
import joblib


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
