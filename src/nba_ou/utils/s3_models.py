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


def load_joblib_from_bytes(b: bytes):
    return joblib.load(io.BytesIO(b))
