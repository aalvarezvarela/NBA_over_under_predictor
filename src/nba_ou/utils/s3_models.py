import io
from dataclasses import dataclass

import boto3
import joblib


@dataclass(frozen=True)
class S3ModelLocation:
    bucket: str
    key: str


def make_s3_client(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3")


def read_s3_object_bytes(*, s3_client, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def load_joblib_from_bytes(b: bytes):
    return joblib.load(io.BytesIO(b))
