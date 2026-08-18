"""Copy the Aiven line-history store to S3 as Parquet.

Lands beside the Supabase backups, under the same
``backups/db/<YYYY-MM-DD>/<schema>/`` layout, so a restore reads the same way
regardless of which database a table came from. The schema name keeps the two
apart: everything here goes under ``.../line_history/``.

**Partitions are the thing to get right.** ``lh_line`` is LIST-partitioned by
season, and ``information_schema.tables`` reports the parent *and* every
partition as ``BASE TABLE``. Backing up the naive table list would therefore
write all 1.8M rows twice -- once via the parent, once across the leaves. This
module discovers targets through ``pg_class.relkind`` instead: the parent is
skipped and each partition is exported on its own. That also keeps peak memory
to one season rather than the whole fact table, and makes a single season
restorable without touching the rest.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import psycopg

from nba_ou.postgre_db.config.db_config import get_config
from nba_ou.utils.s3_models import make_s3_client

from .schema import SCHEMA

#: Shared with ``scripts/backup_db_to_s3.py`` so both databases restore alike.
S3_BACKUP_PREFIX = "backups/db"

#: Rows per SELECT batch. Bounds the driver's buffer, not the final frame.
CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class BackupTarget:
    """One relation to export."""

    table: str
    is_partition: bool

    @property
    def label(self) -> str:
        return "partition" if self.is_partition else "table"


@dataclass
class BackupResult:
    bucket: str
    prefix: str
    uploaded: list[str] = field(default_factory=list)
    rows: int = 0
    bytes_written: int = 0
    skipped_parents: list[str] = field(default_factory=list)


def get_s3_settings() -> dict[str, str | None]:
    """Bucket / region / profile from ``[S3]``, overridable by env vars."""
    import os

    config = get_config()
    profile = os.getenv("S3_AWS_PROFILE", config.get("S3", "AWS_PROFILE", fallback=""))
    return {
        "bucket": os.getenv("S3_BACKUP_BUCKET") or config.get("S3", "BUCKET"),
        "region": os.getenv("AWS_REGION") or config.get("S3", "AWS_REGION"),
        # Empty means "use the ambient credentials", which is what CI provides
        # via OIDC; a named profile is only for local runs.
        "profile": (profile.strip() or None) if profile else None,
    }


def discover_backup_targets(
    conn: psycopg.Connection,
    schema: str = SCHEMA,
) -> tuple[list[BackupTarget], list[str]]:
    """Relations worth exporting, plus the partitioned parents deliberately skipped.

    ``relkind`` distinguishes what ``information_schema`` cannot:

    * ``p`` -- partitioned parent. Holds no rows itself; skipped.
    * ``r`` with a ``pg_inherits`` row -- a partition leaf. Exported.
    * ``r`` with none -- an ordinary table. Exported.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname,
                   c.relkind,
                   (i.inhrelid IS NOT NULL) AS is_partition
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_inherits i ON i.inhrelid = c.oid
            WHERE n.nspname = %s
              AND c.relkind IN ('r', 'p')
            ORDER BY c.relname
            """,
            (schema,),
        )
        rows = cur.fetchall()

    targets: list[BackupTarget] = []
    skipped: list[str] = []
    for relname, relkind, is_partition in rows:
        if relkind == "p":
            skipped.append(relname)
            continue
        targets.append(BackupTarget(table=relname, is_partition=bool(is_partition)))
    return targets, skipped


def export_table_to_parquet(
    conn: psycopg.Connection,
    schema: str,
    table: str,
) -> tuple[bytes, int]:
    """Export one relation to an in-memory Parquet file."""
    query = f'SELECT * FROM "{schema}"."{table}"'
    chunks = [
        chunk
        for chunk in pd.read_sql(query, conn, chunksize=CHUNK_SIZE)  # type: ignore[call-overload]
    ]
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    buffer = io.BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    buffer.seek(0)
    return buffer.read(), len(frame)


def backup_line_history(
    conn: psycopg.Connection,
    *,
    schema: str = SCHEMA,
    dry_run: bool = False,
    date_tag: str | None = None,
    s3_client: Any = None,
    progress: bool = True,
) -> BackupResult:
    """Export every relation in ``schema`` to ``s3://<bucket>/backups/db/<tag>/<schema>/``.

    Each run writes under its own date tag, so runs never overwrite one another
    and a restore is just "pick a date".
    """
    tag = date_tag or datetime.now(UTC).strftime("%Y-%m-%d")
    settings = get_s3_settings()
    bucket = str(settings["bucket"])
    prefix = f"{S3_BACKUP_PREFIX}/{tag}/{schema}"

    result = BackupResult(bucket=bucket, prefix=prefix)
    targets, result.skipped_parents = discover_backup_targets(conn, schema)

    if progress:
        mode = "[DRY RUN] would write" if dry_run else "writing"
        print(f"{mode} to s3://{bucket}/{prefix}/")
        if result.skipped_parents:
            print(
                "  partitioned parents skipped (their rows live in the "
                f"partitions): {', '.join(result.skipped_parents)}"
            )

    if not targets:
        if progress:
            print(f"  no relations found in schema '{schema}'.")
        return result

    if s3_client is None and not dry_run:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    for target in targets:
        key = f"{prefix}/{target.table}.parquet"
        if dry_run:
            if progress:
                print(f"  → {target.table} ({target.label})  →  s3://{bucket}/{key}")
            result.uploaded.append(key)
            continue

        if progress:
            print(f"  exporting {target.table} …", end=" ", flush=True)
        data, row_count = export_table_to_parquet(conn, schema, target.table)
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)

        result.uploaded.append(key)
        result.rows += row_count
        result.bytes_written += len(data)
        if progress:
            print(f"{row_count:,} rows, {len(data) / (1024 * 1024):.2f} MB ✓")

    if progress:
        verb = "would upload" if dry_run else "uploaded"
        size_mb = result.bytes_written / (1024 * 1024)
        print(
            f"\nDone. {verb} {len(result.uploaded)} file(s), "
            f"{result.rows:,} rows, {size_mb:.2f} MB"
        )
        print(f"S3 prefix: s3://{bucket}/{prefix}/")

    return result


def list_backup_dates(
    *,
    schema: str = SCHEMA,
    s3_client: Any = None,
    limit: int = 24,
) -> list[str]:
    """Date tags that already hold a backup of ``schema``, newest first."""
    settings = get_s3_settings()
    bucket = str(settings["bucket"])
    if s3_client is None:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    paginator = s3_client.get_paginator("list_objects_v2")
    tags: set[str] = set()
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{S3_BACKUP_PREFIX}/", Delimiter="/"
    ):
        for entry in page.get("CommonPrefixes") or []:
            tag = entry["Prefix"].removeprefix(f"{S3_BACKUP_PREFIX}/").strip("/")
            if tag:
                tags.add(tag)

    dated = sorted(tags, reverse=True)
    keep: list[str] = []
    for tag in dated:
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=f"{S3_BACKUP_PREFIX}/{tag}/{schema}/", MaxKeys=1
        )
        if response.get("KeyCount"):
            keep.append(tag)
        if len(keep) >= limit:
            break
    return keep
