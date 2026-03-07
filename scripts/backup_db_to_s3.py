"""
Backup all Supabase PostgreSQL schemas to S3 as compressed CSV files.

Each schema/table is exported as a gzipped CSV and uploaded to:
    s3://<BUCKET>/backups/db/<YYYY-MM-DD>/<schema>/<table>.csv.gz

Usage:
    python scripts/backup_db_to_s3.py              # backup all schemas
    python scripts/backup_db_to_s3.py --schemas nba_games nba_ou_predictions
    python scripts/backup_db_to_s3.py --dry-run     # list what would be backed up
"""

from __future__ import annotations

import argparse
import gzip
import io
from datetime import datetime, timezone

import boto3
import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_config,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BACKUP_PREFIX = "backups/db"


def _get_s3_settings() -> dict[str, str]:
    """Read S3 bucket / profile / region from config.ini + env vars."""
    config = get_config()
    import os

    bucket = os.getenv("S3_BACKUP_BUCKET") or config.get("S3", "BUCKET")
    region = os.getenv("AWS_REGION") or config.get("S3", "AWS_REGION")
    profile = os.getenv("S3_AWS_PROFILE", config.get("S3", "AWS_PROFILE", fallback=""))
    profile = profile.strip() if profile else None

    return {"bucket": bucket, "region": region, "profile": profile}


def _make_s3_client(settings: dict[str, str]):
    session_kwargs: dict = {"region_name": settings["region"]}
    if settings["profile"]:
        session_kwargs["profile_name"] = settings["profile"]
    session = boto3.Session(**session_kwargs)
    return session.client("s3")


# ---------------------------------------------------------------------------
# Schema / table discovery
# ---------------------------------------------------------------------------

ALL_SCHEMAS = [
    "nba_games",
    "nba_players",
    "nba_injuries",
    "nba_refs",
    "nba_odds",
    "nba_odds_mgm",
    "odds_sportsbook",
    "odds_yahoo",
    "nba_ou_predictions",
]


def discover_tables(conn, schema: str) -> list[str]:
    """Return table names inside *schema*."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (schema,),
        )
        return [row[0] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE = 50_000  # rows per SELECT batch


def export_table_to_gzipped_csv(conn, schema: str, table: str) -> bytes:
    """
    Stream a full table into an in-memory gzipped CSV.

    Uses server-side cursors to keep memory bounded for large tables.
    """
    query = f'SELECT * FROM "{schema}"."{table}"'

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        first_chunk = True
        for chunk_df in pd.read_sql(
            query,
            conn,
            chunksize=CHUNK_SIZE,  # type: ignore[arg-type]
        ):
            csv_bytes = chunk_df.to_csv(index=False, header=first_chunk).encode("utf-8")
            gz.write(csv_bytes)
            first_chunk = False

    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Main backup routine
# ---------------------------------------------------------------------------


def backup_schemas(
    schemas: list[str] | None = None,
    *,
    dry_run: bool = False,
    date_tag: str | None = None,
) -> dict[str, list[str]]:
    """
    Export each table in the requested schemas and upload to S3.

    Returns a dict  {schema: [uploaded_keys]}.
    """
    schemas = schemas or ALL_SCHEMAS
    tag = date_tag or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    s3_settings = _get_s3_settings()
    bucket = s3_settings["bucket"]

    if dry_run:
        print(f"[DRY RUN] Would back up to s3://{bucket}/{S3_BACKUP_PREFIX}/{tag}/\n")

    s3 = None if dry_run else _make_s3_client(s3_settings)
    conn = connect_nba_db()

    results: dict[str, list[str]] = {}

    try:
        for schema in schemas:
            tables = discover_tables(conn, schema)
            if not tables:
                print(f"  ⚠  Schema '{schema}': no tables found, skipping.")
                continue

            keys: list[str] = []
            for table in tables:
                s3_key = f"{S3_BACKUP_PREFIX}/{tag}/{schema}/{table}.csv.gz"

                if dry_run:
                    print(f"  → {schema}.{table}  →  s3://{bucket}/{s3_key}")
                    keys.append(s3_key)
                    continue

                print(f"  Exporting {schema}.{table} …", end=" ", flush=True)
                data = export_table_to_gzipped_csv(conn, schema, table)
                size_mb = len(data) / (1024 * 1024)
                print(f"{size_mb:.2f} MB", end=" ", flush=True)

                s3.put_object(Bucket=bucket, Key=s3_key, Body=data)
                print(f"→ s3://{bucket}/{s3_key} ✓")
                keys.append(s3_key)

            results[schema] = keys
    finally:
        conn.close()

    # Summary
    total = sum(len(v) for v in results.values())
    verb = "would upload" if dry_run else "uploaded"
    print(f"\n{'─' * 50}")
    print(f"Done. {verb} {total} file(s) across {len(results)} schema(s).")
    print(f"S3 prefix: s3://{bucket}/{S3_BACKUP_PREFIX}/{tag}/")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup Supabase DB schemas to S3 as gzipped CSVs"
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=None,
        help=f"Schemas to back up (default: all). Choices: {', '.join(ALL_SCHEMAS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be backed up without uploading",
    )
    parser.add_argument(
        "--date-tag",
        default=None,
        help="Override date tag folder (default: today YYYY-MM-DD)",
    )
    args = parser.parse_args()

    backup_schemas(
        schemas=args.schemas,
        dry_run=args.dry_run,
        date_tag=args.date_tag,
    )


if __name__ == "__main__":
    main()
