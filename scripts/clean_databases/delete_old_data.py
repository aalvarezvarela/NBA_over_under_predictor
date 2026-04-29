"""
Delete old data from Supabase schemas to save storage space.

This script removes historical records based on season_year thresholds:
- nba_games: deletes data older than 2015 (season_year < 2015)
- nba_players: deletes data older than 2018 (season_year < 2018)

Usage:
    python scripts/clean_databases/delete_old_data.py                    # interactive mode
    python scripts/clean_databases/delete_old_data.py --dry-run          # preview what will be deleted
    python scripts/clean_databases/delete_old_data.py --yes              # skip confirmation prompts
    python scripts/clean_databases/delete_old_data.py --schema nba_games # delete from specific schema only
"""

from __future__ import annotations

import argparse
from typing import Literal

from nba_ou.postgre_db.config.db_config import connect_nba_db
from psycopg import sql

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SchemaType = Literal["nba_games", "nba_players"]

DELETION_RULES: dict[SchemaType, dict] = {
    "nba_games": {
        "table": "nba_games",
        "min_season_year": 2014,  # Delete < 2014 (i.e., 2013, 2012, ...)
        "description": "NBA games data",
    },
    "nba_players": {
        "table": "nba_players",
        "min_season_year": 2018,  # Delete < 2018 (i.e., 2017, 2016, ...)
        "description": "NBA player stats",
    },
}


# ---------------------------------------------------------------------------
# Count and deletion functions
# ---------------------------------------------------------------------------


def count_old_records(
    schema: SchemaType,
    min_season_year: int,
    table: str,
) -> tuple[int, int, list[int]]:
    """
    Count records that will be deleted and kept.

    Returns:
        (old_count, total_count, seasons_to_delete)
    """
    conn = connect_nba_db()

    try:
        with conn.cursor() as cur:
            # Count records to delete
            count_query = sql.SQL(
                """
                SELECT COUNT(*), array_agg(DISTINCT season_year ORDER BY season_year)
                FROM {}.{}
                WHERE season_year < %s
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(count_query, (min_season_year,))
            result = cur.fetchone()
            old_count = result[0] if result else 0
            seasons = result[1] if result and result[1] else []

            # Count total records
            total_query = sql.SQL(
                """
                SELECT COUNT(*)
                FROM {}.{}
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(total_query)
            total_count = cur.fetchone()[0]

        return old_count, total_count, seasons

    finally:
        conn.close()


def delete_old_records(
    schema: SchemaType,
    min_season_year: int,
    table: str,
) -> int:
    """
    Delete records older than min_season_year.

    Returns:
        Number of rows deleted.
    """
    conn = connect_nba_db()

    try:
        with conn.cursor() as cur:
            delete_query = sql.SQL(
                """
                DELETE FROM {}.{}
                WHERE season_year < %s
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(delete_query, (min_season_year,))
            deleted_count = cur.rowcount

        conn.commit()
        return deleted_count

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main cleanup routine
# ---------------------------------------------------------------------------


def cleanup_old_data(
    schemas: list[SchemaType] | None = None,
    *,
    dry_run: bool = False,
    skip_confirmation: bool = False,
) -> dict[SchemaType, int]:
    """
    Delete old data from the specified schemas.

    Returns:
        Dict mapping schema name to number of rows deleted.
    """
    schemas = schemas or list(DELETION_RULES.keys())
    results: dict[SchemaType, int] = {}

    print("=" * 70)
    print("Database Cleanup: Delete Old Data")
    print("=" * 70)

    if dry_run:
        print("\n🔍 DRY RUN MODE: No data will be deleted\n")

    # Step 1: Gather statistics
    print("Analyzing current data...\n")

    stats: dict[SchemaType, tuple[int, int, list[int]]] = {}
    for schema in schemas:
        rule = DELETION_RULES[schema]
        table = rule["table"]
        min_year = rule["min_season_year"]
        desc = rule["description"]

        try:
            old_count, total_count, seasons = count_old_records(schema, min_year, table)
            stats[schema] = (old_count, total_count, seasons)

            kept_count = total_count - old_count
            pct = (old_count / total_count * 100) if total_count > 0 else 0

            print(f"📊 {schema}.{table} ({desc})")
            print(f"   Total records:      {total_count:,}")
            print(f"   Records to delete:  {old_count:,} ({pct:.1f}%)")
            print(f"   Records to keep:    {kept_count:,}")
            print(f"   Cutoff:             season_year < {min_year}")

            if seasons:
                seasons_str = ", ".join(str(s) for s in seasons)
                print(f"   Seasons to delete:  {seasons_str}")
            else:
                print(f"   Seasons to delete:  (none)")

            print()

        except Exception as e:
            print(f"   ⚠️  Error analyzing {schema}: {e}\n")
            continue

    if dry_run:
        print("─" * 70)
        print("✓ Dry run complete. No data was deleted.")
        return {}

    # Step 2: Confirm deletion
    total_to_delete = sum(s[0] for s in stats.values())

    if total_to_delete == 0:
        print("─" * 70)
        print("✓ No old data found. Nothing to delete.")
        return {}

    if not skip_confirmation:
        print("─" * 70)
        print(f"⚠️  WARNING: You are about to delete {total_to_delete:,} records.")
        print()
        response = input("Do you want to proceed? (yes/no): ").strip().lower()

        if response not in ["yes", "y"]:
            print("\n❌ Deletion cancelled.")
            return {}

    # Step 3: Delete data
    print("\n" + "─" * 70)
    print("Deleting old data...\n")

    for schema in schemas:
        if schema not in stats:
            continue

        old_count = stats[schema][0]
        if old_count == 0:
            print(f"⏭️  {schema}: No records to delete")
            continue

        rule = DELETION_RULES[schema]
        table = rule["table"]
        min_year = rule["min_season_year"]

        try:
            print(f"🗑️  Deleting from {schema}.{table}...", end=" ", flush=True)
            deleted = delete_old_records(schema, min_year, table)
            print(f"✓ Deleted {deleted:,} rows")
            results[schema] = deleted

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("Cleanup Complete")
    print("=" * 70)

    total_deleted = sum(results.values())
    print(f"\n✓ Successfully deleted {total_deleted:,} records")

    for schema, count in results.items():
        print(f"  - {schema}: {count:,} rows")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete old data from Supabase to save storage space"
    )
    parser.add_argument(
        "--schema",
        choices=list(DELETION_RULES.keys()),
        default=None,
        help="Clean up a specific schema only (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (use with caution)",
    )

    args = parser.parse_args()

    schemas = [args.schema] if args.schema else None

    cleanup_old_data(
        schemas=schemas,
        dry_run=args.dry_run,
        skip_confirmation=args.yes,
    )


if __name__ == "__main__":
    main()
