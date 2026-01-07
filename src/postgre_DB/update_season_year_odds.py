"""
One-time script to populate the season_year column in the odds table.

Logic:
- If game_date is January-July (months 1-7) → season_year = year - 1
- If game_date is August-December (months 8-12) → season_year = year

Example:
- 2024-01-15 → season_year = 2023
- 2024-11-20 → season_year = 2024
"""

from db_config import connect_nba_db, get_schema_name_odds
from psycopg import sql
from tqdm import tqdm


def update_season_year_in_odds():
    """
    Update the season_year column for all records in the odds table.
    Uses the game_date to determine the season year based on NBA season logic.
    """
    schema = get_schema_name_odds()
    table = schema  # convention: schema == table

    print("Connecting to database...")
    conn = connect_nba_db()

    try:
        with conn.cursor() as cur:
            # First, check how many records need updating
            count_query = sql.SQL("""
                SELECT COUNT(*)
                FROM {}.{}
                WHERE season_year IS NULL
            """).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(count_query)
            null_count = cur.fetchone()[0]
            print(f"Found {null_count} records with NULL/empty season_year")

            # Fetch all records with game_date
            select_query = sql.SQL("""
                SELECT game_date, team_home, team_away
                FROM {}.{}
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(select_query)
            records = cur.fetchall()
            print(f"Processing {len(records)} total records...")

            # Update in batches
            update_query = sql.SQL("""
                UPDATE {}.{}
                SET season_year = %s
                WHERE game_date = %s AND team_home = %s AND team_away = %s
            """).format(sql.Identifier(schema), sql.Identifier(table))

            updated_count = 0
            batch_size = 100
            batch = []

            for game_date, team_home, team_away in tqdm(
                records, desc="Updating season_year"
            ):
                if game_date is None:
                    continue

                # Extract year and month
                year = game_date.year
                month = game_date.month

                # Calculate season_year based on NBA season logic
                if month in [1, 2, 3, 4, 5, 6, 7]:  # January to July
                    season_year = year - 1
                else:  # August to December (8, 9, 10, 11, 12)
                    season_year = year

                batch.append((season_year, game_date, team_home, team_away))

                # Execute batch update
                if len(batch) >= batch_size:
                    cur.executemany(update_query, batch)
                    conn.commit()
                    updated_count += len(batch)
                    batch = []

            # Execute remaining records
            if batch:
                cur.executemany(update_query, batch)
                conn.commit()
                updated_count += len(batch)

            print(f"\n✅ Successfully updated {updated_count} records!")

            # Verify the update
            verify_query = sql.SQL("""
                SELECT 
                    season_year,
                    COUNT(*) as count,
                    MIN(game_date) as min_date,
                    MAX(game_date) as max_date
                FROM {}.{}
                WHERE season_year IS NOT NULL
                GROUP BY season_year
                ORDER BY season_year DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            cur.execute(verify_query)
            results = cur.fetchall()

            print("\n📊 Season Year Distribution:")
            print("-" * 70)
            print(f"{'Season Year':<15} {'Count':<10} {'Date Range':<45}")
            print("-" * 70)
            for season_year, count, min_date, max_date in results:
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                print(f"{season_year:<15} {count:<10} {date_range:<45}")

            # Check for any remaining NULL values
            cur.execute(count_query)
            remaining_nulls = cur.fetchone()[0]
            if remaining_nulls > 0:
                print(
                    f"\n⚠️ Warning: {remaining_nulls} records still have NULL season_year"
                )
            else:
                print("\n✅ All records have been updated successfully!")

    except Exception as e:
        print(f"❌ Error updating season_year: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    print("=" * 70)
    print("NBA Odds - Season Year Update Script")
    print("=" * 70)
    print("\nThis script will update the season_year column based on game_date:")
    print("  • January-July → season_year = year - 1")
    print("  • August-December → season_year = year")
    print("\n" + "=" * 70 + "\n")

    update_season_year_in_odds()
