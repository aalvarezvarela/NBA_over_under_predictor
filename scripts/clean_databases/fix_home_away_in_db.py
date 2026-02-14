"""
Script to fix home/away parsing errors in the games database.

This script:
1. Loads all games from the database
2. Identifies and fixes home/away parsing errors
3. Updates only the changed rows back to the database
"""

import argparse

import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_games
from psycopg import sql


def load_all_games_from_db() -> pd.DataFrame | None:
    """Load all games from the database."""
    schema = get_schema_name_games()
    table = schema

    conn = None
    try:
        conn = connect_nba_db()
        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            ORDER BY game_date DESC
        """).format(sql.Identifier(schema), sql.Identifier(table))

        query = query_obj.as_string(conn)
        df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} game records from database")
        return df

    except Exception as e:
        print(f"Error loading games from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()


def update_home_column_by_team(updates: list[tuple[str, str, bool]]) -> int:
    """
    Update the HOME column for specific (game_id, team_abbreviation) combinations.

    Args:
        updates: List of tuples (game_id, team_abbreviation, home_value)

    Returns:
        Number of rows updated
    """
    if not updates:
        return 0

    schema = get_schema_name_games()
    table = schema

    conn = connect_nba_db()
    try:
        update_query = sql.SQL("""
            UPDATE {}.{}
            SET home = %s
            WHERE game_id = %s AND team_abbreviation = %s
        """).format(sql.Identifier(schema), sql.Identifier(table))

        # Prepare data as list of tuples (home_value, game_id, team_abbr)
        update_data = [
            (bool(home_val), str(game_id), str(team_abbr))
            for game_id, team_abbr, home_val in updates
        ]

        with conn.cursor() as cur:
            cur.executemany(update_query, update_data)

        conn.commit()
        return len(update_data)

    except Exception as e:
        conn.rollback()
        print(f"Error updating database: {e}")
        import traceback

        traceback.print_exc()
        return 0

    finally:
        conn.close()


def fix_home_away_errors_in_database(dry_run: bool = True) -> dict[str, int]:
    """
    Fix home/away parsing errors in the games database.

    Args:
        dry_run: If True, only show what would be changed without updating

    Returns:
        Dictionary with statistics about the operation
    """
    # Load original data
    print("Loading games from database...")
    df_original = load_all_games_from_db()

    if df_original is None or df_original.empty:
        print("No games found in database")
        return {"total_games": 0, "rows_changed": 0, "rows_updated": 0}

    # Create a copy for fixing
    df_fixed = df_original.copy()

    # Helper function to get column name regardless of case
    def get_col(name_upper: str, df: pd.DataFrame) -> str:
        if name_upper in df.columns:
            return name_upper
        elif name_upper.lower() in df.columns:
            return name_upper.lower()
        else:
            return None

    game_id_col = get_col("GAME_ID", df_original)
    home_col = get_col("HOME", df_original)
    matchup_col = get_col("MATCHUP", df_original)
    team_abbr_col = get_col("TEAM_ABBREVIATION", df_original)

    if not all([game_id_col, home_col, matchup_col, team_abbr_col]):
        print(
            f"Required columns not found. game_id: {game_id_col}, home: {home_col}, matchup: {matchup_col}, team_abbreviation: {team_abbr_col}"
        )
        return {"total_games": len(df_original), "rows_changed": 0, "rows_updated": 0}

    # Find problematic games (where GAME_ID + HOME has duplicates)
    problematic_rows = df_fixed[
        df_fixed[[game_id_col, home_col]].duplicated(keep=False)
    ]

    if problematic_rows.empty:
        print("\nNo problematic rows found!")
        return {"total_games": len(df_original), "rows_changed": 0, "rows_updated": 0}

    print(
        f"\nFound {len(problematic_rows)} rows with potential home/away parsing errors."
    )
    print(f"Unique games affected: {problematic_rows[game_id_col].nunique()}")

    # Fix HOME column based on matchup for problematic rows
    rows_to_update = []

    for game_id in problematic_rows[game_id_col].unique():
        game_rows = df_fixed[df_fixed[game_id_col] == game_id]

        for idx in game_rows.index:
            matchup = df_fixed.loc[idx, matchup_col]
            team_abbr = df_fixed.loc[idx, team_abbr_col]

            # Determine if this team should be home based on matchup
            should_be_home = None

            if "@" in matchup:
                # Format: "AWAY @ HOME"
                home_team = matchup.split("@")[1].strip()
                should_be_home = team_abbr == home_team
            elif "vs." in matchup:
                # Format: "HOME vs. AWAY"
                home_team = matchup.split("vs.")[0].strip()
                should_be_home = team_abbr == home_team

            if should_be_home is not None:
                current_home = df_fixed.loc[idx, home_col]
                if current_home != should_be_home:
                    rows_to_update.append(
                        {
                            "index": idx,
                            "game_id": game_id,
                            "team": team_abbr,
                            "matchup": matchup,
                            "old_home": current_home,
                            "new_home": should_be_home,
                        }
                    )
                    df_fixed.loc[idx, home_col] = should_be_home

    num_changed = len(rows_to_update)
    print(f"\nFixed HOME column for {num_changed} rows.")

    if num_changed == 0:
        print("No changes needed!")
        return {"total_games": len(df_original), "rows_changed": 0, "rows_updated": 0}

    # Show sample of changes
    print("\nSample of changes:")
    sample_size = min(10, num_changed)
    for update in rows_to_update[:sample_size]:
        print(
            f"  Game {update['game_id']}: {update['team']:3} HOME {update['old_home']} -> {update['new_home']} (Matchup: {update['matchup']})"
        )

    if num_changed > sample_size:
        print(f"  ... and {num_changed - sample_size} more changes")

    # Update database if not dry run
    rows_updated = 0
    if not dry_run:
        print(f"\nUpdating {num_changed} rows in database...")
        updates = [(u["game_id"], u["team"], u["new_home"]) for u in rows_to_update]
        rows_updated = update_home_column_by_team(updates)
        print(f"Successfully updated {rows_updated} rows")
    else:
        print("\nDRY RUN: No changes made to database")
        print("Run with --apply to actually update the database")

    return {
        "total_games": len(df_original),
        "rows_changed": num_changed,
        "rows_updated": rows_updated,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix home/away parsing errors in the games database"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually apply changes to database (default is dry-run mode)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("HOME/AWAY PARSING ERROR FIX")
    print("=" * 70)

    if not args.apply:
        print("\n*** DRY RUN MODE ***")
        print("No changes will be made to the database.")
        print("Use --apply flag to actually update the database.\n")
    else:
        print("\n*** APPLY MODE ***")
        print("Changes WILL be made to the database.\n")

    results = fix_home_away_errors_in_database(dry_run=not args.apply)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total games in database: {results['total_games']}")
    print(f"Rows with changes detected: {results['rows_changed']}")
    print(f"Rows updated in database: {results['rows_updated']}")
    print("=" * 70)
