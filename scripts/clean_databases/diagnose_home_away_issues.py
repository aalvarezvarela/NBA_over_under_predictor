"""
Diagnostic script to understand home/away parsing errors in the games database.
"""

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


def diagnose_problematic_games():
    """Analyze the structure and issues in the games table."""
    df = load_all_games_from_db()

    if df is None or df.empty:
        print("No data to analyze")
        return

    print("\n" + "=" * 70)
    print("TABLE STRUCTURE")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nColumn types:\n{df.dtypes}")

    # Helper function to get column name regardless of case
    def get_col(name_upper: str) -> str:
        if name_upper in df.columns:
            return name_upper
        elif name_upper.lower() in df.columns:
            return name_upper.lower()
        else:
            return None

    game_id_col = get_col("GAME_ID")
    home_col = get_col("HOME")
    matchup_col = get_col("MATCHUP")
    team_abbr_col = get_col("TEAM_ABBREVIATION")

    print("\n" + "=" * 70)
    print("COLUMN MAPPING")
    print("=" * 70)
    print(f"game_id column: {game_id_col}")
    print(f"home column: {home_col}")
    print(f"matchup column: {matchup_col}")
    print(f"team_abbreviation column: {team_abbr_col}")

    if not game_id_col:
        print("\nERROR: No game_id column found!")
        return

    # Check for duplicate game_ids
    print("\n" + "=" * 70)
    print("GAME_ID ANALYSIS")
    print("=" * 70)
    unique_games = df[game_id_col].nunique()
    total_rows = len(df)
    print(f"Unique game_ids: {unique_games}")
    print(f"Total rows: {total_rows}")
    print(f"Rows per game (avg): {total_rows / unique_games:.2f}")

    if total_rows > unique_games:
        print("\n→ This appears to be TEAM-LEVEL data (multiple rows per game)")
    else:
        print("\n→ This appears to be GAME-LEVEL data (one row per game)")

    if home_col:
        print("\n" + "=" * 70)
        print("PROBLEMATIC GAMES ANALYSIS")
        print("=" * 70)

        # Find games where (GAME_ID, HOME) has duplicates
        problematic = df[df[[game_id_col, home_col]].duplicated(keep=False)]

        if problematic.empty:
            print(
                "No problematic rows found (no duplicate GAME_ID + HOME combinations)"
            )
        else:
            print(
                f"Found {len(problematic)} rows with duplicate (GAME_ID, HOME) combinations"
            )
            print(f"Unique games affected: {problematic[game_id_col].nunique()}")

            # Show some examples
            print("\nExample problematic games:")
            sample_game_ids = problematic[game_id_col].unique()[:5]

            cols_to_show = [game_id_col, home_col]
            if matchup_col:
                cols_to_show.append(matchup_col)
            if team_abbr_col:
                cols_to_show.append(team_abbr_col)

            for gid in sample_game_ids:
                print(f"\n  Game ID: {gid}")
                game_rows = df[df[game_id_col] == gid][cols_to_show]
                for _, row in game_rows.iterrows():
                    print(f"    {dict(row)}")

    if matchup_col and team_abbr_col:
        print("\n" + "=" * 70)
        print("MATCHUP FORMAT ANALYSIS")
        print("=" * 70)

        # Sample some matchups
        sample_matchups = df[[matchup_col, team_abbr_col, home_col]].head(20)
        print("\nSample matchups:")
        for _, row in sample_matchups.iterrows():
            matchup = row[matchup_col]
            team = row[team_abbr_col]
            home = row[home_col]

            # Determine expected home team from matchup
            if "@" in matchup:
                expected_home_team = matchup.split("@")[1].strip()
                is_home = team == expected_home_team
            elif "vs." in matchup:
                expected_home_team = matchup.split("vs.")[0].strip()
                is_home = team == expected_home_team
            else:
                is_home = "?"

            status = "✓" if is_home == home else "✗"
            print(
                f"  {status} {matchup:20} | Team: {team:3} | HOME: {home} | Expected: {is_home}"
            )


if __name__ == "__main__":
    diagnose_problematic_games()
