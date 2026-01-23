"""
Cleanup script for NBA players data:
- Excludes Preseason and All Star games based on SEASON_TYPE_MAP game_id prefixes.
- Aggregates total minutes per (game_id, team_id).
- Removes team/game rows with total minutes below the threshold.
"""

import argparse

import pandas as pd
import psycopg

from nba_ou.config.constants import REGULATION_GAME_MINUTES, SEASON_TYPE_MAP
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_players
from psycopg import sql


def _excluded_prefixes() -> list[str]:
    excluded_types = {"Preseason", "All Star"}
    return [prefix for prefix, name in SEASON_TYPE_MAP.items() if name in excluded_types]


def _parse_minutes(min_series: pd.Series) -> pd.Series:
    def parse_value(value) -> float:
        if pd.isna(value) or value == "":
            return 0.0
        value_str = str(value)
        if ":" in value_str:
            parts = value_str.split(":")
            if len(parts) == 2:
                minutes_part = parts[0].strip()
                seconds_part = parts[1].strip()
                try:
                    minutes_val = float(minutes_part)
                except ValueError:
                    return 0.0
                if seconds_part.isdigit():
                    return minutes_val + int(seconds_part) / 60.0
            return 0.0
        try:
            return float(value_str)
        except ValueError:
            return 0.0

    return min_series.apply(parse_value)


def fetch_players_df(
    conn: psycopg.Connection,
    season_year: int | None,
) -> pd.DataFrame:
    schema = get_schema_name_players()
    table = schema

    query = sql.SQL(
        """
        SELECT game_id, team_id, min, season_year
        FROM {}.{}
        """
    ).format(sql.Identifier(schema), sql.Identifier(table))

    params: list[object] = []
    if season_year is not None:
        query = query + sql.SQL(" WHERE season_year = %s")
        params.append(season_year)

    return pd.read_sql_query(query.as_string(conn), conn, params=params)


def find_invalid_game_ids(
    df: pd.DataFrame,
    min_minutes: int,
    excluded_prefixes: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "total_minutes"])

    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)

    if excluded_prefixes:
        excluded_mask = df["game_id"].str.startswith(tuple(excluded_prefixes))
        df = df.loc[~excluded_mask]

    if df.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "total_minutes"])

    df["min_numeric"] = _parse_minutes(df["min"])
    team_minutes = (
        df.groupby(["game_id", "team_id"], as_index=False)["min_numeric"]
        .sum()
        .rename(columns={"min_numeric": "total_minutes"})
    )

    return team_minutes.loc[team_minutes["total_minutes"] < min_minutes]


def _count_player_rows_for_game_ids(
    conn: psycopg.Connection,
    game_ids: list[str],
    season_year: int | None,
) -> int:
    if not game_ids:
        return 0

    schema = get_schema_name_players()
    table = schema

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_invalid_games (
                game_id TEXT
            ) ON COMMIT DROP
            """
        )
        cur.executemany(
            "INSERT INTO tmp_invalid_games (game_id) VALUES (%s)",
            [(game_id,) for game_id in game_ids],
        )
        query = sql.SQL(
            """
            SELECT COUNT(*)
            FROM {}.{} p
            JOIN tmp_invalid_games t
              ON p.game_id = t.game_id
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))
        params: list[object] = []
        if season_year is not None:
            query = query + sql.SQL(" WHERE p.season_year = %s")
            params.append(season_year)
        cur.execute(query, params)
        return cur.fetchone()[0]


def delete_players_for_game_ids(
    conn: psycopg.Connection,
    game_ids: list[str],
    season_year: int | None,
) -> int:
    if not game_ids:
        return 0

    schema = get_schema_name_players()
    table = schema

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_invalid_games (
                game_id TEXT
            ) ON COMMIT DROP
            """
        )
        cur.executemany(
            "INSERT INTO tmp_invalid_games (game_id) VALUES (%s)",
            [(game_id,) for game_id in game_ids],
        )
        query = sql.SQL(
            """
            DELETE FROM {}.{} p
            USING tmp_invalid_games t
            WHERE p.game_id = t.game_id
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))
        params: list[object] = []
        if season_year is not None:
            query = query + sql.SQL(" AND p.season_year = %s")
            params.append(season_year)
        cur.execute(query, params)
        deleted = cur.rowcount

    conn.commit()
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove player rows for team-games with total minutes below a threshold. "
            "Excludes Preseason and All Star games based on SEASON_TYPE_MAP."
        )
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=REGULATION_GAME_MINUTES-5,
        help=f"Minimum total minutes per team (default: {REGULATION_GAME_MINUTES-5})",
    )
    parser.add_argument(
        "--dry-mode",
        action="store_true",
        default=True,
        help="Preview invalid team-games without deleting rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max rows to display in dry mode (default: 25)",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=2022,
        help="Season year to filter (default: 2025)",
    )
    args = parser.parse_args()

    excluded_prefixes = _excluded_prefixes()
    print(f"Excluded prefixes (season types): {excluded_prefixes}")

    conn = None
    try:
        conn = connect_nba_db()
        df_players = fetch_players_df(conn, args.season_year)
        invalid_rows = find_invalid_game_ids(
            df_players, args.min_minutes, excluded_prefixes
        )

        if invalid_rows.empty:
            print("No team-games found below the minutes threshold.")
            return 0

        invalid_game_ids = sorted(invalid_rows["game_id"].unique().tolist())
        total_players = _count_player_rows_for_game_ids(
            conn, invalid_game_ids, args.season_year
        )
        conn.rollback()

        print(
            f"Found {len(invalid_rows)} team-games below {args.min_minutes} minutes "
            f"across {len(invalid_game_ids)} games ({total_players} player rows)."
        )

        if args.dry_mode:
            print("\nPreview (game_id, team_id, total_minutes):")
            preview_rows = invalid_rows.head(args.limit)
            for _, row in preview_rows.iterrows():
                print(
                    f"{row['game_id']} | {row['team_id']} | {row['total_minutes']:.2f}"
                )
            if len(invalid_rows) > args.limit:
                print(f"... {len(invalid_rows) - args.limit} more rows not shown")
            print("\nDry mode enabled: no deletions executed.")
            return 0

        deleted = delete_players_for_game_ids(
            conn, invalid_game_ids, args.season_year
        )
        print(f"Deleted {deleted} player rows.")
        return 0

    except Exception as exc:
        print(f"Error cleaning players data: {exc}")
        if conn is not None:
            conn.rollback()
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
