import psycopg
from typing import TYPE_CHECKING
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
)
from psycopg import sql

if TYPE_CHECKING:
    import pandas as pd

PREDICTIONS_SCHEMA = "nba_predictions"
PREDICTIONS_TABLE = "nba_predictions"


def get_predictions_schema_and_table() -> tuple[str, str]:
    """Return the canonical predictions schema/table names."""
    return PREDICTIONS_SCHEMA, PREDICTIONS_TABLE


def schema_exists(schema_name: str = None) -> bool:
    """Check if the predictions schema exists in the database."""
    try:
        if schema_name is None:
            schema_name, _ = get_predictions_schema_and_table()

        conn = connect_nba_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            exists = cur.fetchone()
        conn.close()
        return exists is not None
    except Exception as e:
        print(f"Error checking schema existence: {e}")
        return False


def create_prediction_schema_if_not_exists(
    conn: psycopg.Connection, schema: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_predictions_table(drop: bool = False):
    """
    Create table nba_predictions inside schema nba_predictions.

    Args:
        drop (bool): If True, drops the existing table before creating it. Default: False
    """

    schema, table = get_predictions_schema_and_table()
    conn = connect_nba_db()

    try:
        create_prediction_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            # Drop table if requested
            if drop:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
                print(f"Dropped existing table '{schema}.{table}'")

            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {}.{} (
                        id SERIAL PRIMARY KEY,
                        game_id TEXT NOT NULL,
                        season_type TEXT,
                        game_date DATE,
                        game_time TEXT,
                        team_name_team_home TEXT,
                        team_name_team_away TEXT,
                        total_over_under_line NUMERIC,
                        pred_line_error NUMERIC NOT NULL,
                        pred_total_points NUMERIC NOT NULL,
                        pred_pick TEXT,
                        model_name TEXT NOT NULL,
                        model_type TEXT,
                        model_version TEXT,
                        prediction_date TEXT,
                        prediction_datetime TIMESTAMP NOT NULL,
                        time_to_match_minutes INTEGER,
                        na_columns_count INTEGER,
                        na_columns_names TEXT,
                        total_scored_points NUMERIC,
                        home_pts NUMERIC,
                        away_pts NUMERIC
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            # Forward-compatible migration for existing tables
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS na_columns_count INTEGER"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS na_columns_names TEXT"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            # Unique constraint on (game_id, model_name, prediction_datetime)
            # Always drop and recreate constraint to ensure it has correct columns
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} DROP CONSTRAINT IF EXISTS unique_game_prediction"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {}.{}
                    ADD CONSTRAINT unique_game_prediction 
                    UNIQUE (game_id, model_name, prediction_datetime)
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

        conn.commit()
        print(f"Table '{schema}.{table}' is ready.")
    finally:
        conn.close()


def upload_predictions_to_postgre(df: "pd.DataFrame"):
    """
    Insert the summary DataFrame into schema.table nba_predictions.
    Upserts on (game_id, model_name, prediction_datetime).
    """
    import pandas as pd

    create_predictions_table()

    schema, table = get_predictions_schema_and_table()

    conn = connect_nba_db()

    try:
        # Normalize column names to match DB (lowercase in DB)
        df = df.rename(
            columns={
                "GAME_ID": "game_id",
                "SEASON_TYPE": "season_type",
                "GAME_DATE": "game_date",
                "GAME_TIME": "game_time",
                "TEAM_NAME_TEAM_HOME": "team_name_team_home",
                "TEAM_NAME_TEAM_AWAY": "team_name_team_away",
                "TOTAL_OVER_UNDER_LINE": "total_over_under_line",
                "PRED_LINE_ERROR": "pred_line_error",
                "PRED_TOTAL_POINTS": "pred_total_points",
                "PRED_PICK": "pred_pick",
                "MODEL_NAME": "model_name",
                "MODEL_TYPE": "model_type",
                "MODEL_VERSION": "model_version",
                "PREDICTION_DATE": "prediction_date",
                "PREDICTION_DATETIME": "prediction_datetime",
                "TIME_TO_MATCH_MINUTES": "time_to_match_minutes",
                "NA_COLUMNS_COUNT": "na_columns_count",
                "NA_COLUMNS_NAMES": "na_columns_names",
                "HOME_PTS": "home_pts",
                "AWAY_PTS": "away_pts",
            }
        )

        # If both uppercase and lowercase versions exist, keep the first non-null value.
        duplicate_cols = df.columns[df.columns.duplicated()].unique().tolist()
        for col in duplicate_cols:
            dup_values = df.loc[:, df.columns == col]
            df = df.loc[:, df.columns != col]
            df[col] = dup_values.bfill(axis=1).iloc[:, 0]

        # Ensure optional enrichment columns exist after normalization.
        for col in (
            "na_columns_count",
            "na_columns_names",
            "total_scored_points",
            "home_pts",
            "away_pts",
        ):
            if col not in df.columns:
                df[col] = None

        # Keep only DB columns (excluding id which is SERIAL)
        columns = [
            "game_id",
            "season_type",
            "game_date",
            "game_time",
            "team_name_team_home",
            "team_name_team_away",
            "total_over_under_line",
            "pred_line_error",
            "pred_total_points",
            "pred_pick",
            "model_name",
            "model_type",
            "model_version",
            "prediction_date",
            "prediction_datetime",
            "time_to_match_minutes",
            "na_columns_count",
            "na_columns_names",
            "total_scored_points",
            "home_pts",
            "away_pts",
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for upload: {missing_columns}")

        # Convert pandas NA to None
        df = df.where(pd.notna(df), None)

        values = [tuple(row) for row in df[columns].itertuples(index=False, name=None)]
        if not values:
            return

        conflict_columns = ("game_id", "model_name", "prediction_datetime")
        update_assignments = [
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
            for col in columns
            if col not in conflict_columns
        ]

        insert_query = sql.SQL(
            """
            INSERT INTO {}.{} ({})
            VALUES ({})
            ON CONFLICT ({}) DO UPDATE SET {}
            """
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            sql.SQL(", ").join(map(sql.Identifier, conflict_columns)),
            sql.SQL(", ").join(update_assignments),
        )

        with conn.cursor() as cur:
            cur.executemany(insert_query, values)
        conn.commit()

    finally:
        conn.close()


if __name__ == "__main__":
    create_predictions_table(True)
    print("nba_predictions table is ready.")
