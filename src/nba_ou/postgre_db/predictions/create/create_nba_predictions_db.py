import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_predictions,
)
from psycopg import sql


def schema_exists(schema_name: str = None) -> bool:
    """Check if the predictions schema exists in the database."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_predictions()

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


def create_predictions_table():
    """
    Create the nba_predictions table inside schema SCHEMA_NAME_PREDICTIONS.
    """

    schema = get_schema_name_predictions()
    table = schema
    conn = connect_nba_db()

    try:
        create_prediction_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
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
                        model_name TEXT,
                        model_type TEXT,
                        model_version TEXT,
                        prediction_date TEXT NOT NULL,
                        time_to_match_minutes INTEGER,
                        total_scored_points NUMERIC
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            # Unique constraint (idempotent) on (game_id, prediction_date)
            cur.execute(
                sql.SQL(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conname = 'unique_game_prediction'
                        ) THEN
                            ALTER TABLE {}.{}
                            ADD CONSTRAINT unique_game_prediction UNIQUE (game_id, prediction_date);
                        END IF;
                    END $$;
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

        conn.commit()
        print(f"Table '{schema}.{table}' is ready.")
    finally:
        conn.close()


def upload_predictions_to_postgre(df: pd.DataFrame):
    """
    Insert the summary DataFrame into schema.table nba_predictions.
    Upserts on (game_id, prediction_date).
    """
    create_predictions_table()

    schema = get_schema_name_predictions()
    table = schema  # convention: schema == table

    conn = connect_nba_db()

    try:
        # Ensure required columns exist
        if "total_scored_points" not in df.columns:
            df["total_scored_points"] = None

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
                "TIME_TO_MATCH_MINUTES": "time_to_match_minutes",
            }
        )

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
            "time_to_match_minutes",
            "total_scored_points",
        ]

        # Convert pandas NA to None
        df = df.where(pd.notna(df), None)

        values = [tuple(row) for row in df[columns].itertuples(index=False, name=None)]

        placeholders = ", ".join(["%s"] * len(columns))
        update_assignments = ", ".join(
            [
                f"{col} = EXCLUDED.{col}"
                for col in columns
                if col not in ("game_id", "prediction_date")
            ]
        )

        insert_query = sql.SQL(
            f"""
            INSERT INTO {{}}.{{}} ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (game_id, prediction_date) DO UPDATE SET {update_assignments}
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))

        with conn.cursor() as cur:
            cur.executemany(insert_query, values)
        conn.commit()

    finally:
        conn.close()


if __name__ == "__main__":
    create_predictions_table()
    print("nba_predictions table is ready.")
