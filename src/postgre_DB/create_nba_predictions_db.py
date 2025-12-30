import pandas as pd
import psycopg
from psycopg import sql

from .db_config import (
    connect_nba_db,
    get_schema_name_predictions,
)


def create_prediction_schema_if_not_exists(conn: psycopg.Connection, schema: str) -> None:
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
                        game_number_team_home INTEGER,
                        team_name_team_away TEXT,
                        game_number_team_away INTEGER,
                        matchup TEXT,
                        total_over_under_line NUMERIC,
                        average_total_over_money NUMERIC,
                        average_total_under_money NUMERIC,
                        most_common_total_over_money NUMERIC,
                        most_common_total_under_money NUMERIC,
                        predicted_total_score NUMERIC,
                        margin_difference_prediction_vs_over_under NUMERIC,
                        regressor_prediction TEXT,
                        classifier_prediction_model2 TEXT,
                        prediction_date TEXT,
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
                "GAME_NUMBER_TEAM_HOME": "game_number_team_home",
                "TEAM_NAME_TEAM_AWAY": "team_name_team_away",
                "GAME_NUMBER_TEAM_AWAY": "game_number_team_away",
                "MATCHUP": "matchup",
                "TOTAL_OVER_UNDER_LINE": "total_over_under_line",
                "PREDICTED_TOTAL_SCORE": "predicted_total_score",
                "PREDICTION_DATE": "prediction_date",
                "TIME_TO_MATCH_MINUTES": "time_to_match_minutes",
                # Your original human-readable name
                "Margin Difference Prediction vs Over/Under": "margin_difference_prediction_vs_over_under",
                "Margin_Difference_Prediction_vs_Over_Under": "margin_difference_prediction_vs_over_under",
                "Regressor Prediction": "regressor_prediction",
                "Regressor_Prediction": "regressor_prediction",
                "Classifier_Prediction_model2": "classifier_prediction_model2",
            }
        )

        # Add new columns if missing
        for new_col in [
            "average_total_over_money",
            "average_total_under_money",
            "most_common_total_over_money",
            "most_common_total_under_money",
        ]:
            if new_col not in df.columns:
                df[new_col] = None

        # Keep only DB columns (excluding id which is SERIAL)
        columns = [
            "game_id",
            "season_type",
            "game_date",
            "game_time",
            "team_name_team_home",
            "game_number_team_home",
            "team_name_team_away",
            "game_number_team_away",
            "matchup",
            "total_over_under_line",
            "average_total_over_money",
            "average_total_under_money",
            "most_common_total_over_money",
            "most_common_total_under_money",
            "predicted_total_score",
            "margin_difference_prediction_vs_over_under",
            "regressor_prediction",
            "classifier_prediction_model2",
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
