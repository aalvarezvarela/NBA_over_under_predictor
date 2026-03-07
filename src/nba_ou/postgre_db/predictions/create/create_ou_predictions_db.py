import argparse
from typing import TYPE_CHECKING

import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_predictions,
)
from psycopg import sql

if TYPE_CHECKING:
    import pandas as pd

def get_predictions_schema_and_table() -> tuple[str, str]:
    """Return the canonical predictions schema/table names."""
    schema = get_schema_name_predictions().strip()
    table = schema
    return schema, table


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
    Create predictions table inside the configured predictions schema.

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
                        prediction_value_type TEXT NOT NULL,
                        total_over_under_line NUMERIC,
                        total_bet365_line_at_prediction NUMERIC,
                        pred_line_error NUMERIC,
                        pred_total_points NUMERIC,
                        pred_pick TEXT,
                        model_name TEXT NOT NULL,
                        model_type TEXT,
                        model_version TEXT,
                        prediction_date TEXT,
                        prediction_datetime TIMESTAMP NOT NULL,
                        time_to_match_minutes INTEGER,
                        na_columns_count INTEGER,
                        na_columns_names TEXT,
                        shap_base_value NUMERIC,
                        shap_top_positive_features TEXT,
                        shap_top_negative_features TEXT,
                        prediction_source TEXT NOT NULL DEFAULT 'unknown',
                        source_run_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        total_scored_points NUMERIC,
                        home_pts NUMERIC,
                        away_pts NUMERIC,
                        CONSTRAINT chk_prediction_value_type
                            CHECK (
                                prediction_value_type IN ('TOTAL_POINTS', 'DIFF_FROM_LINE')
                            ),
                        CONSTRAINT chk_prediction_target_present
                            CHECK (
                                pred_line_error IS NOT NULL
                                OR pred_total_points IS NOT NULL
                            ),
                        CONSTRAINT chk_pred_pick
                            CHECK (
                                pred_pick IS NULL
                                OR pred_pick IN ('OVER', 'UNDER', 'PUSH')
                            )
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            # Forward-compatible migration for existing tables
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS prediction_value_type TEXT"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS total_over_under_line NUMERIC"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS total_bet365_line_at_prediction NUMERIC"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS pred_line_error NUMERIC"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS pred_total_points NUMERIC"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS pred_pick TEXT"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ALTER COLUMN pred_line_error DROP NOT NULL"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ALTER COLUMN pred_total_points DROP NOT NULL"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

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
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS shap_base_value NUMERIC"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {}.{}
                    ADD COLUMN IF NOT EXISTS shap_top_positive_features TEXT
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {}.{}
                    ADD COLUMN IF NOT EXISTS shap_top_negative_features TEXT
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS prediction_source TEXT"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS source_run_id TEXT"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}.{}
                    SET prediction_source = 'unknown'
                    WHERE prediction_source IS NULL
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ALTER COLUMN prediction_source SET DEFAULT 'unknown'"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ALTER COLUMN prediction_source SET NOT NULL"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}.{}
                    SET prediction_value_type = 'DIFF_FROM_LINE'
                    WHERE prediction_value_type IS NULL
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ALTER COLUMN prediction_value_type SET NOT NULL"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} DROP CONSTRAINT IF EXISTS chk_prediction_value_type"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {}.{}
                    ADD CONSTRAINT chk_prediction_value_type
                    CHECK (
                        prediction_value_type IN ('TOTAL_POINTS', 'DIFF_FROM_LINE')
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            # Ensure table allows either prediction target:
            # line error OR total points (at least one must be present).
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} DROP CONSTRAINT IF EXISTS chk_prediction_target_present"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {}.{}
                    ADD CONSTRAINT chk_prediction_target_present
                    CHECK (
                        pred_line_error IS NOT NULL
                        OR pred_total_points IS NOT NULL
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            cur.execute(
                sql.SQL("ALTER TABLE {}.{} DROP CONSTRAINT IF EXISTS chk_pred_pick").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    ALTER TABLE {}.{}
                    ADD CONSTRAINT chk_pred_pick
                    CHECK (
                        pred_pick IS NULL
                        OR pred_pick IN ('OVER', 'UNDER', 'PUSH')
                    )
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            # Unique constraint on (game_id, model_name, prediction_datetime, prediction_value_type)
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
                    UNIQUE (game_id, model_name, prediction_datetime, prediction_value_type)
                    """
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )

            index_specs = [
                (f"idx_{table}_game_id", "game_id"),
                (f"idx_{table}_game_date", "game_date"),
                (f"idx_{table}_prediction_datetime", "prediction_datetime"),
                (f"idx_{table}_model_name", "model_name"),
            ]
            for index_name, column_name in index_specs:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} ({})").format(
                        sql.Identifier(index_name),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(column_name),
                    )
                )

            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {} ON {}.{} (game_id)
                    WHERE total_scored_points IS NULL OR home_pts IS NULL OR away_pts IS NULL
                    """
                ).format(
                    sql.Identifier(f"idx_{table}_pending_results"),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )

        conn.commit()
        print(f"Table '{schema}.{table}' is ready.")
    finally:
        conn.close()


def upload_predictions_to_postgre(df: "pd.DataFrame"):
    """
    Insert the summary DataFrame into the configured predictions schema/table.
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
                "PREDICTION_VALUE_TYPE": "prediction_value_type",
                "TOTAL_OVER_UNDER_LINE": "total_over_under_line",
                "TOTAL_BET365_LINE_AT_PREDICTION": "total_bet365_line_at_prediction",
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
                "SHAP_BASE_VALUE": "shap_base_value",
                "SHAP_TOP_POSITIVE_FEATURES": "shap_top_positive_features",
                "SHAP_TOP_NEGATIVE_FEATURES": "shap_top_negative_features",
                "PREDICTION_SOURCE": "prediction_source",
                "SOURCE_RUN_ID": "source_run_id",
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
            "prediction_value_type",
            "total_over_under_line",
            "total_bet365_line_at_prediction",
            "pred_line_error",
            "pred_total_points",
            "pred_pick",
            "na_columns_count",
            "na_columns_names",
            "shap_base_value",
            "shap_top_positive_features",
            "shap_top_negative_features",
            "prediction_source",
            "source_run_id",
            "total_scored_points",
            "home_pts",
            "away_pts",
        ):
            if col not in df.columns:
                df[col] = None

        df["prediction_value_type"] = df["prediction_value_type"].fillna(
            "DIFF_FROM_LINE"
        )
        df["prediction_source"] = (
            df["prediction_source"]
            .fillna(df["model_type"])
            .fillna("unknown")
            .astype(str)
            .str.strip()
        )
        df.loc[df["prediction_source"] == "", "prediction_source"] = "unknown"
        source_run_id_fallback = (
            df["model_name"].astype(str).fillna("unknown")
            + "|"
            + df["prediction_datetime"].astype(str).fillna("unknown")
            + "|"
            + df["prediction_value_type"].astype(str).fillna("unknown")
        )
        df["source_run_id"] = df["source_run_id"].fillna(source_run_id_fallback)

        # Keep only DB columns (excluding id which is SERIAL)
        columns = [
            "game_id",
            "season_type",
            "game_date",
            "game_time",
            "team_name_team_home",
            "team_name_team_away",
            "prediction_value_type",
            "total_over_under_line",
            "total_bet365_line_at_prediction",
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
            "shap_base_value",
            "shap_top_positive_features",
            "shap_top_negative_features",
            "prediction_source",
            "source_run_id",
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

        conflict_columns = (
            "game_id",
            "model_name",
            "prediction_datetime",
            "prediction_value_type",
        )
        update_assignments = [
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
            for col in columns
            if col not in conflict_columns
        ]
        update_assignments.append(sql.SQL("updated_at = NOW()"))

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


def recreate_predictions_table() -> None:
    """Drop and recreate predictions table with the latest schema."""
    create_predictions_table(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or recreate predictions table in configured schema"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate table from scratch",
    )
    args = parser.parse_args()

    if args.recreate:
        recreate_predictions_table()
    else:
        create_predictions_table(False)
    
    schema, table = get_predictions_schema_and_table()
    print(f"{schema}.{table} table is ready.")
