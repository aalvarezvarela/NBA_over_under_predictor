import pandas as pd

from .db_config import connect_postgres_db, connect_predictions_db, get_config


def create_predictions_database():
    """Create the predictions database if it doesn't exist."""
    db_name = get_predictions_db_name()
    conn = connect_postgres_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,)
    )
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Database '{db_name}' created successfully!")
    else:
        print(f"Database '{db_name}' already exists.")
    cursor.close()
    conn.close()


def get_predictions_db_name():
    config = get_config()
    return config.get("Database", "DB_NAME_PREDICTIONS")


DB_NAME = get_predictions_db_name()


def create_predictions_table():
    """Create the nba_predictions table if it doesn't exist."""
    create_predictions_database()
    conn = connect_predictions_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_predictions (
            id SERIAL PRIMARY KEY,
            GAME_ID TEXT NOT NULL,
            SEASON_TYPE TEXT,
            GAME_DATE DATE,
            GAME_TIME TEXT,
            TEAM_NAME_TEAM_HOME TEXT,
            GAME_NUMBER_TEAM_HOME INTEGER,
            TEAM_NAME_TEAM_AWAY TEXT,
            GAME_NUMBER_TEAM_AWAY INTEGER,
            MATCHUP TEXT,
            TOTAL_OVER_UNDER_LINE NUMERIC,
            average_total_over_money NUMERIC,
            average_total_under_money NUMERIC,
            most_common_total_over_money NUMERIC,
            most_common_total_under_money NUMERIC,
            PREDICTED_TOTAL_SCORE NUMERIC,
            Margin_Difference_Prediction_vs_Over_Under NUMERIC,
            Regressor_Prediction TEXT,
            Classifier_Prediction_model2 TEXT,
            PREDICTION_DATE TEXT,
            TIME_TO_MATCH_MINUTES INTEGER,
            total_scored_points NUMERIC
        )
        """
    )
    # Ensure unique constraint exists (idempotent)
    cursor.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'unique_game_prediction'
            ) THEN
                ALTER TABLE nba_predictions ADD CONSTRAINT unique_game_prediction UNIQUE (GAME_ID, PREDICTION_DATE);
            END IF;
        END$$;
    """)
    conn.commit()
    cursor.close()
    conn.close()


def insert_predictions(df: pd.DataFrame):
    """Insert the summary DataFrame into nba_predictions table."""
    create_predictions_database()
    conn = connect_predictions_db()
    cursor = conn.cursor()
    # Add empty total_scored_points column if not present
    if "total_scored_points" not in df.columns:
        df["total_scored_points"] = None
    # Rename columns to match DB
    df = df.rename(
        columns={
            "Margin Difference Prediction vs Over/Under": "Margin_Difference_Prediction_vs_Over_Under",
            "Regressor Prediction": "Regressor_Prediction",
            "Classifier_Prediction_model2": "Classifier_Prediction_model2",
        }
    )
    # Add new columns if missing, default to None
    for new_col in [
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]:
        if new_col not in df.columns:
            df[new_col] = None

    # Only keep columns that exist in the table
    columns = [
        "GAME_ID",
        "SEASON_TYPE",
        "GAME_DATE",
        "GAME_TIME",
        "TEAM_NAME_TEAM_HOME",
        "GAME_NUMBER_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "GAME_NUMBER_TEAM_AWAY",
        "MATCHUP",
        "TOTAL_OVER_UNDER_LINE",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
        "PREDICTED_TOTAL_SCORE",
        "Margin_Difference_Prediction_vs_Over_Under",
        "Regressor_Prediction",
        "Classifier_Prediction_model2",
        "PREDICTION_DATE",
        "TIME_TO_MATCH_MINUTES",
        "total_scored_points",
    ]
    # Insert rows
    for _, row in df[columns].iterrows():
        values = tuple(row)
        placeholders = ", ".join(["%s"] * len(columns))
        update_assignments = ", ".join(
            [
                f"{col} = EXCLUDED.{col}"
                for col in columns
                if col not in ["GAME_ID", "PREDICTION_DATE", "id"]
            ]
        )
        insert_query = f"""
        INSERT INTO nba_predictions ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (GAME_ID, PREDICTION_DATE) DO UPDATE SET {update_assignments}
        """
        cursor.execute(insert_query, values)
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    create_predictions_table()
    print("nba_predictions table is ready.")
