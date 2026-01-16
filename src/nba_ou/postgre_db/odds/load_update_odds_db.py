import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_odds
from psycopg import sql


def load_odds_data(season_years: str = None) -> pd.DataFrame:
    """
    Load odds data from PostgreSQL database.
    Alias for get_existing_odds_from_db for convenience.

    Args:
        season_year (str, optional): Season year to filter by (e.g., "2024")

    Returns:
        pd.DataFrame: Odds data from database
    """
    return get_existing_odds_from_db(season_year=season_years)


def get_existing_odds_from_db(season_year: str = None) -> pd.DataFrame:
    """
    Query existing odds data from PostgreSQL database.

    Args:
        season_year (str, list, or None): Season year(s) to filter by.
            - str: Single season year (e.g., "2024")
            - list: List of season strings (e.g., ['2023-24', '2024-25'])
            - None: Load all seasons

    Returns:
        pd.DataFrame: Odds data from database
    """
    schema = get_schema_name_odds()
    table = schema  # convention: schema == table

    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            if season_year is not None:
                # Handle list of seasons (e.g., ['2023-24', '2024-25'])
                if isinstance(season_year, list):
                    season_years = [int(s.split("-")[0]) for s in season_year]
                    query = sql.SQL("""
                        SELECT *
                        FROM {}.{}
                        WHERE season_year = ANY(%s)
                        ORDER BY game_date DESC
                    """).format(sql.Identifier(schema), sql.Identifier(table))
                    cur.execute(query, (season_years,))
                else:
                    # Handle single season year string (e.g., "2024")
                    query = sql.SQL("""
                        SELECT *
                        FROM {}.{}
                        WHERE season_year = %s
                        ORDER BY game_date DESC
                    """).format(sql.Identifier(schema), sql.Identifier(table))
                    cur.execute(query, (int(season_year),))
            else:
                query = sql.SQL("""
                    SELECT *
                    FROM {}.{}
                    ORDER BY game_date DESC
                """).format(sql.Identifier(schema), sql.Identifier(table))
                cur.execute(query)

            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

        df = pd.DataFrame(rows, columns=columns)

    finally:
        conn.close()

    # Convert numeric columns
    numeric_cols = [
        "most_common_total_line",
        "average_total_line",
        "most_common_moneyline_home",
        "average_moneyline_home",
        "most_common_moneyline_away",
        "average_moneyline_away",
        "most_common_spread_home",
        "average_spread_home",
        "most_common_spread_away",
        "average_spread_away",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df)} odds records from database")
    return df


def update_odds_db(df_odds: pd.DataFrame) -> bool:
    if df_odds is None or df_odds.empty:
        print("No odds data to upload to PostgreSQL.")
        return False

    schema = get_schema_name_odds()
    table = schema  # convention: schema == table

    df_upload = df_odds.copy()

    # Optional but recommended: normalize to lowercase to match DB column names
    df_upload.columns = [c.lower() for c in df_upload.columns]

    # Drop rows where most_common_total_line or average_total_line is null/NaN
    initial_count = len(df_upload)
    df_upload = df_upload.dropna(
        subset=["most_common_total_line", "average_total_line"], how="any"
    )
    dropped_count = initial_count - len(df_upload)
    if dropped_count > 0:
        print(
            f"Dropped {dropped_count} rows with null most_common_total_line or average_total_line"
        )

    if df_upload.empty:
        print("No valid odds data to upload after filtering null values.")
        return False

    # Convert game_date to timestamp (UTC)
    if "game_date" in df_upload.columns:
        df_upload["game_date"] = pd.to_datetime(
            df_upload["game_date"], utc=True, errors="coerce"
        )

        # Calculate season_year based on game_date
        def calculate_season_year(date):
            if pd.isna(date):
                return None
            month = date.month
            year = date.year
            # January to July → season_year = year - 1
            # August to December → season_year = year
            return year - 1 if month in [1, 2, 3, 4, 5, 6, 7] else year

        df_upload["season_year"] = df_upload["game_date"].apply(calculate_season_year)

    # Convert numeric columns
    numeric_cols = [
        "most_common_total_line",
        "average_total_line",
        "most_common_moneyline_home",
        "average_moneyline_home",
        "most_common_moneyline_away",
        "average_moneyline_away",
        "most_common_spread_home",
        "average_spread_home",
        "most_common_spread_away",
        "average_spread_away",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]
    for col in numeric_cols:
        if col in df_upload.columns:
            df_upload[col] = pd.to_numeric(df_upload[col], errors="coerce")

    # Convert pandas NA to None
    df_upload = df_upload.where(pd.notna(df_upload), None)

    columns = df_upload.columns.tolist()
    values = [tuple(row) for row in df_upload.itertuples(index=False, name=None)]

    col_idents = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)

    insert_query = sql.SQL("""
        INSERT INTO {}.{} ({})
        VALUES ({})
        ON CONFLICT (game_date, team_home, team_away) DO NOTHING
    """).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        col_idents,
        placeholders,
    )

    print(f"Uploading {len(df_upload)} odds records to database...")

    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            batch_size = 1000
            for i in range(0, len(values), batch_size):
                cur.executemany(insert_query, values[i : i + batch_size])
            conn.commit()

        print(f"✅ Successfully uploaded {len(values)} odds records to database!")
        return True

    finally:
        conn.close()
