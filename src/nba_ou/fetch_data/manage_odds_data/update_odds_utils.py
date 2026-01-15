"""
NBA Over/Under Predictor - Odds Data Utilities

This module contains utility functions for fetching, processing, and merging
NBA betting odds data from external APIs.
"""

import os
import time
from collections import Counter

import pandas as pd
import requests
from config.constants import TEAM_NAME_STANDARDIZATION as TEAM_NAME_EQUIVALENT_DICT
from nba_api.stats.endpoints import LeagueGameFinder
from postgre_DB.db_config import connect_nba_db, connect_odds_db, get_schema_name_odds
from psycopg import sql
from tqdm import tqdm


def get_events_for_date(sport_id, date, BASE_URL, HEADERS):
    url = f"{BASE_URL}/sports/{sport_id}/events/{date}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        events = response.json().get("events", [])
        print(f"Events for sport ID {sport_id} on {date}:")
        return events

    else:
        print(f"Failed to retrieve events: {response.status_code}")
        print(response.json())
        return []


def most_common(lst):
    return Counter(lst).most_common(1)[0][0] if lst else None


def load_odds_data(season_year: str = None) -> pd.DataFrame:
    """
    Load odds data from PostgreSQL database.
    Alias for get_existing_odds_from_db for convenience.

    Args:
        season_year (str, optional): Season year to filter by (e.g., "2024")

    Returns:
        pd.DataFrame: Odds data from database
    """
    return get_existing_odds_from_db(season_year=season_year)


def get_existing_odds_from_db(season_year: str = None) -> pd.DataFrame:
    """
    Query existing odds data from PostgreSQL database.
    If season_year is provided, filters by the season_year column.
    """
    schema = get_schema_name_odds()
    table = schema  # convention: schema == table

    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            if season_year:
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


def american_to_decimal(a: float) -> float:
    a = float(a)
    if not a or a == 0:
        return None
    if a > 0:
        return 1.0 + a / 100.0
    if a < 0:
        return 1.0 + 100.0 / abs(a)


def process_odds_date(
    date: str, BASE_URL: str, HEADERS: dict, is_today=False
) -> pd.DataFrame:
    sport_id = 4
    matches = get_events_for_date(sport_id, date, BASE_URL, HEADERS)
    time.sleep(0.5)
    if len(matches) == 0:
        print(f"No matches found for {date}")
        return pd.DataFrame()

    total_field = "total_over_delta" if not is_today else "total_over"
    moneyline_home_field = "moneyline_home_delta" if not is_today else "moneyline_home"
    moneyline_away_field = "moneyline_away_delta" if not is_today else "moneyline_away"
    spread_home_field = (
        "point_spread_home_delta" if not is_today else "point_spread_home"
    )
    spread_away_field = (
        "point_spread_away_delta" if not is_today else "point_spread_away"
    )
    total_money_over_field = (
        "total_over_money_delta" if not is_today else "total_over_money"
    )
    total_money_under_field = (
        "total_under_money_delta" if not is_today else "total_under_money"
    )
    match_data = []
    for event in matches:
        if not event.get("teams"):
            continue

        team_home = next(team["name"] for team in event["teams"] if team["is_home"])
        team_away = next(team["name"] for team in event["teams"] if team["is_away"])
        event_date = pd.to_datetime(event["event_date"])

        total_lines, moneyline_home, moneyline_away, spread_home, spread_away = (
            [],
            [],
            [],
            [],
            [],
        )
        total_over_money_deltas = []
        total_under_money_deltas = []
        total_over_money_deltas_for_common = []
        total_under_money_deltas_for_common = []

        for line in event.get("lines", {}).values():
            total_info = line.get("total", {})
            moneyline_info = line.get("moneyline", {})
            spread_info = line.get("spread", {})

            if total_info and total_field in total_info:
                total_line = abs(total_info[total_field])
                if total_line > 100:
                    total_lines.append(total_line)

                    # Extract total_over_money_delta and total_under_money_delta if present, convert to decimal odds
                    if (
                        total_money_over_field in total_info
                        and abs(total_info[total_money_over_field]) > 10
                    ):
                        val = total_info[total_money_over_field]
                        val = american_to_decimal(val)
                        total_over_money_deltas.append(val)
                        total_over_money_deltas_for_common.append(val)

                    if (
                        total_money_under_field in total_info
                        and abs(total_info[total_money_under_field]) > 10
                    ):
                        val = total_info[total_money_under_field]
                        val = american_to_decimal(val)
                        total_under_money_deltas.append(val)
                        total_under_money_deltas_for_common.append(val)

            if moneyline_info:
                if moneyline_home_field in moneyline_info:
                    val = moneyline_info[moneyline_home_field]
                    if abs(val) >= 0.05:
                        moneyline_home.append(val)

                if moneyline_away_field in moneyline_info:
                    val = moneyline_info[moneyline_away_field]
                    if abs(val) >= 0.05:
                        moneyline_away.append(val)
            if spread_info:
                if spread_home_field in spread_info:
                    val = spread_info[spread_home_field]
                    if abs(val) >= 0.05:
                        spread_home.append(val)

                if spread_away_field in spread_info:
                    val = spread_info[spread_away_field]
                    if abs(val) >= 0.05:
                        spread_away.append(val)

        match_data.append(
            {
                "game_date": event_date,
                "team_home": team_home,
                "team_away": team_away,
                "most_common_total_line": most_common(total_lines),
                "average_total_line": sum(total_lines) / len(total_lines)
                if total_lines
                else None,
                "most_common_moneyline_home": most_common(moneyline_home),
                "average_moneyline_home": sum(moneyline_home) / len(moneyline_home)
                if moneyline_home
                else None,
                "most_common_moneyline_away": most_common(moneyline_away),
                "average_moneyline_away": sum(moneyline_away) / len(moneyline_away)
                if moneyline_away
                else None,
                "most_common_spread_home": most_common(spread_home),
                "average_spread_home": sum(spread_home) / len(spread_home)
                if spread_home
                else None,
                "most_common_spread_away": most_common(spread_away),
                "average_spread_away": sum(spread_away) / len(spread_away)
                if spread_away
                else None,
                "average_total_over_money": sum(total_over_money_deltas)
                / len(total_over_money_deltas)
                if total_over_money_deltas
                else None,
                "average_total_under_money": sum(total_under_money_deltas)
                / len(total_under_money_deltas)
                if total_under_money_deltas
                else None,
                "most_common_total_over_money": most_common(
                    total_over_money_deltas_for_common
                )
                if total_over_money_deltas_for_common
                else None,
                "most_common_total_under_money": most_common(
                    total_under_money_deltas_for_common
                )
                if total_under_money_deltas_for_common
                else None,
            }
        )

    return pd.DataFrame(match_data)



def update_and_get_odds_df(
    date_to_predict,
    odds_folder,
    df_name,
    season_to_download,
    ODDS_API_KEY,
    BASE_URL,
    save_csv: bool = False,
):
    HEADERS = {
        "x-rapidapi-key": ODDS_API_KEY,
        "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
    }

    # Load from database (primary source)
    season_year = season_to_download[:4]  # Extract year from season like "2024-25"
    df_odds = get_existing_odds_from_db(season_year)

    # Ensure game_date is datetime
    df_odds["game_date"] = pd.to_datetime(df_odds["game_date"])

    game_finder = LeagueGameFinder(
        season_nullable=season_to_download, league_id_nullable="00"
    )
    games_df = game_finder.get_data_frames()[0]

    unique_dates = sorted(
        pd.to_datetime(games_df["GAME_DATE"]).dt.strftime("%Y-%m-%d").unique(),
        reverse=True,
    )
    unique_dates = [
        date
        for date in unique_dates
        if date not in df_odds["game_date"].dt.strftime("%Y-%m-%d").values
    ]

    # Loop through valid dates and collect processed data
    for date in tqdm(unique_dates, desc="Processing odds per date"):
        # if date is month 10 9 or 8, skip it
        if pd.to_datetime(date).month in [8, 9, 10]:
            print(f"Skipping date {date} as it is outside the NBA season.")
            continue

        df_day = process_odds_date(date, BASE_URL, HEADERS)
        if df_day.empty:
            print(f"No data for {date}")
            continue
        # append it to df_odds
        df_odds = pd.concat([df_odds, df_day], ignore_index=True)
        df_odds.sort_values(by="game_date", inplace=True, ascending=False)
        df_odds.reset_index(drop=True, inplace=True)

    if save_csv and odds_folder:
        # Save updated odds data to CSV
        odds_path = os.path.join(odds_folder, df_name)
        df_odds.to_csv(odds_path, index=False)
        print(f"✓ Odds data saved to CSV at {odds_path}")
    # Here update after today and tomorrow are loaded
    update_odds_db(df_odds)

    df_today = process_odds_date(date_to_predict, BASE_URL, HEADERS, is_today=True)
    df_odds = pd.concat([df_odds, df_today], ignore_index=True)
    # Add tomorrow date as the API sometimes gets this in tomorrows date
    tomorrow_date = pd.to_datetime(date_to_predict) + pd.Timedelta(days=1)
    df_tomorrow = process_odds_date(
        tomorrow_date.strftime("%Y-%m-%d"), BASE_URL, HEADERS, is_today=True
    )
    df_odds = pd.concat([df_odds, df_tomorrow], ignore_index=True)
    df_odds.sort_values(by="game_date", inplace=True, ascending=False)
    df_odds.reset_index(drop=True, inplace=True)

    return df_odds


if __name__ == "__main__":
    ODDS_API_KEY = ""
    odds_folder = "/home/adrian_alvarez/Projects/NBA-predictor/odds_data/"

    df_name = "odds_data.csv"

    df = update_and_get_odds_df(
        odds_folder,
        df_name,
        ODDS_API_KEY,
        BASE_URL="https://therundown-therundown-v1.p.rapidapi.com",
    )
