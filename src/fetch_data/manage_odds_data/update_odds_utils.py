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


def process_odds_df(df_odds, use_metric: str = "most_common"):
    """
    Process odds dataframe with option to use 'average' or 'most_common' metrics.

    Args:
        df_odds: DataFrame with odds data
        use_metric: Either "average" or "most_common" (default: "average")
    """
    # Ensure game_date is datetime (handle both timezone-aware and naive datetimes)
    if not pd.api.types.is_datetime64_any_dtype(df_odds["game_date"]):
        df_odds["game_date"] = pd.to_datetime(df_odds["game_date"], utc=True)
        df_odds["game_date"] = df_odds["game_date"].dt.tz_localize(None)

    df_odds = df_odds[df_odds["most_common_total_line"] > 50]

    df_odds.sort_values(by=["game_date", "team_home"], ascending=False, inplace=True)
    df_odds = df_odds.drop_duplicates(subset=["game_date", "team_home"], keep="first")

    df_odds["game_date_adjusted"] = df_odds["game_date"].copy()

    df_odds = df_odds.sort_values(by="game_date", ascending=False)

    mask = df_odds["game_date_adjusted"].dt.hour < 6
    df_odds.loc[mask, "game_date_adjusted"] = df_odds.loc[
        mask, "game_date_adjusted"
    ] - pd.Timedelta(days=1)
    df_odds["game_date"] = pd.to_datetime(df_odds["game_date"]).dt.strftime("%Y-%m-%d")
    df_odds["game_date_adjusted"] = pd.to_datetime(
        df_odds["game_date_adjusted"]
    ).dt.strftime("%Y-%m-%d")

    df_odds["team_home"] = df_odds["team_home"].map(TEAM_NAME_EQUIVALENT_DICT)
    df_odds["team_away"] = df_odds["team_away"].map(TEAM_NAME_EQUIVALENT_DICT)

    # Determine which columns to use based on use_metric parameter
    prefix = use_metric if use_metric in ["average", "most_common"] else "average"

    # Step 2: Rename df_new columns to match
    df_odds_renamed = df_odds.rename(
        columns={
            "game_date": "GAME_DATE_ORIGINAL",
            "game_date_adjusted": "GAME_DATE",
            "team_home": "TEAM_NAME_TEAM_HOME",
            "team_away": "TEAM_NAME_TEAM_AWAY",
            f"{prefix}_total_line": "TOTAL_OVER_UNDER_LINE",
            f"{prefix}_moneyline_home": "MONEYLINE_HOME",
            f"{prefix}_moneyline_away": "MONEYLINE_AWAY",
            f"{prefix}_spread_home": "SPREAD_HOME",
        }
    )

    # Step 3: Reorder both dataframes to the same column order
    final_columns = [
        "GAME_DATE",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "MONEYLINE_HOME",
        "MONEYLINE_AWAY",
        "TOTAL_OVER_UNDER_LINE",
        "SPREAD_HOME",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]

    df_odds_processed = df_odds_renamed[final_columns + ["GAME_DATE_ORIGINAL"]]

    df_odds_processed["SPREAD_AWAY"] = -df_odds_processed["SPREAD_HOME"]

    return df_odds_processed


def merge_teams_df_with_odds(df_odds, df_team, use_metric: str = "most_common"):
    """
    Merge team dataframe with odds data.

    Args:
        df_odds: Odds dataframe
        df_team: Team dataframe
        use_metric: Either "average" or "most_common" (default: "average")
    """
    df_odds = process_odds_df(df_odds, use_metric=use_metric)

    df_team["GAME_DATE"] = pd.to_datetime(df_team["GAME_DATE"], format="%Y-%m-%d")

    df_team["TEAM_NAME"] = df_team["TEAM_NAME"].map(
        lambda x: TEAM_NAME_EQUIVALENT_DICT.get(x, x)
    )

    df_odds[df_odds.duplicated(subset=["GAME_DATE", "TEAM_NAME_TEAM_HOME"], keep=False)]

    # Find duplicated rows based on GAME_DATE and TEAM_NAME_TEAM_HOME
    dupes_mask = df_odds.duplicated(
        subset=["GAME_DATE", "TEAM_NAME_TEAM_HOME"], keep=False
    )

    # Replace GAME_DATE with GAME_DATE_ORIGINAL only in those duplicated rows
    df_odds.loc[dupes_mask, "GAME_DATE"] = df_odds.loc[dupes_mask, "GAME_DATE_ORIGINAL"]

    df_odds[df_odds.duplicated(subset=["GAME_DATE", "TEAM_NAME_TEAM_AWAY"], keep=False)]
    dupes_mask = df_odds.duplicated(
        subset=["GAME_DATE", "TEAM_NAME_TEAM_AWAY"], keep=False
    )

    # Replace GAME_DATE with GAME_DATE_ORIGINAL only in those duplicated rows
    df_odds.loc[dupes_mask, "GAME_DATE"] = df_odds.loc[dupes_mask, "GAME_DATE_ORIGINAL"]

    df_team["GAME_DATE"] = pd.to_datetime(df_team["GAME_DATE"])
    df_odds["GAME_DATE"] = pd.to_datetime(df_odds["GAME_DATE"])
    df_odds["GAME_DATE_ORIGINAL"] = pd.to_datetime(df_odds["GAME_DATE_ORIGINAL"])

    # Home teams from df_odds
    df_home = df_odds[
        [
            "GAME_DATE",
            "TEAM_NAME_TEAM_HOME",
            "MONEYLINE_HOME",
            "TOTAL_OVER_UNDER_LINE",
            "SPREAD_HOME",
            "GAME_DATE_ORIGINAL",
            "average_total_over_money",
            "average_total_under_money",
            "most_common_total_over_money",
            "most_common_total_under_money",
        ]
    ].copy()

    df_home = df_home.rename(
        columns={
            "TEAM_NAME_TEAM_HOME": "TEAM_NAME",
            "MONEYLINE_HOME": "MONEYLINE",
            "SPREAD_HOME": "SPREAD",
        }
    )
    df_home["HOME"] = True

    # Away teams from df_odds
    df_away = df_odds[
        [
            "GAME_DATE",
            "TEAM_NAME_TEAM_AWAY",
            "MONEYLINE_AWAY",
            "TOTAL_OVER_UNDER_LINE",
            "SPREAD_AWAY",
            "GAME_DATE_ORIGINAL",
            "average_total_over_money",
            "average_total_under_money",
            "most_common_total_over_money",
            "most_common_total_under_money",
        ]
    ].copy()

    df_away = df_away.rename(
        columns={
            "TEAM_NAME_TEAM_AWAY": "TEAM_NAME",
            "MONEYLINE_AWAY": "MONEYLINE",
            "SPREAD_AWAY": "SPREAD",
        }
    )
    df_away["HOME"] = False

    # Combined betting data for "first-pass" merge

    # 1) Prepare df_betting (same as before)
    df_betting = pd.concat([df_home, df_away], ignore_index=True)

    # Also prepare a version keyed on GAME_DATE_ORIGINAL
    df_betting_original = df_betting.drop(columns="GAME_DATE").rename(
        columns={"GAME_DATE_ORIGINAL": "GAME_DATE"}
    )

    # 2) Merge #1: on GAME_DATE
    df_team1 = df_team.merge(
        df_betting.drop(columns="GAME_DATE_ORIGINAL", errors="ignore"),
        how="left",
        on=["GAME_DATE", "TEAM_NAME", "HOME"],
        suffixes=("", "_betting"),
    )

    # 3) Merge #2: on GAME_DATE_ORIGINAL
    df_team2 = df_team.merge(
        df_betting_original,
        how="left",
        on=["GAME_DATE", "TEAM_NAME", "HOME"],
        suffixes=("", "_betting"),
    )

    # 4) Coalesce columns of interest
    df_team_final = df_team1.copy()

    for col in [
        "MONEYLINE",
        "SPREAD",
        "TOTAL_OVER_UNDER_LINE",
        "average_total_over_money",
        "average_total_under_money",
        "most_common_total_over_money",
        "most_common_total_under_money",
    ]:
        # If the first merge is missing data, fill from the second merge
        df_team_final[col] = df_team1[col].fillna(df_team2[col])

    mask_missing_tot = df_team_final["TOTAL_OVER_UNDER_LINE"].isna()
    df_missing_tot = df_team_final[mask_missing_tot].copy()

    # 2) Merge ignoring HOME, which can produce duplicates
    df_missing_tot_merged = df_missing_tot.merge(
        df_betting, how="left", on=["GAME_DATE", "TEAM_NAME"], suffixes=("", "_inv")
    )

    # 3) Now condense df_missing_tot_merged so that each (GAME_DATE, TEAM_NAME)
    #    appears at most once. We'll just pick the first valid TOT found:
    df_missing_tot_merged = (
        df_missing_tot_merged.dropna(
            subset=["TOTAL_OVER_UNDER_LINE_inv"]
        )  # keep rows that actually have TOT
        .drop_duplicates(subset=["GAME_DATE", "TEAM_NAME"], keep="first")
        .copy()
    )

    # 4) Rename so we only keep one TOT column
    df_missing_tot_merged["TOTAL_OVER_UNDER_LINE_new"] = df_missing_tot_merged[
        "TOTAL_OVER_UNDER_LINE"
    ].fillna(df_missing_tot_merged["TOTAL_OVER_UNDER_LINE_inv"])

    # 5) Merge that condensed info back into df_team_final
    df_team_final = df_team_final.merge(
        df_missing_tot_merged[["GAME_DATE", "TEAM_NAME", "TOTAL_OVER_UNDER_LINE_new"]],
        how="left",
        on=["GAME_DATE", "TEAM_NAME"],
    )

    # 6) Fill the original TOT where it was missing
    df_team_final["TOTAL_OVER_UNDER_LINE"] = df_team_final[
        "TOTAL_OVER_UNDER_LINE"
    ].fillna(df_team_final["TOTAL_OVER_UNDER_LINE_new"])

    # 7) Drop the helper column
    df_team_final.drop(columns=["TOTAL_OVER_UNDER_LINE_new"], inplace=True)

    # drop duplicates
    df_team_final.drop_duplicates(
        subset=["GAME_DATE", "TEAM_NAME", "HOME"], keep=False, inplace=True
    )

    return df_team_final


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
