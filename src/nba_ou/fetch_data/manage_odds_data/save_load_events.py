"""
NBA Over/Under Predictor - Save and Load Events

This module provides utilities to save and load raw event data from the odds API
as pickle files for caching and offline analysis.
"""

import os
import pickle
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from update_odds_utils import get_events_for_date


def save_events_for_date(
    sport_id: int,
    date: str,
    output_folder: str,
    BASE_URL: str,
    HEADERS: dict,
    filename: str = None,
) -> str:
    """
    Fetch events for a specific date and save them as a pickle file.

    Args:
        sport_id: Sport ID (4 for NBA)
        date: Date in format 'YYYY-MM-DD'
        output_folder: Directory where pickle file will be saved
        BASE_URL: Base URL for the API
        HEADERS: API headers including authentication
        filename: Optional custom filename (without extension)

    Returns:
        Path to the saved pickle file

    Example:
        >>> HEADERS = {
        ...     "x-rapidapi-key": "your_api_key",
        ...     "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
        ... }
        >>> BASE_URL = "https://therundown-therundown-v1.p.rapidapi.com"
        >>> path = save_events_for_date(4, "2025-01-15", "./events", BASE_URL, HEADERS)
    """
    # Fetch events from API
    events = get_events_for_date(sport_id, date, BASE_URL, HEADERS)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        filename = f"events_{date}_sport{sport_id}"

    filepath = os.path.join(output_folder, f"{filename}.pkl")

    # Save events as pickle
    with open(filepath, "wb") as f:
        pickle.dump(events, f)

    print(f"✓ Saved {len(events)} events to {filepath}")
    return filepath


def load_events_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load events from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        List of event dictionaries

    Example:
        >>> events = load_events_from_file("./events/events_2025-01-15_sport4.pkl")
        >>> print(f"Loaded {len(events)} events")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        events = pickle.load(f)

    print(f"✓ Loaded {len(events)} events from {filepath}")
    return events


def save_multiple_dates(
    sport_id: int,
    dates: List[str],
    output_folder: str,
    BASE_URL: str,
    HEADERS: dict,
) -> List[str]:
    """
    Fetch and save events for multiple dates.

    Args:
        sport_id: Sport ID (4 for NBA)
        dates: List of dates in format 'YYYY-MM-DD'
        output_folder: Directory where pickle files will be saved
        BASE_URL: Base URL for the API
        HEADERS: API headers including authentication

    Returns:
        List of paths to saved pickle files
    """
    saved_files = []
    for date in dates:
        try:
            filepath = save_events_for_date(
                sport_id, date, output_folder, BASE_URL, HEADERS
            )
            saved_files.append(filepath)
        except Exception as e:
            print(f"✗ Error saving events for {date}: {e}")

    return saved_files


def most_common(lst):
    return Counter(lst).most_common(1)[0][0] if lst else None


def american_to_decimal(a: float) -> float:
    a = float(a)
    if not a or a == 0:
        return None
    if a > 0:
        return 1.0 + a / 100.0
    if a < 0:
        return 1.0 + 100.0 / abs(a)


def process_odds_date(
    matches,
    is_today: bool = False,
) -> pd.DataFrame:
 
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


if __name__ == "__main__":
    # Example usage
    import configparser

    # Load API key from config
    config = configparser.ConfigParser()
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config.secrets.ini"
    )
    config.read(config_path)

    ODDS_API_KEY = config.get("Odds", "ODDS_API_KEY", fallback="")

    if not ODDS_API_KEY:
        print("Error: ODDS_API_KEY not found in config.secrets.ini")
        exit(1)

    HEADERS = {
        "x-rapidapi-key": ODDS_API_KEY,
        "x-rapidapi-host": "therundown-therundown-v1.p.rapidapi.com",
    }
    BASE_URL = "https://therundown-therundown-v1.p.rapidapi.com"

    # Example: Save events for today
    date = "2023-01-30"
    output_folder = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/events_cache"
    )

    print(f"Fetching events for {date}...")
    filepath = save_events_for_date(4, date, output_folder, BASE_URL, HEADERS)

    # Load them back
    print("\nLoading events back...")
    filepath = "/home/adrian_alvarez/Projects/NBA-predictor/all_data/raw_odds_data/matches_2023-02-04.pkl"
    filepath = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/events_cache/events_2025-12-27_sport4.pkl"
    events = load_events_from_file(filepath)
    print(f"Successfully loaded {len(events)} events")
    events_df = process_odds_date(events, is_today=True)


for event in events:
    print(
        f"- {event.get('teams', 'Unknown Teams')} on {event.get('commence_time', 'Unknown Time')}"
    )
