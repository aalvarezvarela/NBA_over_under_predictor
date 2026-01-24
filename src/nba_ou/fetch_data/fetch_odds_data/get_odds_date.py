import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests


def get_events_for_date(
    sport_id: int,
    date: str,
    base_url: str,
    headers: dict[str, str],
    timeout: int = 10,
    retry_delay: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Fetch events for a given sport and date.

    Behaviour:
    - 200 → return events
    - 429 → retry once, then raise
    - any other non-200 → raise
    - network error → raise
    """
    url = f"{base_url}/sports/{sport_id}/events/{date}"

    for attempt in (1, 2):  # initial attempt + exactly one retry
        try:
            response = requests.get(url, headers=headers, timeout=timeout)

            if response.status_code == 200:
                return response.json().get("events", [])

            if response.status_code == 429:
                if attempt == 1:
                    time.sleep(retry_delay)
                    continue
                raise RuntimeError(
                    f"Rate limited (429) after retry for sport_id={sport_id}, date={date}"
                )

            # Any other HTTP error
            response.raise_for_status()

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Request failed for sport_id={sport_id}, date={date}"
            ) from exc

    # Defensive: this should never be reached
    raise RuntimeError("Unreachable state in get_events_for_date")


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


def save_matches_pickle(
    matches: Any,
    date: str,
    pickle_path: str,
    prefix: str = "odds_matches",
    strict: bool = True,
) -> Optional[str]:
    """
    Save matches object to a dated pickle file.

    Parameters
    ----------
    matches : Any
        Object to serialize.
    date : str
        Date identifier used in filename.
    pickle_path : str
        Target directory.
    prefix : str, optional
        Filename prefix, by default "odds_matches".
    strict : bool, optional
        If True (default), raise on failure.
        If False, log warning and return None.

    Returns
    -------
    Optional[str]
        File path on success, None on failure when strict=False.

    Raises
    ------
    RuntimeError
        If strict=True and saving fails.
    """
    filename = f"{str(date)}_{prefix}.pkl"
    filepath = os.path.join(pickle_path, filename)

    try:
        Path(pickle_path).mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(matches, f)

        return filepath

    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Failed to save pickle file for date={date} at {filepath}"
            ) from exc

        print(f"Warning: Failed to save pickle file for {date}: {exc}")
        return None


def process_odds_date(
    date: str,
    BASE_URL: str,
    HEADERS: dict,
    is_today=False,
    save_pickle=False,
    pickle_path=None,
) -> pd.DataFrame:
    sport_id = 4

    matches = get_events_for_date(sport_id, date, BASE_URL, HEADERS)

    time.sleep(0.25)

    if len(matches) == 0:
        print(f"No matches found for {date}")
        return pd.DataFrame()

    # Save raw matches data as pickle if configured and not today
    if save_pickle and not is_today and pickle_path:
        save_matches_pickle(
            matches, date, pickle_path, prefix="odds_matches", strict=True
        )

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
