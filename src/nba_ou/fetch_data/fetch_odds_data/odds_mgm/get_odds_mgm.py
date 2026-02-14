import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.utils.general_utils import get_season_year_from_date


def _safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _pick_team_names(event: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    team_home = None
    team_away = None

    for arr_key in ("teams_normalized", "teams"):
        arr = event.get(arr_key)
        if isinstance(arr, list) and arr:
            for t in arr:
                if not isinstance(t, dict):
                    continue
                name = t.get("name")
                if t.get("is_home") is True:
                    team_home = name
                elif t.get("is_away") is True:
                    team_away = name
            if team_home or team_away:
                break

    return team_home, team_away


def build_mgm_odds_df_from_events(
    events: list[dict[str, Any]],
    west_coast_tz: str = "America/Los_Angeles",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ev in events:
        game_date_utc = pd.to_datetime(ev.get("event_date"), utc=True, errors="coerce")

        # "day converting the UTC to West coast time" -> local calendar date in PT
        # Keep as a date object (no time), which is typically what you want for naming/partitioning
        game_date_local = (
            game_date_utc.tz_convert(west_coast_tz).date()
            if pd.notna(game_date_utc)
            else None
        )

        season_year = (
            get_season_year_from_date(game_date_local) if game_date_local else None
        )
        team_home_original, team_away_original = _pick_team_names(ev)
        team_home = TEAM_NAME_STANDARDIZATION.get(
            team_home_original, team_home_original
        )
        team_away = TEAM_NAME_STANDARDIZATION.get(
            team_away_original, team_away_original
        )

        row: dict[str, Any] = {
            "game_date_captured": game_date_utc,  # UTC capture time
            "game_date": game_date_local,  # local PT calendar day
            "team_home": team_home,
            "team_away": team_away,
            "team_home_original": team_home_original,
            "team_away_original": team_away_original,
            "season_year": season_year,
            "mgm_total_line": None,
            "mgm_moneyline_home": None,
            "mgm_moneyline_away": None,
            "mgm_spread_home": None,
            "mgm_spread_away": None,
            "mgm_total_over_money": None,
            "mgm_total_under_money": None,
        }

        lines = ev.get("lines", {})
        mgm_block = None
        if isinstance(lines, dict):
            mgm_block = lines.get("22") or lines.get(22)

        if not isinstance(mgm_block, dict):
            continue

        total_over_delta = _safe_get(mgm_block, "total", "total_over_delta")
        total_under_delta = _safe_get(mgm_block, "total", "total_under_delta")
        row["mgm_total_line"] = (
            abs(total_over_delta)
            if total_over_delta is not None
            else total_under_delta
        )

        if row["mgm_total_line"] is None or row["mgm_total_line"] < 100:
            continue

        row["mgm_moneyline_home"] = _safe_get(
            mgm_block, "moneyline", "moneyline_home_delta"
        )
        row["mgm_moneyline_away"] = _safe_get(
            mgm_block, "moneyline", "moneyline_away_delta"
        )

        row["mgm_spread_home"] = _safe_get(
            mgm_block, "spread", "point_spread_home_delta"
        )
        row["mgm_spread_away"] = _safe_get(
            mgm_block, "spread", "point_spread_away_delta"
        )

        row["mgm_total_over_money"] = _safe_get(
            mgm_block, "total", "total_over_money_delta"
        )
        row["mgm_total_under_money"] = _safe_get(
            mgm_block, "total", "total_under_money_delta"
        )

        rows.append(row)

    columns = [
        "game_date_captured",
        "game_date",
        "team_home",
        "team_away",
        "team_home_original",
        "team_away_original",
        "season_year",
        "mgm_total_line",
        "mgm_moneyline_home",
        "mgm_moneyline_away",
        "mgm_spread_home",
        "mgm_spread_away",
        "mgm_total_over_money",
        "mgm_total_under_money",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    return df[columns]


def load_events_from_json(json_path: str | Path) -> list[dict[str, Any]]:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return payload["events"]

    raise ValueError(
        "Unexpected JSON structure: expected a list or a dict with an 'events' list."
    )


if __name__ == "__main__":
    # Example usage
    events = load_events_from_json(
        "/home/adrian_alvarez/gdrive/NBA_data/TheRundownApi_data/2023-03-29.json"
    )
    df_mgm = build_mgm_odds_df_from_events(events)  # PT by default
    print(df_mgm.head(1))
