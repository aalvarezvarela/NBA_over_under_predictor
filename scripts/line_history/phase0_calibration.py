"""Phase 0 calibration for the SBR line-history -> Aiven migration.

Read-only. Resolves the three unknowns that must be settled before any data is
loaded:

a) **Timezone.** The scraped ``timestamp`` column is timezone-naive. The zone is
   recovered in two steps: first identify which daylight-saving calendar the
   clock follows (EU and US flip on different dates, so they are separable),
   then read off the winter/summer offsets. The closing line of each
   (game, book, market) is the clock, because a pre-game market stops moving at
   tipoff. See :func:`calibrate_timezone`.
b) **game_id match rate.** SBR ``event_id`` is mapped to NBA ``game_id`` via
   (game_date, team_home, team_away). Every row loaded to Aiven must carry a
   ``game_id``, so the unmatched share is a gate, not a footnote.
c) **Tipoff coverage.** Every matched game needs a tipoff to compute
   ``mins_to_tip``.

Usage::

    python scripts/line_history/phase0_calibration.py
    python scripts/line_history/phase0_calibration.py --seasons 2024-25 2025-26
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    _normalize_sbr_team_name,
    build_games_home_away_for_line_history,
    load_games_for_line_history_creation,
)

DEFAULT_ROOT = Path("data/sbr_line_history")
CANDIDATE_TIMEZONES = ["Europe/Madrid", "UTC", "America/New_York", "Europe/London"]

READ_COLUMNS = [
    "game_date",
    "season_year",
    "event_id",
    "team_home",
    "team_away",
    "bookmaker_slug",
    "market",
    "timestamp",
    "row_kind",
]


def season_label_to_year(label: str) -> int:
    """``"2024-25"`` -> ``2024``."""
    return int(str(label).split("-")[0])


def load_season_line_history(root: Path, season_label: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(root / season_label / "line_history" / "*.csv")))
    if not paths:
        return pd.DataFrame(columns=READ_COLUMNS)

    frames = []
    for path in paths:
        try:
            frames.append(
                pd.read_csv(path, usecols=READ_COLUMNS, parse_dates=["timestamp"])
            )
        except pd.errors.EmptyDataError:
            continue

    if not frames:
        return pd.DataFrame(columns=READ_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    return df


def resolve_game_ids(
    line_history: pd.DataFrame,
    games_home_away: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``game_id`` via (game_date, team_home, team_away)."""
    out = line_history.copy()
    out["team_home"] = out["team_home"].map(_normalize_sbr_team_name)
    out["team_away"] = out["team_away"].map(_normalize_sbr_team_name)
    return out.merge(
        games_home_away[["game_id", "game_date", "team_home", "team_away"]],
        on=["game_date", "team_home", "team_away"],
        how="left",
    )


def closing_lines(matched: pd.DataFrame) -> pd.DataFrame:
    """Last recorded timestamp per (game, book, market), joined to tipoff."""
    last = (
        matched.groupby(["game_id", "bookmaker_slug", "market"], as_index=False)[
            "timestamp"
        ]
        .max()
        .merge(
            matched[["game_id", "tipoff_utc"]].drop_duplicates("game_id"),
            on="game_id",
            how="left",
        )
    )
    return last.dropna(subset=["tipoff_utc"])


def calibrate_timezone(matched: pd.DataFrame) -> dict:
    """Recover the timezone the naive ``timestamp`` column was rendered in.

    Two independent measurements, neither of which assumes anything about how
    late a book keeps moving a line:

    1. **Which DST calendar does the clock follow?** Treating the naive value as
       UTC, the closing-line offset is measured separately inside and outside
       each of the EU and US daylight-saving windows. A ~+1h step on one
       calendar and ~0 on the other identifies the continent. (EU flips on the
       last Sundays of March/October, the US on the second Sunday of March and
       first Sunday of November, so the two are separable.)
    2. **What is the absolute offset?** The median closing-line offset under a
       UTC assumption *is* the zone's offset, because a pre-game closing line
       sits at tipoff. Winter and summer offsets are read off separately.

    Together these pin the zone exactly: EU calendar + (+1h winter, +2h summer)
    is CET/CEST.
    """
    last = closing_lines(matched)
    delta = (
        last["timestamp"].dt.tz_localize("UTC") - last["tipoff_utc"]
    ).dt.total_seconds() / 3600.0
    scoped = last.assign(delta=delta)
    scoped = scoped[scoped["delta"].between(-6, 6)]
    if len(scoped) < 200:
        return {"n": int(len(scoped))}

    tipoff = scoped["tipoff_utc"]
    naive_tip = tipoff.dt.tz_localize(None)
    eu_dst = (
        tipoff.dt.tz_convert("Europe/Madrid").dt.tz_localize(None) - naive_tip
    ).dt.total_seconds() / 3600.0 > 1.5
    us_dst = (
        naive_tip - tipoff.dt.tz_convert("America/New_York").dt.tz_localize(None)
    ).dt.total_seconds() / 3600.0 < 4.5

    def median_where(mask: pd.Series) -> float:
        values = scoped["delta"][mask]
        return float(values.median()) if len(values) > 50 else float("nan")

    eu_on, eu_off = median_where(eu_dst), median_where(~eu_dst)
    us_on, us_off = median_where(us_dst), median_where(~us_dst)

    return {
        "n": int(len(scoped)),
        "eu_step": round(eu_on - eu_off, 2),
        "us_step": round(us_on - us_off, 2),
        "winter_offset": round(eu_off, 2),
        "summer_offset": round(eu_on, 2),
    }


def verify_zone(matched: pd.DataFrame, tz: str) -> dict:
    """Residual closing-line offset after converting from ``tz``; want ~0."""
    localized = matched["timestamp"].dt.tz_localize(
        tz, ambiguous="NaT", nonexistent="NaT"
    )
    last = closing_lines(
        matched.assign(timestamp=localized).dropna(subset=["timestamp"])
    )
    delta = (
        last["timestamp"].dt.tz_convert("UTC") - last["tipoff_utc"]
    ).dt.total_seconds() / 3600.0
    scoped = delta[delta.between(-6, 6)]
    if scoped.empty:
        return {"tz": tz, "n": 0}
    return {
        "tz": tz,
        "n": int(len(scoped)),
        "p50": round(float(scoped.median()), 2),
        "score": abs(round(float(scoped.median()), 2)),
        "ambiguous": int(localized.isna().sum() - matched["timestamp"].isna().sum()),
    }


def analyze_season(
    season_label: str,
    root: Path,
    games_home_away: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict:
    raw = load_season_line_history(root, season_label)
    if raw.empty:
        return {"season": season_label, "rows": 0}

    resolved = resolve_game_ids(raw, games_home_away)
    matched = resolved.dropna(subset=["game_id"]).merge(
        schedule[["game_id", "tipoff_utc"]], on="game_id", how="left"
    )

    dst = calibrate_timezone(matched)
    scores = [verify_zone(matched, tz) for tz in CANDIDATE_TIMEZONES]
    ranked = sorted((s for s in scores if s.get("n")), key=lambda s: s["score"])
    best_tz = ranked[0]["tz"] if ranked else None

    # In-play share under the winning zone: these rows are a leakage hazard.
    inplay_share = None
    ambiguous = None
    if best_tz:
        localized = matched["timestamp"].dt.tz_localize(
            best_tz, ambiguous="NaT", nonexistent="NaT"
        )
        mins = (
            localized.dt.tz_convert("UTC") - matched["tipoff_utc"]
        ).dt.total_seconds() / 60.0
        inplay_share = round(float((mins >= 0).mean()), 4)
        ambiguous = int(localized.isna().sum() - matched["timestamp"].isna().sum())

    unmatched = resolved[resolved["game_id"].isna()]
    no_tipoff = matched[matched["tipoff_utc"].isna()]

    return {
        "dst": dst,
        "unmatched_prefixes": unmatched["event_id"].nunique(),
        "no_tipoff_games": no_tipoff["game_id"].nunique(),
        "no_tipoff_ids": sorted(no_tipoff["game_id"].dropna().unique())[:5],
        "season": season_label,
        "rows": int(len(raw)),
        "dedup_rows": int(
            len(
                raw.drop_duplicates(
                    ["event_id", "bookmaker_slug", "market", "timestamp"]
                )
            )
        ),
        "games": int(raw["event_id"].nunique()),
        "match_rate": round(float(resolved["game_id"].notna().mean()), 4),
        "unmatched_games": int(
            resolved[resolved["game_id"].isna()]["event_id"].nunique()
        ),
        "tipoff_coverage": round(float(matched["tipoff_utc"].notna().mean()), 4),
        "best_tz": best_tz,
        "inplay_share": inplay_share,
        "ambiguous_dst": ambiguous,
        "tz_scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seasons", nargs="*", default=None)
    args = parser.parse_args()

    root: Path = args.root
    seasons = args.seasons or sorted(
        p.name for p in root.iterdir() if p.is_dir() and "-" in p.name
    )

    print(f"Seasons: {', '.join(seasons)}\n")

    print("Fetching NBA schedules (tipoff source)...")
    schedule = fetch_schedules([season_label_to_year(s) for s in seasons])
    print(
        f"  {len(schedule)} games, {schedule['tipoff_utc'].notna().sum()} with tipoff\n"
    )

    print("Loading games from Postgres...")
    games_home_away = build_games_home_away_for_line_history(
        load_games_for_line_history_creation()
    )
    print(f"  {len(games_home_away)} home/away pairs\n")

    results = [
        analyze_season(season, root, games_home_away, schedule) for season in seasons
    ]

    print("=" * 78)
    print("(b) game_id match rate + (c) tipoff coverage")
    print("=" * 78)
    header = f"{'season':>9} {'rows':>9} {'dedup':>9} {'games':>6} {'match':>7} {'unmatched':>10} {'tipoff':>7}"
    print(header)
    for r in results:
        if not r.get("rows"):
            print(f"{r['season']:>9}  (empty)")
            continue
        print(
            f"{r['season']:>9} {r['rows']:>9,} {r['dedup_rows']:>9,} {r['games']:>6} "
            f"{r['match_rate']:>7.2%} {r['unmatched_games']:>10} {r['tipoff_coverage']:>7.2%}"
        )

    print()
    print("=" * 78)
    print("(a1) which DST calendar does the clock follow?")
    print("     naive ts treated as UTC; ~+1h step on one calendar identifies it")
    print("=" * 78)
    print(
        f"{'season':>9} {'EU step':>8} {'US step':>8} {'winter off':>11} {'summer off':>11}"
    )
    for r in results:
        d = r.get("dst") or {}
        if not d.get("n"):
            continue
        print(
            f"{r['season']:>9} {d['eu_step']:>+8.2f} {d['us_step']:>+8.2f} "
            f"{d['winter_offset']:>+11.2f} {d['summer_offset']:>+11.2f}"
        )

    print()
    print("=" * 78)
    print("(a2) residual closing-line offset after conversion (want p50 ~ 0.00)")
    print("=" * 78)
    for r in results:
        if not r.get("rows"):
            continue
        print(f"\n  {r['season']}  ->  best = {r['best_tz']}")
        for s in r["tz_scores"]:
            if not s.get("n"):
                print(f"    {s['tz']:>18}  (insufficient data)")
                continue
            flag = "  <== " if s["tz"] == r["best_tz"] else "      "
            print(
                f"    {s['tz']:>18}  p50={s['p50']:>+6.2f} h  "
                f"ambiguous={s['ambiguous']:>4}{flag}"
            )

    print()
    print("=" * 78)
    print("in-play exposure (rows at or after tipoff, under the winning zone)")
    print("  these are LEAKAGE for any pre-game feature -- filter on mins_to_tip < 0")
    print("=" * 78)
    print(f"{'season':>9} {'in-play share':>14} {'ambiguous DST rows':>20}")
    for r in results:
        if not r.get("rows") or r["inplay_share"] is None:
            continue
        print(f"{r['season']:>9} {r['inplay_share']:>13.2%} {r['ambiguous_dst']:>20,}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    # The DST calendar is the primary evidence: it is structural, and unlike the
    # closing-line level it cannot be moved by how sparse a season's data is.
    confident = [
        r
        for r in results
        if (r.get("dst") or {}).get("n") and abs(r["dst"]["eu_step"] - 1.0) < 0.25
    ]
    unclear = [r for r in results if r.get("rows") and r not in confident]

    if confident:
        winter = sum(r["dst"]["winter_offset"] for r in confident) / len(confident)
        summer = sum(r["dst"]["summer_offset"] for r in confident) / len(confident)
        print(
            f"  {len(confident)} season(s) show a ~+1h step on the EU calendar: "
            f"{', '.join(r['season'] for r in confident)}"
        )
        print(
            f"  implied offsets: winter {winter:+.2f}h, summer {summer:+.2f}h "
            "-> CET/CEST -> Europe/Madrid"
        )
    if unclear:
        print(
            f"  LOW CONFIDENCE (sparse / no clean DST step): "
            f"{', '.join(r['season'] for r in unclear)}"
        )
        print("  -> calibrate these per season before loading, or load them last.")
    print("=" * 78)


if __name__ == "__main__":
    os.environ.setdefault("DB_ENV", "supabase")
    main()
