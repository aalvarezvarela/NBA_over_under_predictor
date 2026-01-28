from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

json_dir = Path(
    "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/TheRundownApi_data"
)


def iter_json_files(folder: Path) -> Iterable[Path]:
    # Common patterns: *.json OR date-named files without extension, etc.
    # Adjust if needed.
    yield from sorted(folder.glob("*.json"))


def safe_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_events_from_payload(payload: Any) -> Iterable[Dict[str, Any]]:
    """
    Accepts:
      - a list of event dicts
      - a single event dict
      - any other structure (ignored)
    """
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "event_id" in item:
                yield item
    elif isinstance(payload, dict) and "event_id" in payload:
        yield payload


def extract_affiliates_from_event(event: Dict[str, Any]) -> Iterable[Tuple[int, str]]:
    """
    Returns unique (affiliate_id, affiliate_name) pairs present in this event.
    We look under event["lines"][...]["affiliate"].
    """
    seen: set[Tuple[int, str]] = set()

    lines = event.get("lines")
    if not isinstance(lines, dict):
        return seen

    for _, line_obj in lines.items():
        if not isinstance(line_obj, dict):
            continue
        aff = line_obj.get("affiliate")
        if not isinstance(aff, dict):
            continue

        aff_id = aff.get("affiliate_id")
        aff_name = aff.get("affiliate_name")

        # Filter out missing/invalid
        if isinstance(aff_id, bool):  # bool is subclass of int
            continue
        if not isinstance(aff_id, int):
            # Sometimes affiliate_id could be a string; try to coerce
            try:
                aff_id = int(aff_id)
            except Exception:
                continue

        if not isinstance(aff_name, str) or not aff_name.strip():
            continue

        seen.add((aff_id, aff_name.strip()))

    return seen


def compute_affiliate_presence_stats(folder: Path) -> None:
    files = list(iter_json_files(folder))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder}")

    total_events = 0
    events_with_any_lines = 0
    events_with_any_affiliate = 0

    # Presence counts: count an affiliate at most once per event
    affiliate_event_presence: Counter[Tuple[int, str]] = Counter()

    # Diagnostics
    file_errors: list[tuple[Path, str]] = []
    events_missing_lines = 0

    for fp in files:
        try:
            payload = safe_load_json(fp)
        except Exception as e:
            file_errors.append((fp, f"{type(e).__name__}: {e}"))
            continue

        for event in iter_events_from_payload(payload):
            total_events += 1

            lines = event.get("lines")
            if isinstance(lines, dict) and len(lines) > 0:
                events_with_any_lines += 1
            else:
                events_missing_lines += 1

            affiliates = extract_affiliates_from_event(event)
            if affiliates:
                events_with_any_affiliate += 1
                for pair in affiliates:
                    affiliate_event_presence[pair] += 1

    if total_events == 0:
        print("No events found across files.")
        return

    def pct(n: int, d: int) -> float:
        return (n / d) * 100.0 if d else 0.0

    print("=== Summary ===")
    print(f"JSON folder: {folder}")
    print(f"Files scanned: {len(files)}")
    if file_errors:
        print(f"Files failed to parse: {len(file_errors)}")
        for fp, err in file_errors[:10]:
            print(f"  - {fp.name}: {err}")
        if len(file_errors) > 10:
            print(f"  ... (+{len(file_errors) - 10} more)")

    print(f"Total events: {total_events}")
    print(
        f"Events with any 'lines' dict: {events_with_any_lines} "
        f"({pct(events_with_any_lines, total_events):.2f}%)"
    )
    print(
        f"Events with >=1 affiliate present: {events_with_any_affiliate} "
        f"({pct(events_with_any_affiliate, total_events):.2f}%)"
    )
    print(
        f"Events missing/empty 'lines': {events_missing_lines} "
        f"({pct(events_missing_lines, total_events):.2f}%)"
    )

    print("\n=== Affiliate presence across events (counted once per event) ===")
    print("Sorted by % of events that include the affiliate.\n")

    rows = []
    for (aff_id, aff_name), n_events_present in affiliate_event_presence.items():
        rows.append(
            (pct(n_events_present, total_events), n_events_present, aff_id, aff_name)
        )

    rows.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    # Print top N (adjust as needed)
    top_n = 50
    print(f"Top {min(top_n, len(rows))} affiliates:")
    for p, n, aff_id, aff_name in rows[:top_n]:
        print(f"  - {aff_name} (id={aff_id}): {n}/{total_events} events ({p:.2f}%)")

    # Optional: show how many unique affiliates you have
    print(f"\nUnique affiliates found: {len(rows)}")


if __name__ == "__main__":
    compute_affiliate_presence_stats(json_dir)
