from pathlib import Path
import pickle
import json
import re

# -----------------------------
# Paths
# -----------------------------
SRC_DIR = Path(
    "/home/adrian_alvarez/Projects/NBA-predictor/all_data/raw_odds_data"
)

DST_DIR = Path.home() / "gdrive" / "NBA_data" / "TheRundownApi_data"
DST_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Filename pattern
# -----------------------------
# Example: matches_2025-04-07.pkl
PATTERN = re.compile(r"matches_(?P<date>\d{4}-\d{2}-\d{2})\.pkl$")

# -----------------------------
# Processing
# -----------------------------
for pkl_path in SRC_DIR.glob("matches_*.pkl"):
    match = PATTERN.match(pkl_path.name)
    if not match:
        print(f"Skipping (unexpected name): {pkl_path.name}")
        continue

    date_str = match.group("date")
    out_path = DST_DIR / f"{date_str}.json"

    # Load pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Atomic write (safer on network mounts)
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.rename(out_path)

    print(f"✓ {pkl_path.name} → {out_path}")
