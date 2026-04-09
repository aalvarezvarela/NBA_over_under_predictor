import re
from pathlib import Path

from nba_ou.modeling.meta_learner_training_data import (
    build_meta_learner_training_data_from_csv,
)

PROJECT_ROOT = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/"
TRAIN_DATA_DIR = Path(PROJECT_ROOT) / "data" / "train_data"
INPUT_CSV_PATH = (
    Path(PROJECT_ROOT)
    / "data"
    / "train_data"
    / "all_odds_training_data_until_20260405.csv"
)
LAST_N_SEASONS = 3
TOP_N_FEATURES_PER_MODEL = 50
DROP_ROWS_MISSING_ANY_PREDICTION = True
SHOW_PROGRESS = True


def _extract_date_from_filename(csv_path: Path) -> str | None:
    """Extract date (YYYYMMDD) from filename like 'all_odds_training_data_until_20260405.csv'."""
    match = re.search(r"(\d{8})", csv_path.stem)
    return match.group(1) if match else None


def _resolve_input_csv_path() -> Path:
    """Get the absolute path of the input CSV and verify it exists."""
    csv_path = INPUT_CSV_PATH.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    return csv_path


def main() -> None:
    csv_path = _resolve_input_csv_path()

    # Extract date from filename and include it in output path
    date_str = _extract_date_from_filename(csv_path)
    if date_str:
        output_path = TRAIN_DATA_DIR / f"meta_learner_last_{LAST_N_SEASONS}_seasons_{date_str}.csv"
    else:
        output_path = TRAIN_DATA_DIR / f"meta_learner_last_{LAST_N_SEASONS}_seasons.csv"

    print(f"Input CSV: {csv_path}")
    print(f"Output CSV: {output_path}")

    result = build_meta_learner_training_data_from_csv(
        csv_path=csv_path,
        last_n_seasons=LAST_N_SEASONS,
        top_n_features_per_model=TOP_N_FEATURES_PER_MODEL,
        drop_rows_missing_any_prediction=DROP_ROWS_MISSING_ANY_PREDICTION,
        output_path=output_path,
        show_progress=SHOW_PROGRESS,
    )

    print(f"Prediction seasons: {result.prediction_seasons}")
    print(f"Selected feature count: {len(result.selected_feature_names)}")
    print(f"Prediction columns: {result.prediction_columns}")
    print(f"Output rows: {len(result.dataframe)}")
    if result.output_path is not None:
        print(f"Saved CSV: {result.output_path}")


if __name__ == "__main__":
    main()
