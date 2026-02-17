from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import (
    create_df_to_predict,
)
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    load_and_predict_tabpfn_client_for_nba_games,
)

from scripts.update_databases.update_all_databases import update_all_databases


def print_banner(date_to_predict: str) -> None:
    line = "=" * 70
    title = "NBA OVER/UNDER TABPFN CLIENT PREDICTION"
    print(f"\n{line}")
    print(title.center(len(line)))
    print(line)
    print(f"  Prediction Date: {date_to_predict}")
    print(line + "\n")


def print_step_header(step_number: int, title: str) -> None:
    sep = "-" * 70
    header = f" STEP {step_number} — {title} "
    print(sep)
    print(header.center(len(sep)))
    print(sep)


def print_status(message: str, ok: bool = True) -> None:
    symbol = "✓" if ok else "✖"
    print(f"  {symbol} {message}")


def predict_nba_games_tabpfn() -> None:
    date_to_predict = datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
    prediction_date = datetime.now(ZoneInfo("Europe/Madrid"))

    print_banner(date_to_predict)

    # Step 1: Update database
    print_step_header(1, "Updating All Databases")
    season_to_update = str(date_to_predict)[:4]
    try:
        update_all_databases(
            start_season_year=int(season_to_update),
            end_season_year=int(season_to_update),
            only_new_games=True,
            headless=True,
        )
        print_status("Databases updated")
    except Exception as e:
        print_status(f"Failed to update databases: {e}", ok=False)
        raise

    # Step 2: Scheduled data
    print_step_header(2, "Fetching Scheduled Games & Reports")
    try:
        scheduled_data = get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            save_reports_path=SETTINGS.report_path,
        )
        print_status("Fetched scheduled games, refs, injuries and odds")
        if scheduled_data["scheduled_games"].empty:
            print_status(
                "⚠️ WARNING: No scheduled games found for the specified date.", ok=False
            )
            return
    except Exception as e:
        print_status(f"Failed to fetch scheduled data: {e}", ok=False)
        raise

    # Step 3: Build features
    print_step_header(3, "Preparing Feature DataFrame")
    try:
        df_to_predict = create_df_to_predict(
            todays_prediction=True,
            scheduled_data=scheduled_data,
            recent_limit_to_include=date_to_predict,
            strict_mode=True,
        )
        print_status("Feature DataFrame prepared")
    except Exception as e:
        print_status(f"Failed to prepare features: {e}", ok=False)
        raise

    if df_to_predict.empty:
        raise ValueError("df_to_predict is empty")

    print_status(f"Found {len(df_to_predict)} game(s) to predict")

    # Step 4: TabPFN inference + upload
    print_step_header(4, "Generating Predictions (TabPFN Client)")
    try:
        _ = load_and_predict_tabpfn_client_for_nba_games(
            df=df_to_predict,
            prediction_date=date_to_predict,
            prediction_datetime=prediction_date,
        )
        print_status("TabPFN predictions generated")
    except Exception as e:
        print_status(f"Failed to generate TabPFN predictions: {e}", ok=False)
        raise


if __name__ == "__main__":
    predict_nba_games_tabpfn()
