import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import (
    create_df_to_predict,
)
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)

# from models import predict_nba_games, save_predictions_to_excel
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.prediction import (
    PREDICTION_TARGET_LINE_ERROR,
    PREDICTION_TARGET_TOTAL_POINTS,
    load_s3_model_and_predict,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    load_and_predict_tabpfn_client_for_nba_games,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import (
    make_s3_client,
)


def print_banner(date_to_predict: str) -> None:
    line = "=" * 70
    title = "NBA OVER/UNDER PREDICTION SYSTEM"
    print(f"\n{line}")
    print(title.center(len(line)))
    print(line)
    print(f"  Prediction Date: {date_to_predict}")
    print(line + "\n")


def print_step_header(step_number: int, title: str) -> None:
    """Print a compact, eye-catching step header."""
    sep = "-" * 70
    header = f" STEP {step_number} — {title} "
    print(sep)
    print(header.center(len(sep)))
    print(sep)


def print_status(message: str, ok: bool = True) -> None:
    """Print a single-line status message with a check or cross."""
    symbol = "✓" if ok else "✖"
    print(f"  {symbol} {message}")


def predict_nba_games(run_tabpfn_client: bool = False) -> None:
    """
    Main execution function for the NBA prediction pipeline.

    This function:
    1. Parses command-line arguments for the prediction date
    2. Loads configuration settings
    3. Updates the game database
    4. Processes data with injuries
    5. Generates predictions
    6. Saves results to Excel
    """

    date_to_predict = (datetime.now(ZoneInfo("US/Pacific"))).strftime("%Y-%m-%d")

    # Print welcome banner
    print_banner(date_to_predict)
    print(
        "  S3 Models:\n"
        f"    - Full Dataset (Line Error): {SETTINGS.s3_regressor_full_dataset_prefix}\n"
        f"    - Full Dataset (Total Points): {SETTINGS.s3_regressor_full_dataset_total_points_prefix}\n"
        f"    - Recent Games (Line Error): {SETTINGS.s3_regressor_recent_games_prefix}\n"
        f"    - Recent Games (Total Points): {SETTINGS.s3_regressor_recent_games_total_points_prefix}\n"
    )

    # Step 1: Update the database
    print_step_header(1, "Updating All Databases")
    season_to_update = get_season_year_from_date(date_to_predict)
    try:
        update_all_databases(
            start_season_year=int(season_to_update),
            end_season_year=int(season_to_update),
            only_new_games=True,
            headless=SETTINGS.headless,
        )
        print_status("Databases updated")

    except Exception as e:
        print_status(f"Failed to update databases: {e}", ok=False)
        raise

    # Step 2: Fetch scheduled games, referees, injuries and odds
    print_step_header(2, "Fetching Scheduled Games & Reports")
    try:
        scheduled_data = get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            headless=SETTINGS.headless,
        )
        print_status("Fetched scheduled games, refs, injuries and odds")
        if scheduled_data["scheduled_games"].empty:
            print_status(
                "⚠️ WARNING: No scheduled games found for the specified date.", ok=False
            )
            return 0

    except Exception as e:
        print_status(f"Failed to fetch scheduled data: {e}", ok=False)
        raise

    # Step 3: Build feature DataFrame for prediction
    print_step_header(3, "Preparing Feature DataFrame")
    try:
        df_to_predict_total = create_df_to_predict(
            todays_prediction=True,
            scheduled_data=scheduled_data,
            recent_limit_to_include=date_to_predict,
            strict_mode=2,
        )
        df_to_predict = df_to_predict_total[
            df_to_predict_total["GAME_DATE"] == date_to_predict
        ].copy()
        print_status("Feature DataFrame prepared")
    
    except Exception as e:
        print_status(f"Failed to prepare features: {e}", ok=False)
        raise

    if df_to_predict.empty:
        print("⚠️  Warning: No games found for the specified date.")
        raise ValueError("df to predict is empty")

    print_status(f"Found {len(df_to_predict)} game(s) to predict")

    # Initialize S3 client once for all model operations
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    prediction_time = datetime.now(ZoneInfo("Europe/Madrid"))

    model_runs = [
       
        (
            "Full Dataset Model (Total Points)",
            SETTINGS.s3_regressor_full_dataset_total_points_prefix,
            "full_dataset_total_points",
            PREDICTION_TARGET_TOTAL_POINTS,
        ),
        
        (
            "Recent Games Model (Total Points)",
            SETTINGS.s3_regressor_recent_games_total_points_prefix,
            "recent_games_total_points",
            PREDICTION_TARGET_TOTAL_POINTS,
        ),
    #     (
    #         "Full Dataset Model (Line Error)",
    #         SETTINGS.s3_regressor_full_dataset_prefix,
    #         "full_dataset",
    #         PREDICTION_TARGET_LINE_ERROR,
    #     ),
    #     (
    #         "Recent Games Model (Line Error)",
    #         SETTINGS.s3_regressor_recent_games_prefix,
    #         "recent_games",
    #         PREDICTION_TARGET_LINE_ERROR,
    #     ),
    ]

    step_number = 4
    for step_title, prefix, model_id, prediction_target in model_runs:
        print_step_header(step_number, f"Generating Predictions ({step_title})")
        step_number += 1
        try:
            if not prefix:
                raise ValueError(
                    f"S3 prefix missing in config for model run '{step_title}'"
                )

            _ = load_s3_model_and_predict(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                prefix=prefix,
                df=df_to_predict,
                model_id=model_id,
                prediction_datetime=prediction_time,
                prediction_target=prediction_target,
            )
            print_status(f"{step_title} predictions generated")
        except Exception as e:
            print_status(f"Failed to generate {step_title} predictions: {e}", ok=False)
            raise

    if not run_tabpfn_client:
        print("\nPrediction pipeline completed successfully.")
        return

    # Step 6: Generate predictions with TabPFN client
    print_step_header(step_number, "Generating Predictions (TabPFN Client)")
    try:
        _ = load_and_predict_tabpfn_client_for_nba_games(
            df=df_to_predict_total,
            prediction_date=date_to_predict,
            prediction_datetime=prediction_time,
        )
        print_status("TabPFN predictions generated")

    except Exception as e:
        print_status(f"Failed to generate TabPFN predictions: {e}", ok=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NBA over/under predictions")
    parser.add_argument(
        "--no-tabpfn",
        action="store_true",
        help="Disable TabPFN predictions",
    )
    args = parser.parse_args()

    predict_nba_games(run_tabpfn_client=False)
