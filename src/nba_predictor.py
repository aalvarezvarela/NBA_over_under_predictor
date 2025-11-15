"""
NBA Over/Under Predictor - Main Execution Script

This is the main entry point for the NBA Over/Under prediction system.
It orchestrates the entire prediction pipeline:
1. Updates the game database with latest data
2. Fetches and processes injury reports
3. Prepares prediction features
4. Runs ML models for predictions
5. Exports results to Excel

Usage:
    python nba_predictor.py                    # Predict for today
    python nba_predictor.py -d 2025-01-15     # Predict for specific date
"""

import argparse
from datetime import datetime

from config import LEGEND, settings
from data_processing import create_df_to_predict
from fetch_data.manage_games_database import update_database
from models import predict_nba_games, save_predictions_to_excel


def print_step_header(step_number: int, title: str) -> None:
    """
    Prints a formatted header for each pipeline step.

    Args:
        step_number: The step number in the pipeline
        title: Description of the step
    """
    print("\n" + "=" * 60)
    print(f"STEP {step_number}: {title}")
    print("=" * 60 + "\n")


def main():
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="NBA Over/Under Prediction System - Generate predictions for NBA games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Generate predictions for today's games
  %(prog)s -d 2025-01-15     Generate predictions for January 15, 2025
        """,
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        help="Prediction date in YYYY-MM-DD format (default: today)",
    )

    args = parser.parse_args()

    # Determine the date to predict
    if args.date:
        try:
            date_to_predict = datetime.strptime(args.date, "%Y-%m-%d").strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            print("❌ Error: Invalid date format. Please use YYYY-MM-DD.")
            return 1
    else:
        date_to_predict = datetime.now().strftime("%Y-%m-%d")

    # Print welcome message
    print("\n" + "=" * 60)
    print("  NBA OVER/UNDER PREDICTION SYSTEM")
    print("=" * 60)
    print(f"📅 Prediction Date: {date_to_predict}")
    print(f"📁 Data Folder: {settings.data_folder}")
    print(f"🤖 Regressor Model: {settings.regressor_model_path}")
    print(f"🤖 Classifier Model: {settings.classifier_model_path}")
    print(f"📊 Output Folder: {settings.output_path}")

    # Ensure output directory exists
    output_dir = settings.get_absolute_path(settings.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_folder = settings.get_absolute_path(settings.data_folder)

    try:
        # Step 1: Update the database
        print_step_header(1, "Updating Game Database")
        update_database(str(data_folder))

        # Step 2: Prepare data for prediction
        print_step_header(2, "Processing Data and Injury Reports")
        df_to_predict = create_df_to_predict(
            data_path=str(data_folder),
            date_to_predict=date_to_predict,
            nba_injury_reports_url=settings.nba_injury_reports_url,
            reports_path=str(settings.get_absolute_path(settings.report_path)),
            filter_for_date_to_predict=True,
        )

        if df_to_predict.empty:
            print("⚠️  Warning: No games found for the specified date.")
            print("    Please check if there are scheduled games on this date.")
            return 0

        print(f"✓ Found {len(df_to_predict)} games to predict")

        # Step 3: Generate predictions
        print_step_header(3, "Generating Predictions")
        predictions_dfs = predict_nba_games(df_to_predict)

        # Step 4: Save predictions
        print_step_header(4, "Saving Results")
        output_file = output_dir / f"NBA_predictions_{date_to_predict}.xlsx"
        save_predictions_to_excel(predictions_dfs, str(output_file), LEGEND)

        # Print success summary
        print("\n" + "=" * 60)
        print("  ✓ PREDICTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📄 Results saved to: {output_file}")
        print(f"📊 Total games analyzed: {len(df_to_predict)}")
        print("\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found - {e}")
        print("   Please check your configuration and data files.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during prediction pipeline: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
