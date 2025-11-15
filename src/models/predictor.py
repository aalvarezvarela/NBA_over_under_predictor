"""
NBA Over/Under Predictor - Prediction Module

This module handles NBA game outcome predictions using trained XGBoost models.
It supports both regression (total score prediction) and classification (over/under)
approaches and generates comprehensive prediction reports.
"""

import pickle

import pandas as pd

try:
    from ..config.settings import settings
except ImportError:
    from config.settings import settings


def predict_nba_games(
    df: pd.DataFrame,
    regressor_path: str | None = None,
    classifier_path: str | None = None,
) -> list[pd.DataFrame]:
    """
    Predicts the outcome of NBA games using trained regression and classification models.

    This function processes NBA game data and makes predictions using two complementary
    approaches:
    1. Regression model: Predicts the total score and compares to over/under line
    2. Classification model: Directly predicts over/under outcome

    Args:
        df: Input DataFrame containing game data with features
        regressor_path: Path to trained regressor model. If None, uses config setting.
        classifier_path: Path to trained classifier model. If None, uses config setting.

    Returns:
        List of DataFrames containing:
            1. Summary DataFrame with key predictions
            2. Summary with top 20 classifier features
            3. Summary with top 20 regressor features
            4. Full feature DataFrame with all data
    """
    # Use config paths if not provided
    if regressor_path is None:
        regressor_path = settings.get_absolute_path(settings.regressor_model_path)
    if classifier_path is None:
        classifier_path = settings.get_absolute_path(settings.classifier_model_path)

    # Drop games without over/under lines
    df = df[df["TOTAL_OVER_UNDER_LINE"] != 0].copy()

    # Load the models
    with open(regressor_path, "rb") as f:
        regressor = pickle.load(f)
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)

    # Extract required features from model
    required_features_regressor = regressor.get_booster().feature_names
    required_features_classifier = classifier.get_booster().feature_names

    df_predictable = df[
        pd.to_numeric(df["TOTAL_OVER_UNDER_LINE"], errors="coerce").notnull()
    ].copy()
    df_non_predictable = df[
        pd.to_numeric(df["TOTAL_OVER_UNDER_LINE"], errors="coerce").isnull()
    ].copy()

    # Convert column to float for prediction
    df_predictable["TOTAL_OVER_UNDER_LINE"] = df_predictable[
        "TOTAL_OVER_UNDER_LINE"
    ].astype(float)

    # Select feature data
    X_reg = df_predictable[required_features_regressor]
    X_clf = df_predictable[required_features_classifier]

    # Make predictions
    predictions = regressor.predict(X_reg)

    df_predictable["PREDICTED_TOTAL_SCORE"] = predictions
    df_predictable["Margin Difference Prediction vs Over/Under"] = (
        df_predictable["PREDICTED_TOTAL_SCORE"]
        - df_predictable["TOTAL_OVER_UNDER_LINE"]
    )
    df_predictable["Regressor Prediction"] = df_predictable[
        "Margin Difference Prediction vs Over/Under"
    ].apply(lambda x: "Over" if x > 0 else ("Under" if x < 0 else "Push"))

    # Classifier predictions
    predictions_classifier = classifier.predict(X_clf)
    df_predictable["Classifier_Prediction_model2"] = predictions_classifier
    df_predictable["Classifier_Prediction_model2"] = df_predictable[
        "Classifier_Prediction_model2"
    ].apply(lambda x: "Over" if x == 1 else "Under")

    df = pd.concat([df_predictable, df_non_predictable]).sort_index()

    # Compute feature importance
    feature_importances_classifier = pd.DataFrame(
        {"Feature": X_clf.columns, "Importance": classifier.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    feature_importances_regressor = pd.DataFrame(
        {"Feature": X_reg.columns, "Importance": regressor.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # Create a list to store the DataFrames
    list_of_dfs = []

    df.rename(columns={"MATCHUP_TEAM_HOME": "MATCHUP"}, inplace=True)
    df["GAME_DATE"] = df["GAME_DATE"].astype(str).str.split("T").str[0]
    # Sheet 1: Summary DataFrame
    summary_columns = [
        "GAME_ID",
        "SEASON_TYPE",
        "GAME_DATE",
        "TEAM_NAME_TEAM_HOME",
        "GAME_NUMBER_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "GAME_NUMBER_TEAM_AWAY",
        "MATCHUP",
        "TOTAL_OVER_UNDER_LINE",
        "PREDICTED_TOTAL_SCORE",
        "Margin Difference Prediction vs Over/Under",
        "Regressor Prediction",
        "Classifier_Prediction_model2",
    ]
    # Rename Matchup_team_home to matchup
    # change complate date to just year month day

    df_summary = df[summary_columns].copy()
    list_of_dfs.append(df_summary)

    # Sheet 2: Summary with Top 10 Most Important Features
    top_10_features_classifier = (
        feature_importances_classifier["Feature"].head(20).tolist()
    )
    summary_with_top_features_columns_classifier = (
        summary_columns + top_10_features_classifier
    )
    top_10_features_regressor = (
        feature_importances_regressor["Feature"].head(20).tolist()
    )
    summary_with_top_features_columns_regressor = (
        summary_columns + top_10_features_regressor
    )

    # Add the top 10 features to the summ

    df_summary_top_features_classifier = df[
        summary_with_top_features_columns_classifier
    ].copy()
    list_of_dfs.append(df_summary_top_features_classifier)

    df_summary_top_features_regressor = df[
        summary_with_top_features_columns_regressor
    ].copy()
    list_of_dfs.append(df_summary_top_features_regressor)

    # Sheet 3: Full Feature DataFrame
    # Order: First same columns as summary, then the rest
    remaining_columns = [col for col in df.columns if col not in summary_columns]
    full_feature_columns = summary_columns + remaining_columns

    df_full_features = df[full_feature_columns].copy()
    list_of_dfs.append(df_full_features)

    return list_of_dfs


def create_column_descriptions_df(catalogue: dict) -> pd.DataFrame:
    """
    Creates a DataFrame containing column descriptions for the prediction report.

    This DataFrame serves as a legend/documentation sheet in the Excel output,
    helping users understand what each column represents.

    Args:
        catalogue: Dictionary mapping column names to their descriptions

    Returns:
        DataFrame with 'ColumnName' and 'Description' columns
    """
    # Create a list of (column_name, description) tuples from the dict
    columns_list = [(col_name, desc) for col_name, desc in catalogue.items()]

    # Build the DataFrame
    df = pd.DataFrame(columns_list, columns=["ColumnName", "Description"])
    return df


def save_predictions_to_excel(
    list_of_dfs: list, file_path: str, catalogue: dict
) -> None:
    """
    Saves prediction results to an Excel file with multiple sheets.

    Creates a comprehensive Excel workbook with separate sheets for:
    - Summary predictions
    - Top features for classifier
    - Top features for regressor
    - Full feature set
    - Legend/column descriptions

    Args:
        list_of_dfs: List of DataFrames to save (summary, features, full data)
        file_path: Path where the Excel file will be saved
        catalogue: Dictionary of column descriptions for the legend sheet
    """
    sheet_names = [
        "Summary",
        "Top_20_classifier_model2",
        "Top_20_regressor_model1",
        "Full_Features",
        "Legend",
    ]

    # Add the legend/descriptions as the last sheet
    list_of_dfs.append(create_column_descriptions_df(catalogue))

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        for df, sheet in zip(list_of_dfs, sheet_names, strict=True):
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Predictions saved successfully at {file_path}")


def main():
    """
    Main execution function for generating NBA game predictions.

    This function:
    1. Loads configuration settings
    2. Fetches and processes game data for today's date
    3. Runs predictions using both models
    4. Saves results to Excel file
    """
    from datetime import datetime

    from ..data_processing import create_df_to_predict

    # Get today's date
    date_to_predict = datetime.now().strftime("%Y-%m-%d")

    print(f"Generating predictions for {date_to_predict}...")
    print(f"Data folder: {settings.data_folder}")
    print(f"Using regressor: {settings.regressor_model_path}")
    print(f"Using classifier: {settings.classifier_model_path}")

    # Create the prediction dataframe
    df_to_predict = create_df_to_predict(
        data_path=settings.get_absolute_path(settings.data_folder),
        date_to_predict=date_to_predict,
        nba_injury_reports_url=settings.nba_injury_reports_url,
        reports_path=settings.get_absolute_path(settings.report_path),
    )

    # Generate predictions
    predictions_dfs = predict_nba_games(df_to_predict)

    # Prepare output path
    output_path = settings.get_absolute_path(settings.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"predictions_{date_to_predict}.xlsx"

    # Create a basic legend/catalogue (you may want to import this from elsewhere)
    # For now, using an empty dict - replace with actual column descriptions
    catalogue = {}

    # Save predictions
    save_predictions_to_excel(predictions_dfs, str(output_file), catalogue)
    print(f"\n✓ Predictions completed successfully!")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    main()
    main()
