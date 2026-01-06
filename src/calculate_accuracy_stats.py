from postgre_DB.update_evaluation_predictions import (
    add_ou_betting_metrics,
    compute_daily_accuracy,
    compute_ou_betting_statistics,
    get_games_with_total_scored_points,
    plot_daily_accuracy,
)

if __name__ == "__main__":
    # Example 1: All games with total_scored_points
    df_past = get_games_with_total_scored_points(only_null=False)
    df_predicted = get_games_with_total_scored_points(only_null=True)

    # Keep only the most recent prediction for each game_id
    df_past =  df_past.sort_values("prediction_date").groupby("game_id", as_index=False).tail(1)

    df_with_metrics = add_ou_betting_metrics(df_past)

    # Overall statistics
    stats = compute_ou_betting_statistics(df_with_metrics, print_report=True)

    # Daily accuracy analysis
    print("\n" + "=" * 110)
    print("DAILY ACCURACY BREAKDOWN")
    print("=" * 110)
    daily_accuracy = compute_daily_accuracy(df_with_metrics)
    print(daily_accuracy.to_string(index=False))
    
    # Filter for specific date
    df_2025_12_30 = df_with_metrics[df_with_metrics['game_date'] == '2025-12-30']
    print(f"\n\nPredictions for 2025-12-30: {len(df_2025_12_30)} games")
    print(df_2025_12_30[['game_id', 'game_date', 'home_team', 'away_team', 
                         'predicted_total_score', 'total_over_under_line', 
                         'total_scored_points', 'regressor_side', 'actual_side', 
                         'regressor_correct']].to_string(index=False))


    # Plot daily accuracy and save to file
    # plot_save_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/daily_accuracy_plot.png"
    plot_daily_accuracy(daily_accuracy, show_plot=True)

    # Optionally save to Excel
    # df_with_metrics.to_excel("/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/temp.xlsx", index=False)
    # daily_accuracy.to_excel("/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/daily_accuracy.xlsx", index=False)
