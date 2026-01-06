from postgre_DB.update_evaluation_predictions import (
    add_ou_betting_metrics,
    compute_daily_accuracy,
    compute_ou_betting_statistics,
    get_games_with_total_scored_points,
    plot_daily_accuracy,
)

if __name__ == "__main__":
    # Example 1: All games with total_scored_points
    df_past = get_games_with_total_scored_points(only_null=False, start_date="2026-01-01")
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
    


    plot_daily_accuracy(daily_accuracy, show_plot=True)

 