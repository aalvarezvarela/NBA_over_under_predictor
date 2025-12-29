from postgre_DB.update_evaluation_predictions import (
    add_ou_betting_metrics,
    compute_ou_betting_statistics,
    get_games_with_total_scored_points,
)

if __name__ == "__main__":
    # Example 1: All games with total_scored_points
    df = get_games_with_total_scored_points()
  
    # Example 2: Filter for a specific date
    df_date = get_games_with_total_scored_points(date="2025-12-28")
  

    df_with_metrics = add_ou_betting_metrics(df)
    # Save to Excel
    # df_with_metrics.to_excel("/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/temp.xlsx", index=False)
    stats = compute_ou_betting_statistics(df_with_metrics, print_report=True)
    # print(stats)
    print("\nGames for 2025-12-28 with metrics:")

    df_with_metrics = add_ou_betting_metrics(df_date)
    # Save to Excel
    # df_with_metrics.to_excel("/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/temp.xlsx", index=False)
    stats = compute_ou_betting_statistics(df_with_metrics, print_report=True)