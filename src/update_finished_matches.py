from postgre_DB.update_total_points_predictions import (
    get_game_ids_with_null_total_scored_points,
    update_total_scored_points,
)

if __name__ == "__main__":
    updates = get_game_ids_with_null_total_scored_points()
    print(f"Found {len(updates)} games to update with total scored points.")
    if not updates.empty:
        update_total_scored_points(updates)
        print("Total scored points updated successfully.")
    else:
        print("No updates needed.")
