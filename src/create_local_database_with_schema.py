from postgre_DB.combine_games_data import combine_all_nba_games
from postgre_DB.create_nba_games_db import (
    create_database,
    create_games_schema,
    load_games_data_to_db,
)
from postgre_DB.create_nba_odds_db import (
    create_odds_table,
    get_recent_odds,
    load_odds_data_to_db,
)
from postgre_DB.create_nba_players_db import (
    create_players_schema_database,
    create_players_table,
    load_players_data_to_db,
)
from postgre_DB.create_nba_predictions_db import create_predictions_table


def create_local_games_schema_database():
    print("Step 1: Creating database...")
    create_database()

    print("\nStep 2: Creating table...")
    create_games_schema()

    print("\nStep 3: Loading combined data...")
    # Load the combined dataframe

    data_dir = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    )
    df_all_games = combine_all_nba_games(data_dir)

    if df_all_games is not None:
        print("\nStep 4: Inserting data into PostgreSQL...")
        load_games_data_to_db(df_all_games)

    print("\n✅ Database setup complete!")


def create_local_players_schema_database():
    print("Step 1: Creating database...")
    create_players_schema_database()

    print("\nStep 2: Creating schema + table...")
    create_players_table()

    print("\nStep 3: Loading combined data...")

    data_dir = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    )
    df_all_players = combine_all_nba_games(data_dir, file_prefix="nba_players")

    if df_all_players is None:
        print("Failed to load combined dataframe!")

    print("\nStep 4: Inserting data into PostgreSQL...")
    load_players_data_to_db(df_all_players)

    print("\nDatabase setup complete.")


def create_local_odds_schema_database():
    create_database()
    print("\nStep 2: Creating schema + nba_odds table...")
    create_odds_table()

    print("\nStep 3: Loading odds data...")
    odds_csv_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/odds_data/odds_data.csv"

    if load_odds_data_to_db(odds_csv_path):
        print("\nOdds setup complete.")

        print("\nSample of recent odds:")
        recent_odds = get_recent_odds(5)
        for odds in recent_odds:
            print(f"\n{odds['game_date']}: {odds['team_home']} vs {odds['team_away']}")
            print(f"  Total Line (avg): {odds['average_total_line']}")
            print(
                f"  Spread (avg): Home {odds['average_spread_home']}, Away {odds['average_spread_away']}"
            )
    else:
        print("Failed to load odds data!")


if __name__ == "__main__":
    # create_local_games_schema_database()
    create_local_players_schema_database()
    # create_local_odds_schema_database()
    # create_predictions_table()
