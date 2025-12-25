import glob
import os

import pandas as pd


def combine_all_nba_games(data_dir, file_prefix="nba_games"):
    """
    Combine all NBA CSV files from season_games_data directory into one dataframe.

    Args:
        data_dir: Directory containing the CSV files
        file_prefix: Prefix of files to combine (e.g., 'nba_games' or 'nba_players')
    """

    # Find all files matching the pattern
    file_pattern = os.path.join(data_dir, f"{file_prefix}_*.csv")
    csv_files = sorted(glob.glob(file_pattern))

    print(f"Found {len(csv_files)} {file_prefix} files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")

    # Read and combine all CSV files
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, dtype=str)
            print(
                f"Loaded: {len(df)} rows, {len(df.columns)} columns from {os.path.basename(file)}"
            )
            #Change "TO" column to "TOV" to avoid conflict with SQL keyword
            if "TO" in df.columns:
                df = df.rename(columns={"TO": "TOV"})
            if "teamName" in df.columns:
                df = df.rename(columns={"teamName": "TEAM_NAME"})
            if file_prefix == "nba_players":
                #drop columns staring with lower case + TEAM_NAME
                cols_to_drop = [col for col in df.columns if col[0].islower()]
                df = df.drop(columns=cols_to_drop)
                if "TEAM_NAME" in df.columns:
                    df = df.drop(columns=["TEAM_NAME"])
            
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nCombined dataframe shape: {combined_df.shape}")
        print(f"Total rows: {len(combined_df)}")
        print(f"\nColumns: {list(combined_df.columns)}")
        # print(
        #     f"\nDate range: {combined_df['GAME_DATE'].min()} to {combined_df['GAME_DATE'].max()}"
        # )

        return combined_df
    else:
        print("No dataframes to combine!")
        return None


if __name__ == "__main__":
    data_dir = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    )

    # Combine games data
    print("=" * 60)
    print("COMBINING GAMES DATA")
    print("=" * 60)
    df_all_games = combine_all_nba_games(data_dir, file_prefix="nba_games")
    print(f"\nGames columns: {list(df_all_games.columns)}")

    # Combine players data
    print("\n" + "=" * 60)
    print("COMBINING PLAYERS DATA")
    print("=" * 60)
    df_all_players = combine_all_nba_games(data_dir, file_prefix="nba_players")
    print(f"\nPlayers columns: {list(df_all_players.columns)}")
