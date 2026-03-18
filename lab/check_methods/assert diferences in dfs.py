import pandas as pd

# Paths
predictions_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/prediction_data_17_march.csv"
# predictions_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data/all_odds_training_data_until_20260311_2.csv"
other_path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data/all_odds_training_data_until_20260317.csv"

# Load data
df_pred = pd.read_csv(predictions_path)
df_pred = df_pred[df_pred["GAME_DATE"] == "2026-03-17"]
df_other = pd.read_csv(other_path)

# Ensure consistent dtype
df_pred["GAME_ID"] = df_pred["GAME_ID"].astype(str)
df_other["GAME_ID"] = df_other["GAME_ID"].astype(str)


# Filter other dataframe using GAME_ID from predictions
df_other_filtered = df_other[df_other["GAME_ID"].isin(df_pred["GAME_ID"])].copy()

# Align both by GAME_ID
df_pred = df_pred.set_index("GAME_ID").sort_index()
df_other_filtered = df_other_filtered.set_index("GAME_ID").sort_index()

# # Filter to just the first GAME_ID
# first_game_id = df_pred.index[0]
# df_pred = df_pred.loc[[first_game_id]]
# df_other_filtered = df_other_filtered.loc[[first_game_id]]

# Keep only common columns
common_cols = df_pred.columns.intersection(df_other_filtered.columns)
df_pred = df_pred[common_cols]
df_other_filtered = df_other_filtered[common_cols]

# -----------------------------------------
# Round numeric columns to avoid tiny float differences
# -----------------------------------------
numeric_cols = df_pred.select_dtypes(include=["number"]).columns

df_pred[numeric_cols] = df_pred[numeric_cols].round(4)
df_other_filtered[numeric_cols] = df_other_filtered[numeric_cols].round(4)

# -----------------------------------------
# Detect differences
# -----------------------------------------
diff_mask = df_pred.ne(df_other_filtered)

cols_with_diff = diff_mask.any()
diff_columns = cols_with_diff[cols_with_diff].index.tolist()

# -----------------------------------------
# Build dataframe with interleaved columns
# -----------------------------------------
dfs = []

for col in diff_columns:
    dfs.append(df_pred[[col]].rename(columns={col: f"{col}_pred"}))
    dfs.append(df_other_filtered[[col]].rename(columns={col: f"{col}_other"}))

df_differences = pd.concat(dfs, axis=1).reset_index()

print("Columns with differences:")
print(diff_columns)

# -----------------------------------------
# Keep only rows with actual differences
# -----------------------------------------
rows_with_diff = diff_mask[diff_columns].any(axis=1)

df_differences = df_differences[
    df_differences["GAME_ID"].isin(df_pred.index[rows_with_diff])
]

# Save results
df_differences.to_csv(
    "predictions_vs_other_differences_10_only3seasons_what.csv", index=False
)
