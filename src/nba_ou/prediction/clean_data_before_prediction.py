# from __future__ import annotations

# from typing import Iterable, Optional

# import pandas as pd
# from nba_ou.data_preparation.missing_data.handle_missing_data import (
#     apply_missing_policy,
#     compute_and_save_train_medians,
# )


# def prepare_training_dataframe(
#     df_stats: pd.DataFrame,
#     *,
#     total_line_col: str = "TOTAL_OVER_UNDER_LINE",
#     game_date_col: str = "GAME_DATE",
#     min_total_line: float = 100.0,
#     drop_mode: str = "strict",
#     base_cols_to_drop: Optional[Iterable[str]] = None,
#     extra_text_cols_to_drop: Optional[Iterable[str]] = None,
# ) -> tuple[pd.DataFrame, dict, dict]:
#     """
#     Replicates the preprocessing steps in your snippet.

#     Steps:
#       1) Filter df_stats in place (dropna on total_line_col, then keep total_line_col > min_total_line)
#       2) Build cols_to_drop:
#          - base identifiers (SEASON_ID, GAME_ID, etc.)
#          - team/matchup string columns
#          - all object/string columns found in df_to_train
#          - ensure GAME_DATE is NOT dropped until the end
#       3) Drop those columns from df_to_train
#       4) Compute train medians
#       5) Apply missing policy with the provided total line column and drop_mode
#       6) Drop GAME_DATE at the end
#     Returns:
#       - cleaned df_to_train
#       - report from apply_missing_policy
#       - train_medians used
#     """

#     # 1) Filter df_stats (matches your in-place behavior)
#     if total_line_col not in df_stats.columns:
#         raise KeyError(f"'{total_line_col}' not found in df_stats columns.")
#     df_stats.dropna(subset=[total_line_col], inplace=True)
#     df_stats = df_stats[df_stats[total_line_col] > min_total_line]

#     # 2) Identify text columns in df_stats
#     text_columns = df_stats.select_dtypes(
#         include=["object", "string"]
#     ).columns.tolist()

#     default_base_cols_to_drop = [
#         "SEASON_ID",
#         "GAME_ID",
#         "SEASON_TYPE",
#         "TEAM_ID_TEAM_HOME",
#         "TEAM_ID_TEAM_AWAY",
#         "IS_OVERTIME",
#     ]
#     if base_cols_to_drop is None:
#         base_cols_to_drop = default_base_cols_to_drop
#     else:
#         base_cols_to_drop = list(base_cols_to_drop)

#     matchup_team_cols = [
#         "MATCHUP_TEAM_HOME",
#         "TEAM_CITY_TEAM_AWAY",
#         "TEAM_ABBREVIATION_TEAM_AWAY",
#         "TEAM_ABBREVIATION_TEAM_HOME",
#         "TEAM_CITY_TEAM_HOME",
#         "TEAM_NAME_TEAM_AWAY",
#         "MATCHUP_TEAM_AWAY",
#         "TEAM_NAME_TEAM_HOME",
#     ]

#     cols_to_drop = list(base_cols_to_drop) + matchup_team_cols

#     # Add all text columns detected automatically
#     cols_to_drop += text_columns

#     # Allow caller to pass additional text-like columns to drop (optional)
#     if extra_text_cols_to_drop:
#         cols_to_drop += list(extra_text_cols_to_drop)

#     # Your snippet explicitly removes GAME_DATE from cols_to_drop so it survives until the end
#     if game_date_col in cols_to_drop:
#         cols_to_drop = [c for c in cols_to_drop if c != game_date_col]

#     # 3) Drop columns
#     df_clean = df_stats.drop(columns=cols_to_drop, errors="ignore").copy()

#     # 4) Training medians
#     train_medians = compute_and_save_train_medians(df_clean)

#     # 5) Missing policy
#     df_clean, report = apply_missing_policy(
#         df_clean,
#         train_medians=train_medians,
#         current_total_line_col=total_line_col,
#         drop_mode=drop_mode,
#     )

#     # 6) Drop GAME_DATE last (matches your snippet)
#     df_clean = df_clean.drop(columns=[game_date_col], errors="ignore")

#     return df_clean
