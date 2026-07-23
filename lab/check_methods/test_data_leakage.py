"""Manual end-to-end leakage check for generated training data.

This module lives under ``lab`` and requires the project databases, so it is
kept import-safe for pytest collection. Run it directly when the external data
sources are available.
"""

import os

import pandas as pd
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict


def run_data_leakage_check(
    date_to_train: str = "2025-12-19",
    safe_limit: str = "2025-10-11",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify that changing the target does not change generated predictors."""
    previous_test_value = os.environ.get("test")

    try:
        os.environ["test"] = "Yes"
        df_total_zero = create_df_to_predict(
            recent_limit_to_include=date_to_train,
            older_limit_to_include=safe_limit,
        )

        os.environ["test"] = "No"
        df_total_normal = create_df_to_predict(
            recent_limit_to_include=date_to_train,
            older_limit_to_include=safe_limit,
        )
    finally:
        if previous_test_value is None:
            os.environ.pop("test", None)
        else:
            os.environ["test"] = previous_test_value

    pd.testing.assert_frame_equal(
        df_total_zero.drop(columns=["TOTAL_POINTS"]),
        df_total_normal.drop(columns=["TOTAL_POINTS"]),
    )
    return df_total_zero, df_total_normal


if __name__ == "__main__":
    zero_target, normal_target = run_data_leakage_check()
    print(f"Leakage check passed for {len(zero_target)} generated games.")
