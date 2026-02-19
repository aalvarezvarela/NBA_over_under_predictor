"""
Module for cleaning dataframes before training.

This module provides functions to:
- Perform basic data filtering and validation
- Remove low-quality columns (high NaN %, ID columns, string columns)
- Detect and remove duplicate or highly similar columns
- Remove constant columns
- Apply missing data policy
"""

import numpy as np
import pandas as pd
from nba_ou.data_preparation.missing_data.handle_missing_data import (
    apply_missing_policy,
)


def basic_cleaning(df: pd.DataFrame, verbose: int = 1) -> pd.DataFrame:
    """
    Perform basic cleaning and filtering on the training dataframe.

    This function:
    - Filters out games with unusually low total points (< 130)
    - Removes rows with missing TOTAL_OVER_UNDER_LINE
    - Filters out games with unrealistic betting lines (< 100)

    Args:
        df (pd.DataFrame): Training dataframe to clean
        verbose (int): Verbosity level (0=silent, 1=basic, 2=detailed). Default: 1

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if "IS_US_HOLIDAY" in df.columns:
        df["IS_US_HOLIDAY"] = (
            df["IS_US_HOLIDAY"]
            .astype("Int64")  # ensures proper numeric handling
            .astype("boolean")  # pandas nullable boolean
        )

    initial_rows = len(df)
    if verbose >= 1:
        print(f"Starting basic cleaning with {initial_rows} rows")

    if verbose >= 2:
        print(f"Removed {initial_rows - len(df)} rows with TOTAL_POINTS <= 130")

    # Count and report NaNs in TOTAL_OVER_UNDER_LINE
    nans = df["TOTAL_OVER_UNDER_LINE"].isna().sum()
    if verbose >= 2:
        print(f"Number of NaNs in TOTAL_OVER_UNDER_LINE: {nans}")

    # Drop rows with missing odds data
    df = df.dropna(subset=["TOTAL_OVER_UNDER_LINE"])
    if verbose >= 2:
        print(f"Removed {nans} rows with NaN in TOTAL_OVER_UNDER_LINE")

    # Filter out unrealistic betting lines
    rows_before = len(df)
    df = df[df["TOTAL_OVER_UNDER_LINE"] > 100].copy()
    if verbose >= 2:
        print(f"Removed {rows_before - len(df)} rows with TOTAL_OVER_UNDER_LINE <= 100")

    if verbose >= 1:
        print(f"Basic cleaning complete: {len(df)} rows remaining\n")

    return df


def advanced_column_cleaning(
    df: pd.DataFrame,
    nan_threshold: float = 50.0,
    corr_threshold: float = 0.99,
    keep_columns: list[str] | None = None,
    keep_all_cols: bool = False,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Perform advanced column cleaning on the training dataframe.

    This function:
    - Removes columns containing strings in every value
    - Removes columns with 'ID' in the name
    - Removes columns with high NaN percentage (configurable)
    - Removes duplicate columns
    - Removes columns with 99% similarity to another column (unless keep_all_cols=True)
    - Removes columns that are absolute value matches of another column
    - Removes columns with constant values (unless keep_all_cols=True)

    Args:
        df (pd.DataFrame): Training dataframe to clean
        nan_threshold (float): Percentage threshold for NaN values above which
            a column will be removed (e.g., 40.0 means 40%). Default: 50.0
        corr_threshold (float): Correlation threshold above which columns will be considered highly similar
            and one will be removed. Default: 0.99
        keep_columns (list[str] | None): List of column names to always keep regardless of type or quality.
            Useful for preserving date columns or other important non-numeric columns. Default: None
        keep_all_cols (bool): If True, only drops ID, NAME, and string columns; keeps all others
            (high-NaN, constant, duplicate, correlated, absolute matches). Default: False
        verbose (int): Verbosity level (0=silent, 1=basic, 2=detailed). Default: 1

    Returns:
        pd.DataFrame: Dataframe with cleaned columns
    """
    initial_cols = len(df.columns)
    if verbose >= 1:
        print(f"Starting advanced column cleaning with {initial_cols} columns")
    columns_to_drop = set()

    # Set of columns to always keep
    keep_columns_set = set(keep_columns) if keep_columns else set()
    if keep_columns_set and verbose >= 2:
        print(f"\nProtected columns (will not be removed): {sorted(keep_columns_set)}")

    # 1. Remove columns that are purely string (object/string dtype and all non-null values are str)
    if verbose >= 2:
        print("\n1. Checking for pure string columns...")

    string_cols = []

    for col in df.columns:
        # Skip protected columns
        if col in keep_columns_set:
            continue
        dtype = df[col].dtype

        # Only object / string columns are candidates
        if dtype not in ("object", "string"):
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            # column is all NaN → treat as useless string-like column
            string_cols.append(col)
            continue

        # Drop if ALL non-null values are strings
        if non_null.map(type).eq(str).all():
            string_cols.append(col)

    if string_cols:
        if verbose >= 2:
            print(f"   Removing {len(string_cols)} pure string columns:")
            for c in string_cols:
                print(f"      - {c}")
        columns_to_drop.update(string_cols)
    elif verbose >= 2:
        print("   No pure string columns to remove")

    # 2. Remove columns containing 'ID' in the name
    if verbose >= 2:
        print("\n2. Checking for ID columns...")
    id_cols = [
        col
        for col in df.columns
        if "_ID" in col.upper() and col not in keep_columns_set
    ]
    if id_cols:
        if verbose >= 2:
            print(f"   Removing {len(id_cols)} _ID columns: {id_cols}")
        columns_to_drop.update(id_cols)
    elif verbose >= 2:
        print("   No ID columns to remove")

    # 3. Remove columns containing '_NAME' in the name
    if verbose >= 2:
        print("\n3. Checking for _NAME columns...")
    name_cols = [
        col
        for col in df.columns
        if "_NAME" in col.upper() and col not in keep_columns_set
    ]
    if name_cols:
        if verbose >= 2:
            print(f"   Removing {len(name_cols)} _NAME columns: {name_cols}")
        columns_to_drop.update(name_cols)
    elif verbose >= 2:
        print("   No _NAME columns to remove")

    # 4. Remove columns with high NaN values (configurable)
    if verbose >= 2:
        print(f"\n4. Checking for high-NaN columns (>{nan_threshold}%)...")
    high_nan_cols = []
    for col in df.columns:
        if col in columns_to_drop or col in keep_columns_set:
            continue
        nan_pct = df[col].isna().sum() / len(df) * 100
        if nan_pct > nan_threshold:
            high_nan_cols.append((col, nan_pct))

    if high_nan_cols and not keep_all_cols:
        if verbose >= 2:
            print(
                f"   Removing {len(high_nan_cols)} columns with >{nan_threshold}% NaN:"
            )
            for col, pct in high_nan_cols:
                print(f"      - {col}: {pct:.2f}% NaN")
        for col, pct in high_nan_cols:
            columns_to_drop.add(col)
    elif verbose >= 2:
        if keep_all_cols:
            print("   Skipping high-NaN column removal (keep_all_cols=True)")
        else:
            print("   No high-NaN columns to remove")

    # 5. Remove columns with constant values (same value in every row)
    if verbose >= 2:
        print("\n5. Checking for constant columns...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping constant column removal (keep_all_cols=True)")
    else:
        constant_cols = []
        for col in df.columns:
            if col in columns_to_drop or col in keep_columns_set:
                continue
            if df[col].nunique(dropna=False) == 1:
                constant_cols.append(col)

        if constant_cols:
            if verbose >= 2:
                print(
                    f"   Removing {len(constant_cols)} constant columns: {constant_cols}"
                )
            columns_to_drop.update(constant_cols)
        elif verbose >= 2:
            print("   No constant columns to remove")

    # Drop the columns identified so far before checking for duplicates
    df = df.drop(columns=list(columns_to_drop))

    # 6. Check for duplicate columns (exact matches)
    if verbose >= 2:
        print("\n6. Checking for duplicate columns...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping duplicate column removal (keep_all_cols=True)")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        duplicate_pairs = []

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                if df[col1].equals(df[col2]):
                    duplicate_pairs.append((col1, col2))

        if duplicate_pairs:
            if verbose >= 2:
                print(f"   Found {len(duplicate_pairs)} duplicate column pairs:")
            cols_to_remove = set()
            for col1, col2 in duplicate_pairs:
                if verbose >= 2:
                    print(f"      - {col1} == {col2}")
                # Keep the first one, remove the second
                cols_to_remove.add(col2)
            if verbose >= 2:
                print(f"   Removing {len(cols_to_remove)} duplicate columns")
            df = df.drop(columns=list(cols_to_remove))
        elif verbose >= 2:
            print("   No duplicate columns found")

    # 7. Check for highly similar columns (99% correlation)
    if verbose >= 2:
        print("\n7. Checking for highly similar columns (99.5% correlation)...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping highly correlated column removal (keep_all_cols=True)")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr().abs()

        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns with correlation > 0.99
        similar_pairs = []
        for col in upper.columns:
            high_corr_cols = upper[col][upper[col] > corr_threshold].index.tolist()
            for corr_col in high_corr_cols:
                similar_pairs.append((col, corr_col, upper.loc[corr_col, col]))

        if similar_pairs:
            if verbose >= 2:
                print(f"   Found {len(similar_pairs)} highly similar column pairs:")
            cols_to_remove = set()
            for col1, col2, corr in similar_pairs:
                if verbose >= 2:
                    print(f"      - {col1} ~ {col2} (correlation: {corr:.4f})")
                # Keep the first one, remove the second
                cols_to_remove.add(col2)
            if verbose >= 2:
                print(f"   Removing {len(cols_to_remove)} similar columns")
            df = df.drop(columns=list(cols_to_remove))
        elif verbose >= 2:
            print("   No highly similar columns found")

    # 8. Check for columns that are absolute value matches
    if verbose >= 2:
        print("\n8. Checking for absolute value matches...")

    if keep_all_cols:
        if verbose >= 2:
            print("   Skipping absolute value match removal (keep_all_cols=True)")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        abs_match_pairs = []

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                # Check if one is the absolute value of the other
                if df[col1].abs().equals(df[col2].abs()):
                    # Check if they're not already exact duplicates
                    if not df[col1].equals(df[col2]):
                        abs_match_pairs.append((col1, col2))

        if abs_match_pairs:
            if verbose >= 2:
                print(f"   Found {len(abs_match_pairs)} absolute value match pairs:")
            cols_to_remove = set()
            for col1, col2 in abs_match_pairs:
                if verbose >= 2:
                    print(f"      - abs({col1}) == abs({col2})")
                # Keep the first one, remove the second
                cols_to_remove.add(col2)
            if verbose >= 2:
                print(f"   Removing {len(cols_to_remove)} absolute match columns")
            df = df.drop(columns=list(cols_to_remove))
        elif verbose >= 2:
            print("   No absolute value matches found")

    final_cols = len(df.columns)
    if verbose >= 1:
        print(
            f"\nAdvanced column cleaning complete: {initial_cols} → {final_cols} columns "
            f"({initial_cols - final_cols} removed)\n"
        )

    return df


def clean_dataframe_for_training(
    df: pd.DataFrame,
    nan_threshold: float = 5.0,
    corr_threshold: float = 0.99,
    drop_all_na_rows: bool = False,
    drop_2017_na_rows: bool = True,
    create_missing_flags: bool = False,
    keep_columns: list[str] | None = None,
    keep_all_cols: bool = False,
    verbose: int = 1,
    strict_mode: bool = False,
    strict_mode_exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    """
    Complete cleaning pipeline for training dataframe.

    Applies:
    1. Basic row filtering
    2. Advanced column cleaning
    3. Optional: Drop NA rows for 2017 season year
    4. Missing data policy (drop critical rows, zero-fill, infer, fallback to medians)

    Args:
        df (pd.DataFrame): Raw training dataframe
        nan_threshold (float): Percentage threshold for NaN values above which
            a column will be removed. Default: 50.0
        drop_all_na_rows (bool): If True, drop rows that are all NaN. Default: False
        drop_2017_na_rows (bool): If True, drop rows with any NaN values where
            SEASON_YEAR is 2017. Default: False
        keep_all_cols (bool): If True, only drops ID, NAME, and string columns; keeps all others.
            Default: False
        verbose (int): Verbosity level (0=silent, 1=basic, 2=detailed). Default: 1
        strict_mode (bool): If True, raises an error if any NaN values remain after cleaning. Default: False
        strict_mode_exclude_cols (list[str] | None): Columns to exclude from strict mode check.
            Defaults to ['MATCHUP_TEAM_HOME', 'TOTAL_POINTS'] if None.

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if verbose >= 1:
        print("=" * 80)
        print("STARTING DATAFRAME CLEANING PIPELINE")
        print("=" * 80)

    # Basic cleaning
    df_cleaned = basic_cleaning(df, verbose=verbose)

    # Advanced column cleaning
    df_cleaned = advanced_column_cleaning(
        df_cleaned,
        nan_threshold=nan_threshold,
        corr_threshold=corr_threshold,
        keep_columns=keep_columns,
        keep_all_cols=keep_all_cols,
        verbose=verbose,
    )

    # Drop NA rows for 2017 season year if requested
    if drop_2017_na_rows and "SEASON_YEAR" in df_cleaned.columns:
        if verbose >= 1:
            print("\nDropping NA rows for SEASON_YEAR 2017...")

        initial_rows = len(df_cleaned)
        mask_2017 = df_cleaned["SEASON_YEAR"] == 2017
        rows_2017_before = mask_2017.sum()

        # Drop rows where SEASON_YEAR is 2017 and there are any NaN values
        df_cleaned = df_cleaned[~(mask_2017 & df_cleaned.isna().any(axis=1))]

        rows_2017_after = (df_cleaned["SEASON_YEAR"] == 2017).sum()
        rows_removed = initial_rows - len(df_cleaned)

        if verbose >= 2:
            print(f"   2017 rows before: {rows_2017_before}")
            print(f"   2017 rows after: {rows_2017_after}")
            print(f"   Total rows removed: {rows_removed}")
        elif verbose >= 1:
            print(f"   Removed {rows_removed} rows with NaN values from 2017 season")

    # Apply missing data policy
    if verbose >= 1:
        print("\nApplying missing data policy...")
    df_cleaned = apply_missing_policy(
        df_cleaned,
        current_total_line_col="TOTAL_OVER_UNDER_LINE",
        create_missing_flags=create_missing_flags,
        keep_all_cols=keep_all_cols,
    )
    if drop_all_na_rows:
        if verbose >= 1:
            print("\nDropping rows that contain NaN...")

        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()

        if verbose >= 2:
            print(f"Removed {initial_rows - len(df_cleaned)} all-NaN rows")

    # Check for remaining NaN values in strict mode
    if strict_mode:
        # Default exclusions: columns kept for info but not used in model
        if strict_mode_exclude_cols is None:
            strict_mode_exclude_cols = ["MATCHUP_TEAM_HOME", "TOTAL_POINTS"]

        nan_counts = df_cleaned.isna().sum()
        columns_with_nan = nan_counts[nan_counts > 0]

        # Filter out excluded columns
        columns_with_nan = columns_with_nan[
            ~columns_with_nan.index.isin(strict_mode_exclude_cols)
        ]

        if not columns_with_nan.empty:
            error_msg = "Strict mode: NaN values found after cleaning pipeline:\n"
            for col, count in columns_with_nan.items():
                pct = (count / len(df_cleaned)) * 100
                error_msg += f"  - {col}: {count} NaN values ({pct:.2f}%)\n"
            raise ValueError(error_msg)

        if verbose >= 1:
            excluded_info = (
                f" (excluding {strict_mode_exclude_cols})"
                if strict_mode_exclude_cols
                else ""
            )
            print(f"\nStrict mode check passed: No NaN values remaining{excluded_info}")

    if verbose >= 1:
        print("=" * 80)
        print("CLEANING COMPLETE")
        print(f"Final shape: {df_cleaned.shape}")
        print("=" * 80)

    return df_cleaned
