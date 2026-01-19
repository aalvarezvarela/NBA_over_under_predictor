"""
Handle missing data in NBA prediction DataFrames using a deterministic, quality-first policy.

Core idea: make the NA policy explicit and auditable by categorizing columns upfront into:

1) DROP ROWS if any of these are missing (high-signal, non-imputable, or indicates pipeline failure)
2) ZERO-FILL these when missing (neutral constructs where 0 means "no effect" or "flat/undefined")
3) INFER these when missing (rolling-window features inferred from season-to-date counterparts)
4) FINAL FALLBACK to training medians (optional, recommended)

Important:
- This policy must be applied consistently at training and prediction time.
- For backtesting / CV, training medians must be computed within each fold only.

Usage:
    # Training time
    train_medians = compute_and_save_train_medians(df_train, "train_medians.csv")

    # Prediction time
    df_clean, report = apply_missing_policy(
        df_predict,
        train_medians=train_medians,
        current_total_line_col="TOTAL_OVER_UNDER_LINE",
        drop_mode="strict",
    )
"""

import re
from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype

# ------------------------------------------------------------------------------
# 1) EXPLICIT COLUMN POLICY (DEFINE WHAT HAPPENS WHERE)
# ------------------------------------------------------------------------------

# A) Columns for which we drop the entire row if missing (explicit lists + regex groups)
DROP_IF_MISSING_EXACT = [
    # Market odds (high signal, do not impute for quality-first)
    "SPREAD",
    "MONEYLINE_TEAM_HOME",
    "MONEYLINE_TEAM_AWAY",
    # Note: current_total_line_col is appended dynamically in apply_missing_policy
]

DROP_IF_MISSING_CORE_TEAM_STATS_EXACT = [
    # Core season-to-date team strength and pace metrics
    "OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME",
    "OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
    "DEF_RATING_SEASON_BEFORE_AVG_TEAM_HOME",
    "DEF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
    "PACE_PER40_SEASON_BEFORE_AVG_TEAM_HOME",
    "PACE_PER40_SEASON_BEFORE_AVG_TEAM_AWAY",
    "PTS_SEASON_BEFORE_AVG_TEAM_HOME",
    "PTS_SEASON_BEFORE_AVG_TEAM_AWAY",
]

# TOP player columns are determined via regex and a strictness mode.
TOP_PLAYER_CRITICAL_REGEX_TEMPLATE = (
    r"^TOP{top_range}_PLAYER_"
    r"(MIN|PTS|OFF_RATING|DEF_RATING|TS_PCT|PACE_PER40)_"
    r"BEFORE_TEAM_(HOME|AWAY)$"
)

# B) Columns we set to 0 when missing (explicit by pattern)
ZERO_FILL_SUBSTRINGS = [
    "TREND_SLOPE",
    "FORM_Z",
    "DIFF_HOME_MINUS_AWAY",
    "DIFERENCE_",  # misspelling present in your features
    "OFFDEF_MISMATCH",
    "REF_TRIO_DIFFERENCE",
]
ZERO_FILL_EXACT = [
    "STAR_PTS_PCT_DIFF_HOME_MINUS_AWAY",
]
ZERO_FILL_PREFIXES = [
    "TRAVEL_RECENCY_RATIO_",
]
ZERO_FILL_SUBSTRINGS_2 = [
    "_SEASON_BEFORE_STD_",  # STD often NaN with too-few samples; treat as 0 dispersion info
]

# C) Columns we infer from season-to-date averages when missing (by naming convention)
# We attempt to map rolling-window feature -> SEASON_BEFORE_AVG_TEAM_(HOME|AWAY)
INFER_FROM_SEASON_AVG_SUFFIX = "_BEFORE_TEAM_"
INFER_ROLLING_PATTERNS_TO_REMOVE = (
    r"_LAST_(HOME_AWAY|ALL)_5_MATCHES",
    r"_LAST_HOME_AWAY_10_WMA",
    r"_LAST_HOME_AWAY_5_WMA",
    r"_LAST_10_WMA",
    r"_LAST_5_WMA",
)

# D) Optional: compute implied points deterministically if total line + spread available
IMPLIED_PTS_COLS = ("IMPLIED_PTS_HOME", "IMPLIED_PTS_AWAY")


# ------------------------------------------------------------------------------
# 2) POLICY OBJECT (RESOLVED AGAINST A GIVEN DF)
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MissingPolicy:
    drop_cols: list[str]
    zero_cols: list[str]
    infer_pairs: list[tuple[str, str]]  # (col_to_fill, season_avg_fallback)


def _existing(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _resolve_top_player_cols(
    df: pd.DataFrame, drop_mode: Literal["strict", "moderate", "lenient"]
) -> list[str]:
    if drop_mode == "strict":
        top_range = "[1-4]"
    elif drop_mode == "moderate":
        top_range = "[1-2]"
    else:
        top_range = "1"

    regex = TOP_PLAYER_CRITICAL_REGEX_TEMPLATE.format(top_range=top_range)
    return df.filter(regex=regex).columns.tolist()


def _is_zero_fill_col(col: str) -> bool:
    if col in ZERO_FILL_EXACT:
        return True
    if any(col.startswith(p) for p in ZERO_FILL_PREFIXES):
        return True
    if any(s in col for s in ZERO_FILL_SUBSTRINGS):
        return True
    if any(s in col for s in ZERO_FILL_SUBSTRINGS_2):
        return True
    return False


def _build_season_avg_fallback(col: str) -> str | None:
    """
    Map a rolling window feature to SEASON_BEFORE_AVG_TEAM_{HOME|AWAY}.

    Example:
      FG_PCT_LAST_HOME_AWAY_5_MATCHES_BEFORE_TEAM_HOME
        -> FG_PCT_SEASON_BEFORE_AVG_TEAM_HOME
    """
    m = re.search(r"_BEFORE_TEAM_(HOME|AWAY)$", col)
    if not m:
        return None
    side = m.group(1)

    # Remove the trailing side marker
    col_core = re.sub(r"_BEFORE_TEAM_(HOME|AWAY)$", "", col)

    # Remove rolling suffix markers
    for pat in INFER_ROLLING_PATTERNS_TO_REMOVE:
        col_core = re.sub(pat, "", col_core)

    # If still indicates "LAST" then mapping is ambiguous; skip
    if "LAST" in col_core:
        return None

    return f"{col_core}_SEASON_BEFORE_AVG_TEAM_{side}"


def resolve_policy(
    df: pd.DataFrame,
    *,
    current_total_line_col: str | None,
    drop_mode: Literal["strict", "moderate", "lenient"],
) -> MissingPolicy:
    """
    Resolve policy lists for a specific DataFrame (only keep columns that exist).
    """
    # Drop columns: odds + core team stats + top player critical columns
    drop_cols = []
    drop_cols += _existing(df, DROP_IF_MISSING_EXACT)
    if current_total_line_col:
        drop_cols += _existing(df, [current_total_line_col])
    drop_cols += _existing(df, DROP_IF_MISSING_CORE_TEAM_STATS_EXACT)
    drop_cols += _resolve_top_player_cols(df, drop_mode)

    # Zero-fill columns: by explicit pattern rules, numeric only is enforced later
    zero_cols = [c for c in df.columns if _is_zero_fill_col(c)]

    # Infer pairs: for columns with a known season average fallback present in df
    infer_pairs: list[tuple[str, str]] = []
    for c in df.columns:
        fb = _build_season_avg_fallback(c)
        if fb and fb in df.columns:
            infer_pairs.append((c, fb))

    # Ensure no overlap confusion: if a col is in drop, it should not be in zero/infer.
    drop_set = set(drop_cols)
    zero_cols = [c for c in zero_cols if c not in drop_set]
    infer_pairs = [(c, fb) for (c, fb) in infer_pairs if c not in drop_set]

    return MissingPolicy(
        drop_cols=sorted(set(drop_cols)),
        zero_cols=sorted(set(zero_cols)),
        infer_pairs=infer_pairs,
    )


# ------------------------------------------------------------------------------
# 3) APPLY POLICY
# ------------------------------------------------------------------------------


def apply_missing_policy(
    df: pd.DataFrame,
    *,
    train_medians: pd.Series | None = None,
    current_total_line_col: str | None = None,
    drop_mode: Literal["strict", "moderate", "lenient"] = "strict",
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the deterministic policy:
      1) Drop rows missing in DROP set
      2) Zero-fill in ZERO set (numeric columns only)
      3) Infer rolling features from season averages using INFER pairs
      4) Compute implied points (optional)
      5) Final fallback to train medians (optional)
    """
    out = df.copy()
    before_rows = int(len(out))

    policy = resolve_policy(
        out, current_total_line_col=current_total_line_col, drop_mode=drop_mode
    )

    # 1) DROP
    dropped_reasons: dict[str, int] = {}
    if policy.drop_cols:
        drop_mask = out[policy.drop_cols].isna().any(axis=1)
        dropped_reasons["missing_drop_cols_any"] = int(drop_mask.sum())
        out = out.loc[~drop_mask].copy()
    dropped = before_rows - int(len(out))

    # 2) ZERO-FILL (numeric only)
    zero_cols_numeric = [
        c for c in policy.zero_cols if c in out.columns and is_numeric_dtype(out[c])
    ]
    if zero_cols_numeric:
        out.loc[:, zero_cols_numeric] = out.loc[:, zero_cols_numeric].fillna(0.0)

    # 3) INFER from season average (only for numeric columns; do not coerce objects)
    infer_applied = 0
    for c, fb in policy.infer_pairs:
        if (
            c in out.columns
            and fb in out.columns
            and is_numeric_dtype(out[c])
            and is_numeric_dtype(out[fb])
        ):
            before_na = int(out[c].isna().sum())
            if before_na:
                out[c] = out[c].fillna(out[fb])
                after_na = int(out[c].isna().sum())
                if after_na < before_na:
                    infer_applied += 1

    # 4) IMPLIED POINTS (optional; assumes SPREAD is home perspective with standard sign)
    if (
        current_total_line_col
        and current_total_line_col in out.columns
        and "SPREAD" in out.columns
    ):
        total = out[current_total_line_col]
        spread = out["SPREAD"]
        if IMPLIED_PTS_COLS[0] in out.columns and is_numeric_dtype(
            out[IMPLIED_PTS_COLS[0]]
        ):
            out[IMPLIED_PTS_COLS[0]] = out[IMPLIED_PTS_COLS[0]].fillna(
                total / 2.0 - spread / 2.0
            )
        if IMPLIED_PTS_COLS[1] in out.columns and is_numeric_dtype(
            out[IMPLIED_PTS_COLS[1]]
        ):
            out[IMPLIED_PTS_COLS[1]] = out[IMPLIED_PTS_COLS[1]].fillna(
                total / 2.0 + spread / 2.0
            )

    # 5) FINAL FALLBACK: TRAIN MEDIANS
    if train_medians is not None:
        common = [
            c
            for c in out.columns
            if c in train_medians.index and is_numeric_dtype(out[c])
        ]
        if common:
            out.loc[:, common] = out.loc[:, common].fillna(train_medians[common])

    report = {
        "rows_in": before_rows,
        "rows_dropped": int(dropped),
        "rows_out": int(len(out)),
        "drop_rate_pct": round(100.0 * dropped / before_rows, 2)
        if before_rows
        else 0.0,
        "drop_cols_count": int(len(policy.drop_cols)),
        "zero_cols_count": int(len(zero_cols_numeric)),
        "infer_pairs_count": int(len(policy.infer_pairs)),
        "infer_cols_applied_count": int(infer_applied),
        "remaining_na_cells": int(out.isna().sum().sum()),
        "dropped_reasons": dropped_reasons,
        # Make the policy auditable:
        "drop_cols": policy.drop_cols,
        "zero_cols": zero_cols_numeric,
        "infer_pairs_sample": policy.infer_pairs[:25],  # keep report reasonably sized
    }

    return out, report


# ------------------------------------------------------------------------------
# 4) UTILITIES
# ------------------------------------------------------------------------------


def compute_and_save_train_medians(
    df_train: pd.DataFrame, output_path: str
) -> pd.Series:
    """
    Compute medians from training data (numeric only) and save as a 1-column CSV.
    """
    med = df_train.median(numeric_only=True)
    med.to_csv(output_path, header=["median"])
    return med


def load_train_medians(path: str) -> pd.Series:
    """
    Load medians saved by compute_and_save_train_medians.
    """
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] != 1:
        raise ValueError(f"Expected 1 column in medians CSV, got {df.shape[1]}")
    return df.iloc[:, 0]


def diagnose_missing_data(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Return a DataFrame with missing_count and missing_pct per column (top_n by count).
    """
    na_counts = df.isna().sum()
    na_pct = 100.0 * na_counts / len(df) if len(df) else 0.0

    rep = (
        pd.DataFrame({"missing_count": na_counts, "missing_pct": na_pct})
        .loc[lambda x: x["missing_count"] > 0]
        .sort_values("missing_count", ascending=False)
        .head(top_n)
    )
    return rep
