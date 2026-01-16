import numpy as np
import pandas as pd

path = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data/"

FAST_PATH = path + "fast.csv"
SLOW_PATH = path + "slow.csv"

# ----------------------------
# 1) Load (keep empty strings as empty, but also treat typical NA tokens as NaN)
# ----------------------------
na_tokens = ["", "NA", "N/A", "null", "None", "nan", "NaN"]

fast = pd.read_csv(FAST_PATH, na_values=na_tokens, keep_default_na=True)
slow = pd.read_csv(SLOW_PATH, na_values=na_tokens, keep_default_na=True)

print(f"Loaded fast: {fast.shape}, slow: {slow.shape}")

# ----------------------------
# 2) Quick structural checks
# ----------------------------
same_columns = list(fast.columns) == list(slow.columns)
print(f"Same column order: {same_columns}")

if not same_columns:
    only_in_fast = [c for c in fast.columns if c not in slow.columns]
    only_in_slow = [c for c in slow.columns if c not in fast.columns]
    print("Columns only in fast:", only_in_fast)
    print("Columns only in slow:", only_in_slow)

# Align to common columns (still keep order from slow for stable diffs)
common_cols = [c for c in slow.columns if c in fast.columns]
fast = fast[common_cols].copy()
slow = slow[common_cols].copy()


# ----------------------------
# 3) Canonicalize dtypes and missing values to reduce false diffs
#    - Convert obvious numeric columns to numeric when possible
#    - Treat None/NaN consistently
# ----------------------------
def coerce_numeric_where_possible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            # try numeric coercion; if it produces some non-null numbers, keep it
            coerced = pd.to_numeric(out[c], errors="coerce")
            if (
                coerced.notna().sum() > 0
                and coerced.notna().sum() >= out[c].notna().sum() * 0.5
            ):
                out[c] = coerced
    return out


fast = coerce_numeric_where_possible(fast)
slow = coerce_numeric_where_possible(slow)

# Normalize NaNs (important for equality checks)
fast = fast.replace({None: np.nan})
slow = slow.replace({None: np.nan})

# ----------------------------
# 4) Exact equality check (including NaN positions)
# ----------------------------
# pandas.DataFrame.equals treats NaN == NaN as True (which is what you want here)
exact_same = fast.equals(slow)
print(f"EXACT SAME (after alignment/normalization): {exact_same}")

if exact_same:
    raise SystemExit(0)

# ----------------------------
# 5) Differences summary
# ----------------------------
# (a) where values differ at the cell level (NaNs handled)
neq = (fast.ne(slow)) & ~(fast.isna() & slow.isna())
n_diff_cells = int(neq.to_numpy().sum())
n_diff_rows = int(neq.any(axis=1).sum())
n_diff_cols = int(neq.any(axis=0).sum())

print(f"Different cells: {n_diff_cells}")
print(f"Rows with any diff: {n_diff_rows} / {len(fast)}")
print(f"Cols with any diff: {n_diff_cols} / {fast.shape[1]}")

# Top columns by number of diffs
col_diff_counts = neq.sum(axis=0).sort_values(ascending=False)
print("\nTop columns by diff count (non-zero only):")
print(col_diff_counts[col_diff_counts > 0].head(30))

# (b) dtype differences (can explain object vs float diffs)
dtype_diff = pd.DataFrame({"fast_dtype": fast.dtypes, "slow_dtype": slow.dtypes})
dtype_diff = dtype_diff[dtype_diff["fast_dtype"] != dtype_diff["slow_dtype"]]
if not dtype_diff.empty:
    print("\nColumns with dtype differences:")
    print(dtype_diff)

# ----------------------------
# 6) Detailed diff table using pandas.compare
# ----------------------------
# compare shows (self, other) side-by-side for differing cells
diff = slow.compare(fast, keep_equal=False)  # "slow" as baseline, "fast" as new
print(f"\ncompare() output shape: {diff.shape}")

# Save a compact diff CSV (useful if large)
diff.to_csv(f"{path}diff_slow_vs_fast.csv", index=True)
print(f"Wrote: {path}diff_slow_vs_fast.csv")

# ----------------------------
# 7) Row-level diff extract (first N differing rows)
# ----------------------------
N = 25
row_idx = np.where(neq.any(axis=1).to_numpy())[0]
print(f"\nFirst {min(N, len(row_idx))} differing row indices: {row_idx[:N].tolist()}")

# Create a readable row diff dump for the first N rows:
rows = row_idx[:N]
row_dump = []
for r in rows:
    cols = neq.columns[neq.iloc[r].to_numpy()].tolist()
    for c in cols:
        row_dump.append(
            {
                "row": int(r),
                "column": c,
                "slow": slow.iloc[r][c],
                "fast": fast.iloc[r][c],
            }
        )
row_dump_df = pd.DataFrame(row_dump)
row_dump_df.to_csv(f"{path}row_level_differences_first_rows.csv", index=False)
print(f"Wrote: {path}row_level_differences_first_rows.csv")

# ----------------------------
# 8) Optional: float tolerance check (if you suspect tiny rounding diffs)
# ----------------------------
# This DOES NOT affect the "exact" verdict above. It's just diagnostics.
float_cols = [
    c
    for c in fast.columns
    if pd.api.types.is_float_dtype(fast[c]) or pd.api.types.is_float_dtype(slow[c])
]
if float_cols:
    max_abs = {}
    for c in float_cols:
        a = pd.to_numeric(slow[c], errors="coerce")
        b = pd.to_numeric(fast[c], errors="coerce")
        m = np.nanmax(np.abs((a - b).to_numpy()))
        max_abs[c] = m
    max_abs_s = pd.Series(max_abs).sort_values(ascending=False)
    print("\nTop float columns by max absolute difference:")
    print(max_abs_s.head(20))
