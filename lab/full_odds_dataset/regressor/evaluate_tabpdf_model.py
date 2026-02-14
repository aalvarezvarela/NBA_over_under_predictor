# ============================================================
# TabPFN NBA over under: load fitted model and predict on new CSV
# Uses your project cleaning:
#   from nba_ou.data_preparation.missing_data.clean_df_for_training import clean_dataframe_for_training
# ============================================================

import os

import numpy as np
import pandas as pd

# IMPORTANT: your cleaning function
from nba_ou.data_preparation.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from tabpfn.model_loading import load_fitted_tabpfn_model

# Optional: only needed if you want the cloud backend
# pip install tabpfn-client
# import tabpfn_client
# from tabpfn_client import TabPFNRegressor as CloudTabPFNRegressor


# ============================================================
# 0) Config (edit these)
# ============================================================

BACKEND = "local"  # "local" or "client"
DEVICE = "cpu"  # "cpu" or "cuda" (only used for BACKEND="local")

MODEL_PATH = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/models/my_reg_total.tabpfn_fit"
INPUT_CSV = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data/all_odds_training_data_until_20260213.csv"
OUTPUT_CSV = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/preds_reg_total_2024.csv"

# Your requested cleaning settings
NAN_THRESHOLD = 0
DROP_ALL_NA_ROWS = False
VERBOSE_CLEANING = 1


# ============================================================
# 1) Load input CSV (keep ID columns as string, normalize GAME_DATE)
# ============================================================

df_head = pd.read_csv(INPUT_CSV, nrows=5)
dtype_dict = {col: str for col in df_head.columns if "ID" in col.upper()}

df_raw = pd.read_csv(INPUT_CSV, dtype=dtype_dict)

if "GAME_DATE" in df_raw.columns:
    df_raw["GAME_DATE"] = pd.to_datetime(
        df_raw["GAME_DATE"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

df_raw.shape, df_raw.columns[:10]


# ============================================================
# 2) Clean with your project function (as requested)
# ============================================================
df_raw = df_raw[df_raw["GAME_DATE"] > "2026-01-10"]
df_clean = clean_dataframe_for_training(
    df_raw,
    nan_threshold=100,
    drop_all_na_rows=DROP_ALL_NA_ROWS,
    keep_correlated_and_constant=True,

    verbose=VERBOSE_CLEANING,
)
    
df_clean.shape


# ============================================================
# 3) Load fitted TabPFN model (local CPU or CUDA)
# ============================================================

local_model_for_schema = load_fitted_tabpfn_model(MODEL_PATH, device="cpu")

feature_cols = getattr(local_model_for_schema, "feature_names_in_", None)
if feature_cols is None:
    # fallback: try to recover from embedded training data
    for attr in ["X_train_", "X_", "X_train", "Xtr_", "train_X_"]:
        Xmaybe = getattr(local_model_for_schema, attr, None)
        if isinstance(Xmaybe, pd.DataFrame):
            feature_cols = Xmaybe.columns
            break

if feature_cols is None:
    raise RuntimeError(
        "Could not infer feature columns from the fitted model. "
        "Make sure the fitted model exposes feature_names_in_ or stores X_train_ as a DataFrame."
    )

feature_cols = list(map(str, feature_cols))
len(feature_cols), feature_cols[:10]


# ============================================================
# 4) Build X for prediction (drop training-only cols, align to feature schema)
# ============================================================

drop_cols = ["TOTAL_POINTS", "LINE_ERROR", "SEASON_YEAR"]
X_pred = df_clean.drop(columns=drop_cols, errors="ignore")

missing = [c for c in feature_cols if c not in X_pred.columns]
if missing:
    print(
        f"Missing {len(missing)} feature cols in input. Creating them as NaN. First 20:\n{missing[:20]}"
    )
    for c in missing:
        X_pred[c] = np.nan

# Drop extra columns and order exactly as training schema
X_pred = X_pred.loc[:, feature_cols]

X_pred.shape


# ============================================================
# 5) Predict (local or client)
# ============================================================

if BACKEND == "local":
    model = load_fitted_tabpfn_model(MODEL_PATH, device=DEVICE)
    pred_error = model.predict(X_pred.head(1))

elif BACKEND == "client":
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor as CloudTabPFNRegressor

    token = os.environ.get("TABPFN_ACCESS_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
    else:
        # will try interactive auth depending on environment
        tabpfn_client.set_access_token(tabpfn_client.get_access_token())

    # For client we need training data. Try to extract from the fitted model.
    X_train = None
    y_train = None

    for a in ["X_train_", "X_", "X_train", "Xtr_", "train_X_"]:
        v = getattr(local_model_for_schema, a, None)
        if isinstance(v, pd.DataFrame):
            X_train = v
            break
        if isinstance(v, np.ndarray):
            X_train = pd.DataFrame(v, columns=feature_cols)
            break

    for a in ["y_train_", "y_", "y_train", "ytr_", "train_y_"]:
        v = getattr(local_model_for_schema, a, None)
        if isinstance(v, (pd.Series, np.ndarray, list)):
            y_train = np.asarray(v, dtype=float)
            break

    if X_train is None or y_train is None:
        raise RuntimeError(
            "Could not extract training data from the fitted model for client backend. "
            "If your .tabpfn_fit does not embed training data, fit the cloud model from your historical training CSV."
        )

    cloud_model = CloudTabPFNRegressor()
    cloud_model.fit(X_train[feature_cols], y_train)

    pred_error = cloud_model.predict(X_pred)

else:
    raise ValueError("BACKEND must be 'local' or 'client'")

pred_error[:10], len(pred_error)


# ============================================================
# 6) Add prediction columns and save
# ============================================================

df_out = df_clean.copy()
df_out["PRED_LINE_ERROR"] = pred_error
df_out["PRED_PICK"] = np.where(
    df_out["PRED_LINE_ERROR"] > 0,
    "OVER",
    np.where(df_out["PRED_LINE_ERROR"] < 0, "UNDER", "PUSH"),
)

if "TOTAL_OVER_UNDER_LINE" in df_out.columns:
    df_out["PRED_TOTAL_POINTS"] = df_out["TOTAL_OVER_UNDER_LINE"].astype(
        float
    ) + df_out["PRED_LINE_ERROR"].astype(float)

df_out.to_csv(OUTPUT_CSV, index=False)
OUTPUT_CSV, df_out.shape


# ============================================================
# 7) Accuracy / evaluation like your initial script
#    Only runs if TOTAL_POINTS and TOTAL_OVER_UNDER_LINE exist in the input.
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def over_under_betting_accuracy(
    true_error: np.ndarray, pred_error: np.ndarray
) -> float:
    true_sign = np.sign(true_error)
    pred_sign = np.sign(pred_error)
    valid = (true_sign != 0) & (pred_sign != 0)
    if not np.any(valid):
        return np.nan
    return float(np.mean(true_sign[valid] == pred_sign[valid]))


def error_sign_accuracy(
    y_true_error, y_pred_error, include_pushes: bool = False
) -> float:
    y_true_error = np.asarray(y_true_error, dtype=float)
    y_pred_error = np.asarray(y_pred_error, dtype=float)

    true_sign = np.sign(y_true_error)
    pred_sign = np.sign(y_pred_error)

    if include_pushes:
        valid = ~((true_sign == 0) ^ (pred_sign == 0))
    else:
        valid = (true_sign != 0) & (pred_sign != 0)

    if not np.any(valid):
        return np.nan

    return float(np.mean(true_sign[valid] == pred_sign[valid]))


def evaluate_by_confidence_quantiles(
    y_true_error: pd.Series,
    y_pred_error: np.ndarray,
    quantiles=(0.05, 0.10, 0.20, 0.30, 0.50, 1.0),
    include_pushes: bool = False,
) -> pd.DataFrame:
    y_true = y_true_error.to_numpy(dtype=float)
    y_pred = np.asarray(y_pred_error, dtype=float)

    abs_pred = np.abs(y_pred)
    n_total = len(abs_pred)

    order = np.argsort(-abs_pred)

    rows = []
    for q in quantiles:
        k = int(np.ceil(q * n_total))
        idx = order[:k]

        acc = (
            np.nan
            if k == 0
            else error_sign_accuracy(
                y_true[idx], y_pred[idx], include_pushes=include_pushes
            )
        )

        rows.append(
            {
                "top_confidence_fraction": float(q),
                "n_games": k,
                "pct_of_test": (k / n_total) if n_total else np.nan,
                "directional_accuracy": acc,
                "min_abs_pred_error": float(abs_pred[idx].min()) if k else np.nan,
                "mean_abs_pred_error": float(abs_pred[idx].mean()) if k else np.nan,
            }
        )

    return pd.DataFrame(rows)


def evaluate_by_abs_pred_error_thresholds(
    y_true_error: pd.Series,
    y_pred_error: np.ndarray,
    thresholds=np.arange(0.0, 5.0, 0.2),
    include_pushes: bool = False,
) -> pd.DataFrame:
    y_true = y_true_error.to_numpy(dtype=float)
    y_pred = np.asarray(y_pred_error, dtype=float)

    abs_pred = np.abs(y_pred)
    n_total = len(y_true)

    rows = []
    for t in thresholds:
        mask = abs_pred > t
        n = int(mask.sum())

        acc = (
            np.nan
            if n == 0
            else error_sign_accuracy(
                y_true[mask], y_pred[mask], include_pushes=include_pushes
            )
        )

        rows.append(
            {
                "abs_pred_error_gt": float(t),
                "n_games": n,
                "pct_of_test": (n / n_total) if n_total else np.nan,
                "directional_accuracy": acc,
                "mean_abs_pred_error": float(abs_pred[mask].mean()) if n else np.nan,
            }
        )

    return pd.DataFrame(rows)


can_score = ("TOTAL_POINTS" in df_out.columns) and (
    "TOTAL_OVER_UNDER_LINE" in df_out.columns
)

if not can_score:
    print(
        "Skipping evaluation: need TOTAL_POINTS and TOTAL_OVER_UNDER_LINE in the CSV to compute true LINE_ERROR."
    )
else:
    df_out["TRUE_LINE_ERROR"] = df_out["TOTAL_POINTS"].astype(float) - df_out[
        "TOTAL_OVER_UNDER_LINE"
    ].astype(float)

    y_true = df_out["TRUE_LINE_ERROR"].to_numpy(dtype=float)
    y_pred = df_out["PRED_LINE_ERROR"].to_numpy(dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    betting_acc = over_under_betting_accuracy(y_true, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("R2 Score:", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Over/Under Betting Accuracy:", betting_acc)

    thr_df = evaluate_by_abs_pred_error_thresholds(
        y_true_error=df_out["TRUE_LINE_ERROR"],
        y_pred_error=y_pred,
        thresholds=np.arange(0.0, 5.0, 0.2),
    )

    quant_df = evaluate_by_confidence_quantiles(
        y_true_error=df_out["TRUE_LINE_ERROR"],
        y_pred_error=y_pred,
    )

    # If you are in a notebook, display(thr_df) and display(quant_df) is fine
    print("\nThreshold based evaluation (head):")
    print(thr_df.head(10).to_string(index=False))

    print("\nQuantile based evaluation:")
    print(quant_df.to_string(index=False))
