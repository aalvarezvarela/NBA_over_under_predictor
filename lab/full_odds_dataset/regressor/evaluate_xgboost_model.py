import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nba_ou.data_preparation.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = Path(
    "/home/adrian_alvarez/Projects/NBA_over_under_predictor/models/dif_to_line/xgb_line_error_22_01.joblib"
)
META_PATH = Path(
    "/home/adrian_alvarez/Projects/NBA_over_under_predictor/models/dif_to_line/xgb_line_error_22_01.meta.json"
)

INPUT_CSV = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/train_data/all_odds_training_data_until_20260213.csv"
# OUTPUT_CSV = Path("/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/preds_xgb_line_error_22_01.csv")

with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

meta.keys()

model = joblib.load(MODEL_PATH)
type(model)
model.feature_names_in_


def infer_feature_columns(model, meta: dict) -> list[str]:
    # 1) Prefer meta
    for k in ["feature_columns", "feature_cols", "columns", "x_columns", "features"]:
        v = meta.get(k)
        if isinstance(v, list) and len(v) > 0:
            return [str(c) for c in v]

    # 2) sklearn convention
    v = getattr(model, "feature_names_in_", None)
    if v is not None:
        return [str(c) for c in list(v)]

    # 3) xgboost sklearn wrapper sometimes has get_booster().feature_names
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            return [str(c) for c in booster.feature_names]

    raise RuntimeError("Could not infer expected feature columns from meta or model.")


feature_cols = infer_feature_columns(model, meta)
len(feature_cols), feature_cols[:10]

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
df_raw = df_raw[df_raw["GAME_DATE"] > "2026-01-20"]
df_clean = clean_dataframe_for_training(
    df_raw,
    nan_threshold=100,
    drop_all_na_rows=DROP_ALL_NA_ROWS,
    keep_all_cols=True,
    verbose=VERBOSE_CLEANING,
)

df_clean.shape


def build_X_for_model(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    drop_cols = ["TOTAL_POINTS", "LINE_ERROR", "SEASON_YEAR"]
    X = df.drop(columns=drop_cols, errors="ignore")

    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        print(
            f"Missing {len(missing)} expected columns, creating them as NaN. First 20: {missing[:20]}"
        )
        for c in missing:
            X[c] = np.nan

    # Drop extras and order exactly
    X = X.loc[:, feature_cols]
    return X


X_pred = build_X_for_model(df_clean, feature_cols)
X_pred.shape
pred_line_error = model.predict(X_pred)
pred_line_error[:10], len(pred_line_error)

df_out = df_clean.copy()
df_out["PRED_LINE_ERROR"] = pred_line_error
df_out["PRED_PICK"] = np.where(
    df_out["PRED_LINE_ERROR"] > 0,
    "OVER",
    np.where(df_out["PRED_LINE_ERROR"] < 0, "UNDER", "PUSH"),
)

if "TOTAL_OVER_UNDER_LINE" in df_out.columns:
    df_out["PRED_TOTAL_POINTS"] = df_out["TOTAL_OVER_UNDER_LINE"].astype(
        float
    ) + df_out["PRED_LINE_ERROR"].astype(float)


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


def evaluate_error_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test_error: pd.Series,
    thresholds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    """
    Evaluate directional accuracy at different confidence thresholds.

    Threshold rule:
    - Include a game if abs(predicted_error) > t
    """
    # Predict error directly
    y_pred_error = np.asarray(model.predict(X_test), dtype=float)

    margin = np.abs(y_pred_error)

    rows = []
    n_total = len(y_test_error)

    y_true_error_np = y_test_error.to_numpy(dtype=float)

    for t in thresholds:
        mask = margin > t
        n = int(mask.sum())

        acc = (
            np.nan
            if n == 0
            else error_sign_accuracy(y_true_error_np[mask], y_pred_error[mask])
        )

        rows.append(
            {
                "threshold_abs_pred_error_gt": t,
                "n_games": n,
                "pct_of_test": (n / n_total) if n_total else np.nan,
                "directional_accuracy": acc,
            }
        )

    return pd.DataFrame(rows), y_pred_error


can_score = ("TOTAL_POINTS" in df_out.columns) and (
    "TOTAL_OVER_UNDER_LINE" in df_out.columns
)

if not can_score:
    print(
        "Skipping evaluation, CSV does not include TOTAL_POINTS and TOTAL_OVER_UNDER_LINE."
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
    bet_acc = over_under_betting_accuracy(y_true, y_pred)

    print("MSE:", mse)
    print("R2:", r2)
    print("MAE:", mae)
    print("Over Under betting accuracy:", bet_acc)

    results_df, y_pred_test_error = evaluate_error_thresholds(
        model=model,
        X_test=X_pred,
        y_test_error=df_out["TRUE_LINE_ERROR"],  # y_test must be the REAL ERROR series
        thresholds=range(0, 11),
    )
    quant_df = evaluate_by_confidence_quantiles(df_out["TRUE_LINE_ERROR"], y_pred)

    print("\nThreshold-based Evaluation:")
    display(
        results_df.style.format(
            {"pct_of_test": "{:.1%}", "directional_accuracy": "{:.2%}"}
        )
    )
    print("\nQuantile-based Evaluation:")
    display(quant_df)

print("meta feature count:", len(feature_cols))
print("X_pred shape:", X_pred.shape)
print("Missing in input:", [c for c in feature_cols if c not in df_clean.columns][:30])
print("Extra in input:", [c for c in df_clean.columns if c not in feature_cols][:30])
