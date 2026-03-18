import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


def evaluate_total_points_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test_total: pd.Series,
    line_col: str,
    thresholds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    y_pred_total = np.asarray(model.predict(X_test), dtype=float)
    betting_line = pd.to_numeric(X_test[line_col], errors="coerce").to_numpy(
        dtype=float
    )

    pred_edge = y_pred_total - betting_line
    margin = np.abs(pred_edge)

    rows = []
    n_total = len(y_test_total)
    y_true_total_np = y_test_total.to_numpy(dtype=float)

    for t in thresholds:
        mask = margin > t
        n = int(mask.sum())

        acc = (
            np.nan
            if n == 0
            else over_under_betting_accuracy_total_points(
                y_true=y_true_total_np[mask],
                y_pred=y_pred_total[mask],
                betting_line=betting_line[mask],
            )
        )

        rows.append(
            {
                "threshold_abs_pred_edge_gt": t,
                "n_games": n,
                "pct_of_test": (n / n_total) if n_total else np.nan,
                "ou_betting_accuracy": acc,
            }
        )

    return pd.DataFrame(rows), y_pred_total


def over_under_betting_accuracy_total_points(y_true, y_pred, betting_line) -> float:
    """
    Betting accuracy when the model predicts total points directly.

    Parameters
    ----------
    y_true : array-like
        True game total points.
    y_pred : array-like
        Predicted game total points.
    betting_line : array-like
        Sportsbook total line for each game.

    Returns
    -------
    float
        Fraction of non-push cases where predicted side matches actual side.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    betting_line = np.asarray(betting_line, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(betting_line)
    if not np.any(valid):
        return 0.0

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    betting_line = betting_line[valid]

    true_side = np.sign(y_true - betting_line)
    pred_side = np.sign(y_pred - betting_line)

    # remove pushes
    non_push = (true_side != 0) & (pred_side != 0)
    if not np.any(non_push):
        return 0.0

    return float(np.mean(true_side[non_push] == pred_side[non_push]))


class OverUnderScorerTotalPoints:
    """
    sklearn-compatible scorer for models predicting total points.
    It reads the betting line from X[line_col].
    """

    def __init__(self, line_col: str):
        self.line_col = line_col

    def __call__(self, estimator, X, y_true):
        if self.line_col not in X.columns:
            raise KeyError(f"{self.line_col} not found in X for scoring")

        y_pred = estimator.predict(X)
        betting_line = pd.to_numeric(X[self.line_col], errors="coerce").to_numpy(
            dtype=float
        )

        return over_under_betting_accuracy_total_points(
            y_true=np.asarray(y_true, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
            betting_line=betting_line,
        )
