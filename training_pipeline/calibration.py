"""Probability quality for the over/under classifier -- measured, not fixed.

Phase 1 deliberately only *reports* these. Nothing here adjusts a probability;
the point is to find out whether raw XGBoost output is trustworthy enough to
bet against a break-even rate before deciding whether a calibrator is worth the
cost of refitting it inside the daily walk-forward.

Two distinct properties, easy to conflate:

- **Discrimination** -- do the model's higher probabilities correspond to games
  that actually go over more often? Without this, nothing else matters.
- **Calibration** -- when it says 60%, does it happen ~60% of the time? A model
  can discriminate perfectly and still be badly calibrated (a monotone
  transform of a good score), which matters here because the betting rule
  compares the probability to an absolute threshold (52.38% at -110). An
  overconfident model bets far too often.

Why log loss is the tuning objective rather than accuracy: it scores the
probability, not just the side. Calling a loser at 51% is barely penalised;
calling it at 95% is punished hard. Accuracy treats those identically, which
is exactly the distinction the betting rule depends on.

Scale caveat worth carrying: on a ~50/50 outcome log loss has almost no dynamic
range. A perfectly calibrated 55% model scores 0.68814 against 0.69315 for a
coin flip -- the entire distance between "worthless" and "good" is 0.005. Read
these numbers against that spread, not against zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

#: Probabilities are clipped before taking logs. Log loss is unbounded as p
#: approaches 0 or 1, so a single confidently wrong prediction would otherwise
#: dominate the mean and make trials incomparable.
_EPSILON = 1e-9


def log_loss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean binary cross-entropy. Lower is better; 0.69315 == a coin flip."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), _EPSILON, 1.0 - _EPSILON)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean squared error of the probability. 0.25 == a coin flip."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


class CalibrationSummary(BaseModel):
    """Headline probability-quality numbers for one set of predictions."""

    n: int
    base_rate: float
    mean_predicted: float
    log_loss: float
    brier: float
    #: Log loss of always predicting the base rate. The only honest reference
    #: point: beating 0.69315 is trivial if the classes are unbalanced, whereas
    #: beating this means the features contributed something.
    log_loss_base_rate: float
    log_loss_improvement: float
    brier_base_rate: float
    #: Expected calibration error: average |predicted - observed| across
    #: buckets, weighted by bucket size.
    expected_calibration_error: float
    #: Mean predicted minus observed frequency. Positive => systematically
    #: overconfident that games go OVER.
    mean_bias: float


def calibration_table(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Predicted probability vs observed frequency, bucketed.

    Equal-width buckets over [0, 1] rather than quantile buckets: the betting
    rule cares about absolute probability levels (is this above 52.38%?), so
    the buckets should be absolute too. Empty buckets are dropped rather than
    reported as zeros, which would read as "always wrong" instead of "never
    predicted".
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    # Rightmost edge inclusive so p == 1.0 lands in the last bucket.
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_buckets - 1)

    rows: list[dict[str, float]] = []
    for bucket in range(n_buckets):
        mask = index == bucket
        n = int(mask.sum())
        if n == 0:
            continue
        observed = float(y[mask].mean())
        predicted = float(p[mask].mean())
        rows.append(
            {
                "bucket_low": float(edges[bucket]),
                "bucket_high": float(edges[bucket + 1]),
                "n": n,
                "mean_predicted": predicted,
                "observed_frequency": observed,
                # Positive => the model was overconfident in this bucket.
                "bias": predicted - observed,
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    n_buckets: int = 10,
) -> CalibrationSummary:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]

    if len(y) == 0:
        raise ValueError("No valid (finite) rows to score probabilities against.")

    base_rate = float(y.mean())
    base = np.full_like(p, base_rate)

    table = calibration_table(y, p, n_buckets=n_buckets)
    if table.empty:
        ece = float("nan")
        mean_bias = float("nan")
    else:
        weights = table["n"].to_numpy(dtype=float)
        ece = float(
            np.average(table["bias"].abs().to_numpy(dtype=float), weights=weights)
        )
        mean_bias = float(
            np.average(table["bias"].to_numpy(dtype=float), weights=weights)
        )

    model_log_loss = log_loss(y, p)
    base_log_loss = log_loss(y, base)

    return CalibrationSummary(
        n=int(len(y)),
        base_rate=base_rate,
        mean_predicted=float(p.mean()),
        log_loss=model_log_loss,
        brier=brier_score(y, p),
        log_loss_base_rate=base_log_loss,
        # Positive => better than predicting the base rate for every game.
        log_loss_improvement=base_log_loss - model_log_loss,
        brier_base_rate=brier_score(y, base),
        expected_calibration_error=ece,
        mean_bias=mean_bias,
    )
