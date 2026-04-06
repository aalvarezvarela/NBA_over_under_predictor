from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, cast
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nba_ou.modeling.modeling import build_recency_sample_weights
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

PredictionType = Literal["total_points", "line_error"]
ResidualScaleMethod = Literal["std", "mad"]
CalibrationMethod = Literal["identity", "isotonic", "sigmoid"]
OddsFormat = Literal["american", "decimal"]
BinStrategy = Literal["quantile", "uniform"]

FitPredictFunction = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
SplitList = list[tuple[np.ndarray, np.ndarray]]
SplitBuilder = Callable[[pd.DataFrame], SplitList | tuple[SplitList, pd.DataFrame]]
SplitInput = SplitList | tuple[SplitList, pd.DataFrame] | SplitBuilder


@dataclass(frozen=True)
class ResidualScaleEstimate:
    """Leakage-free residual scale estimated from out-of-fold training residuals."""

    scale: float
    method: ResidualScaleMethod
    n_residuals: int
    residuals: np.ndarray
    oof_predictions: pd.DataFrame


@dataclass
class ProbabilityCalibrator:
    """
    Probability calibrator fitted on out-of-sample probabilities only.

    Notes
    -----
    When the calibration sample is too small or contains only one class, the
    calibrator falls back to the identity transform. This is deliberate because
    forcing a fit in that setting is unstable and would create misleading
    probabilities.
    """

    method: CalibrationMethod
    clip_eps: float = 1e-6
    min_train_samples: int = 25
    model: Any | None = None
    is_fitted: bool = False
    fallback_reason: str | None = None

    def fit(
        self,
        raw_probabilities: np.ndarray | pd.Series,
        outcomes: np.ndarray | pd.Series,
    ) -> "ProbabilityCalibrator":
        raw = _coerce_numeric_array(raw_probabilities, name="raw_probabilities")
        y = _coerce_numeric_array(outcomes, name="outcomes")

        valid = np.isfinite(raw) & np.isfinite(y)
        raw = raw[valid]
        y = y[valid]

        if self.method == "identity":
            self.is_fitted = True
            self.fallback_reason = None
            return self

        if raw.size < self.min_train_samples:
            self.is_fitted = False
            self.fallback_reason = (
                f"insufficient calibration samples ({raw.size} < {self.min_train_samples})"
            )
            return self

        unique_y = np.unique(y.astype(int))
        if unique_y.size < 2:
            self.is_fitted = False
            self.fallback_reason = "calibration target has fewer than two classes"
            return self

        raw = np.clip(raw, self.clip_eps, 1.0 - self.clip_eps)

        if self.method == "isotonic":
            self.model = IsotonicRegression(
                y_min=self.clip_eps,
                y_max=1.0 - self.clip_eps,
                out_of_bounds="clip",
            )
            self.model.fit(raw, y)
            self.is_fitted = True
            self.fallback_reason = None
            return self

        if self.method == "sigmoid":
            logit_raw = np.log(raw / (1.0 - raw)).reshape(-1, 1)
            self.model = LogisticRegression(
                solver="lbfgs",
                C=1e6,
                max_iter=1000,
                random_state=0,
            )
            self.model.fit(logit_raw, y.astype(int))
            self.is_fitted = True
            self.fallback_reason = None
            return self

        raise ValueError(f"Unsupported calibration method: {self.method}")

    def transform(
        self,
        raw_probabilities: np.ndarray | pd.Series,
    ) -> np.ndarray:
        raw = _coerce_numeric_array(raw_probabilities, name="raw_probabilities")
        raw = np.clip(raw, self.clip_eps, 1.0 - self.clip_eps)

        if self.method == "identity" or not self.is_fitted or self.model is None:
            return raw

        if self.method == "isotonic":
            calibrated = cast(IsotonicRegression, self.model).predict(raw)
        elif self.method == "sigmoid":
            logit_raw = np.log(raw / (1.0 - raw)).reshape(-1, 1)
            calibrated = cast(LogisticRegression, self.model).predict_proba(logit_raw)[
                :, 1
            ]
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

        return np.clip(np.asarray(calibrated, dtype=float), self.clip_eps, 1.0 - self.clip_eps)


@dataclass(frozen=True)
class FoldProbabilityEvaluationResult:
    """Full probability-workflow result for one outer fold."""

    predictions: pd.DataFrame
    strategy_summary: pd.DataFrame
    calibration_oof_predictions: pd.DataFrame
    residual_scale_estimate: ResidualScaleEstimate
    calibrators: dict[str, ProbabilityCalibrator]


@dataclass(frozen=True)
class ProbabilityBacktestResult:
    """Aggregated walk-forward backtest result."""

    predictions: pd.DataFrame
    fold_strategy_summary: pd.DataFrame
    overall_strategy_summary: pd.DataFrame
    calibration_oof_predictions: pd.DataFrame
    residual_scale_by_fold: pd.DataFrame


def get_model_bundle_feature_names(metadata: dict[str, Any]) -> list[str]:
    """
    Return the exact saved feature list from persisted model metadata.

    The comparison workflow must be anchored to the saved bundle schema rather
    than recomputing features ad hoc inside the notebook.
    """
    schema = metadata.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("Saved metadata is missing the schema section.")

    feature_names = schema.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Saved metadata is missing schema.feature_names.")

    resolved = [str(name) for name in feature_names]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Saved metadata contains duplicate feature names.")
    return resolved


def prepare_model_bundle_feature_frame(
    df: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    fill_missing_features: bool = True,
) -> pd.DataFrame:
    """
    Select the saved bundle feature columns in their persisted order.

    Notes
    -----
    Notebook evaluation data may occasionally be missing columns that existed
    when the bundle was trained, for example after a later cleaning pass drops
    sparse columns. When ``fill_missing_features`` is enabled, those columns are
    reintroduced as all-NaN so inference remains aligned with the persisted
    schema. This keeps the saved bundle as the source of truth while remaining
    explicit about the mismatch.
    """
    feature_names = get_model_bundle_feature_names(metadata)
    missing = [column for column in feature_names if column not in df.columns]
    if missing:
        if not fill_missing_features:
            raise KeyError(
                "Input dataframe is missing saved bundle feature columns. "
                f"Example missing columns: {missing[:10]}"
            )

        warnings.warn(
            "Input dataframe is missing saved bundle feature columns. "
            f"Filling {len(missing)} missing columns with NaN to preserve the "
            f"saved schema. Example missing columns: {missing[:10]}",
            RuntimeWarning,
            stacklevel=2,
        )

    aligned_df = df.copy()
    for column in missing:
        aligned_df[column] = np.nan
    return aligned_df.loc[:, feature_names].copy()


def resolve_model_bundle_training_params(
    metadata: dict[str, Any],
    *,
    objective_name: str = "reg:squarederror",
    min_n_estimators: int = 50,
    default_n_estimators: int = 75,
) -> tuple[dict[str, Any], float | None]:
    """
    Build training parameters from persisted bundle metadata only.

    Notes
    -----
    This mirrors the original refit logic used after Optuna tuning, but it
    reads only the saved metadata rather than any live Optuna study object.
    """
    training_metrics = metadata.get("training_metrics")
    if not isinstance(training_metrics, dict):
        raise ValueError("Saved metadata is missing the training_metrics section.")

    best_params = training_metrics.get("best_params")
    if not isinstance(best_params, dict) or not best_params:
        raise ValueError("Saved metadata is missing training_metrics.best_params.")

    tuned_params = dict(best_params)
    sample_weight_lambda = tuned_params.pop("sample_weight_lambda", None)
    if sample_weight_lambda is not None:
        sample_weight_lambda = float(sample_weight_lambda)
        if sample_weight_lambda < 0:
            raise ValueError("sample_weight_lambda from saved metadata must be >= 0.")

    final_n_estimators = (
        training_metrics.get("median_best_iteration")
        or training_metrics.get("mean_best_iteration")
        or tuned_params.get("n_estimators")
        or default_n_estimators
    )
    final_n_estimators = max(
        int(min_n_estimators),
        int(round(float(final_n_estimators))),
    )

    resolved_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": objective_name,
        "eval_metric": "mae",
        "random_state": 16,
        "n_jobs": -1,
        "verbosity": 0,
        "n_estimators": final_n_estimators,
        **tuned_params,
    }
    return resolved_params, sample_weight_lambda


def build_model_bundle_fit_predict_function(
    metadata: dict[str, Any],
    *,
    target_col: str,
    date_col: str = "GAME_DATE",
    objective_name: str = "reg:squarederror",
) -> FitPredictFunction:
    """
    Create a fold-local fit/predict function from saved bundle metadata.

    The returned callable retrains inside each fold using the persisted feature
    list and persisted model parameters only. It never reads Optuna study state.
    """
    feature_names = get_model_bundle_feature_names(metadata)
    model_params, sample_weight_lambda = resolve_model_bundle_training_params(
        metadata,
        objective_name=objective_name,
    )

    def fit_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
        X_train = prepare_model_bundle_feature_frame(train_df, metadata=metadata)
        X_valid = prepare_model_bundle_feature_frame(valid_df, metadata=metadata)
        y_train = pd.to_numeric(train_df[target_col], errors="coerce")

        if y_train.isna().any():
            raise ValueError(
                f"Training target column {target_col!r} contains missing or non-numeric values."
            )

        if list(X_train.columns) != feature_names or list(X_valid.columns) != feature_names:
            raise ValueError("Saved bundle feature ordering could not be preserved.")

        model = XGBRegressor(**model_params)
        fit_kwargs: dict[str, Any] = {"verbose": False}

        if sample_weight_lambda is not None:
            fit_kwargs["sample_weight"] = build_recency_sample_weights(
                train_df[date_col],
                lambda_=sample_weight_lambda,
            ).to_numpy(dtype=float)

        model.fit(X_train, y_train.to_numpy(dtype=float), **fit_kwargs)
        return np.asarray(model.predict(X_valid), dtype=float)

    return fit_predict


def predict_with_loaded_model_bundle(
    model: Any,
    df: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Generate predictions from a loaded persisted model using saved feature order."""
    X = prepare_model_bundle_feature_frame(df, metadata=metadata)
    predictions = np.asarray(model.predict(X), dtype=float)
    if predictions.shape[0] != len(df):
        raise ValueError(
            "Loaded model returned the wrong number of predictions. "
            f"Expected {len(df)}, got {predictions.shape[0]}."
        )
    return predictions


def convert_regression_predictions_to_over_probabilities(
    y_pred_reg: np.ndarray | pd.Series,
    *,
    line_values: np.ndarray | pd.Series | None,
    residual_scale: float,
    prediction_type: PredictionType,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert regression predictions into raw probabilities of an Over result.

    Parameters
    ----------
    y_pred_reg
        Regressor prediction. This is either predicted total points or predicted
        line error, depending on ``prediction_type``.
    line_values
        Sportsbook line values. Required when ``prediction_type="total_points"``.
    residual_scale
        Residual uncertainty estimate measured in points. This must come from
        training data only.
    prediction_type
        Either ``"total_points"`` or ``"line_error"``.
    clip_eps
        Probability clipping bound used to avoid exact 0/1 values.
    """
    if residual_scale <= 0 or not np.isfinite(residual_scale):
        raise ValueError("residual_scale must be a finite positive number.")

    pred_margin = compute_predicted_margin(
        y_pred_reg=y_pred_reg,
        line_values=line_values,
        prediction_type=prediction_type,
    )
    z_score = pred_margin / float(residual_scale)
    raw_prob = norm.cdf(z_score)
    return np.clip(np.asarray(raw_prob, dtype=float), clip_eps, 1.0 - clip_eps)


def compute_predicted_margin(
    *,
    y_pred_reg: np.ndarray | pd.Series,
    line_values: np.ndarray | pd.Series | None,
    prediction_type: PredictionType,
) -> np.ndarray:
    """Return predicted margin versus the closing total line."""
    y_pred = _coerce_numeric_array(y_pred_reg, name="y_pred_reg")

    if prediction_type == "line_error":
        return y_pred

    if line_values is None:
        raise ValueError("line_values are required when prediction_type='total_points'.")

    line = _coerce_numeric_array(line_values, name="line_values")
    if line.shape[0] != y_pred.shape[0]:
        raise ValueError("y_pred_reg and line_values must have the same length.")
    return y_pred - line


def collect_oof_regression_predictions(
    df: pd.DataFrame,
    *,
    splits: SplitInput,
    fit_predict_fn: FitPredictFunction,
    target_col: str,
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """Collect out-of-fold regression predictions for the provided time-aware splits."""
    resolved_splits = _resolve_splits(df, splits)
    rows: list[pd.DataFrame] = []

    for fold_number, (train_idx, valid_idx) in enumerate(resolved_splits, start=1):
        train_df = df.iloc[train_idx].copy()
        valid_df = df.iloc[valid_idx].copy()

        y_valid = pd.to_numeric(valid_df[target_col], errors="coerce").to_numpy(dtype=float)
        y_pred = np.asarray(fit_predict_fn(train_df, valid_df), dtype=float)

        if y_pred.shape[0] != len(valid_df):
            raise ValueError(
                "fit_predict_fn returned a prediction array with the wrong length. "
                f"Expected {len(valid_df)}, got {y_pred.shape[0]}."
            )

        fold_df = pd.DataFrame(
            {
                "fold": fold_number,
                "row_position": valid_idx,
                "row_index": valid_df.index.to_numpy(),
                "y_true": y_valid,
                "y_pred": y_pred,
            }
        )
        if date_col in valid_df.columns:
            fold_df[date_col] = pd.to_datetime(valid_df[date_col], errors="coerce").to_numpy()
        rows.append(fold_df)

    if not rows:
        raise ValueError("No out-of-fold predictions were generated.")

    oof_df = pd.concat(rows, ignore_index=True)
    if oof_df["row_position"].duplicated().any():
        raise ValueError(
            "Resolved splits produced duplicate validation rows. "
            "Calibration and residual estimation require disjoint validation windows."
        )
    return oof_df.sort_values("row_position").reset_index(drop=True)


def estimate_residual_scale_from_oof(
    df: pd.DataFrame,
    *,
    splits: SplitInput,
    fit_predict_fn: FitPredictFunction,
    target_col: str,
    residual_method: ResidualScaleMethod = "std",
    min_scale: float = 1e-6,
) -> ResidualScaleEstimate:
    """
    Estimate predictive uncertainty from leakage-free out-of-fold residuals.

    The returned residual scale is computed only from training-period data.
    """
    oof_df = collect_oof_regression_predictions(
        df=df,
        splits=splits,
        fit_predict_fn=fit_predict_fn,
        target_col=target_col,
    )
    residuals = (
        pd.to_numeric(oof_df["y_true"], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(oof_df["y_pred"], errors="coerce").to_numpy(dtype=float)
    )
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 2:
        raise ValueError(
            "At least two finite out-of-fold residuals are required to estimate uncertainty."
        )

    scale = estimate_residual_scale(
        residuals=residuals,
        method=residual_method,
        min_scale=min_scale,
    )
    return ResidualScaleEstimate(
        scale=scale,
        method=residual_method,
        n_residuals=int(residuals.size),
        residuals=residuals,
        oof_predictions=oof_df,
    )


def estimate_residual_scale(
    *,
    residuals: np.ndarray | pd.Series,
    method: ResidualScaleMethod = "std",
    min_scale: float = 1e-6,
) -> float:
    """Estimate a scalar residual scale from residuals measured in points."""
    residual_array = _coerce_numeric_array(residuals, name="residuals")
    residual_array = residual_array[np.isfinite(residual_array)]

    if residual_array.size < 2:
        raise ValueError("At least two finite residuals are required.")

    if method == "std":
        scale = float(np.std(residual_array, ddof=1))
    elif method == "mad":
        median = float(np.median(residual_array))
        mad = float(np.median(np.abs(residual_array - median)))
        scale = 1.4826 * mad
    else:
        raise ValueError(f"Unsupported residual scale method: {method}")

    if not np.isfinite(scale):
        raise ValueError("Residual scale could not be estimated.")
    return max(scale, float(min_scale))


def fit_probability_calibrator(
    raw_probabilities: np.ndarray | pd.Series,
    outcomes: np.ndarray | pd.Series,
    *,
    method: CalibrationMethod,
    clip_eps: float = 1e-6,
    min_train_samples: int = 25,
) -> ProbabilityCalibrator:
    """Fit a probability calibrator on raw probabilities and binary labels."""
    calibrator = ProbabilityCalibrator(
        method=method,
        clip_eps=clip_eps,
        min_train_samples=min_train_samples,
    )
    return calibrator.fit(raw_probabilities=raw_probabilities, outcomes=outcomes)


def generate_nested_oof_probability_calibration_data(
    train_df: pd.DataFrame,
    *,
    fit_predict_fn: FitPredictFunction,
    target_col: str,
    line_col: str,
    prediction_type: PredictionType,
    split_builder: SplitInput,
    total_points_col: str = "TOTAL_POINTS",
    residual_method: ResidualScaleMethod = "std",
    strict_nested_residuals: bool = True,
    clip_eps: float = 1e-6,
    date_col: str = "GAME_DATE",
    push_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """
    Build calibration training data from nested out-of-fold probabilities.

    Each calibration row is produced by:
    1. fitting the regressor on an inner-train split,
    2. predicting an inner-validation split,
    3. converting predictions to raw probabilities using a residual scale
       estimated from inner-train data only.

    In strict mode, folds without a valid nested residual-scale estimate are
    skipped rather than falling back to a potentially leaky approximation.
    """
    inner_splits = _resolve_splits(train_df, split_builder)
    rows: list[pd.DataFrame] = []

    fallback_scale: float | None = None
    if not strict_nested_residuals:
        fallback_scale = estimate_residual_scale_from_oof(
            train_df,
            splits=inner_splits,
            fit_predict_fn=fit_predict_fn,
            target_col=target_col,
            residual_method=residual_method,
        ).scale

    for fold_number, (inner_train_idx, inner_valid_idx) in enumerate(inner_splits, start=1):
        inner_train_df = train_df.iloc[inner_train_idx].copy()
        inner_valid_df = train_df.iloc[inner_valid_idx].copy()

        try:
            nested_splits = _resolve_splits(inner_train_df, split_builder)
            nested_scale = estimate_residual_scale_from_oof(
                inner_train_df,
                splits=nested_splits,
                fit_predict_fn=fit_predict_fn,
                target_col=target_col,
                residual_method=residual_method,
            ).scale
        except ValueError:
            if strict_nested_residuals:
                continue
            if fallback_scale is None:
                continue
            nested_scale = fallback_scale

        y_pred_reg = np.asarray(fit_predict_fn(inner_train_df, inner_valid_df), dtype=float)
        if y_pred_reg.shape[0] != len(inner_valid_df):
            raise ValueError(
                "fit_predict_fn returned the wrong number of predictions for a nested fold."
            )

        raw_prob = convert_regression_predictions_to_over_probabilities(
            y_pred_reg=y_pred_reg,
            line_values=inner_valid_df[line_col],
            residual_scale=nested_scale,
            prediction_type=prediction_type,
            clip_eps=clip_eps,
        )
        pred_margin = compute_predicted_margin(
            y_pred_reg=y_pred_reg,
            line_values=inner_valid_df[line_col],
            prediction_type=prediction_type,
        )
        actual_margin = resolve_actual_margin(
            inner_valid_df,
            target_col=target_col,
            line_col=line_col,
            prediction_type=prediction_type,
            total_points_col=total_points_col,
        )
        actual_over, non_push_mask = make_over_labels_from_margin(
            actual_margin,
            push_tolerance=push_tolerance,
        )

        if not np.any(non_push_mask):
            continue

        fold_df = pd.DataFrame(
            {
                "fold": fold_number,
                "row_index": inner_valid_df.index.to_numpy()[non_push_mask],
                "raw_prob_over": raw_prob[non_push_mask],
                "actual_over": actual_over[non_push_mask],
                "pred_margin": pred_margin[non_push_mask],
                "actual_margin": actual_margin[non_push_mask],
                "residual_scale_used": nested_scale,
            }
        )
        if date_col in inner_valid_df.columns:
            fold_df[date_col] = pd.to_datetime(
                inner_valid_df.loc[non_push_mask, date_col],
                errors="coerce",
            ).to_numpy()
        rows.append(fold_df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "fold",
                "row_index",
                "raw_prob_over",
                "actual_over",
                "pred_margin",
                "actual_margin",
                "residual_scale_used",
                date_col,
            ]
        )

    calibration_df = pd.concat(rows, ignore_index=True)
    return calibration_df.sort_values(date_col, na_position="last").reset_index(drop=True)


def evaluate_probability_calibration_fold(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    fit_predict_fn: FitPredictFunction,
    target_col: str,
    line_col: str,
    prediction_type: PredictionType,
    split_builder: SplitInput,
    total_points_col: str = "TOTAL_POINTS",
    date_col: str = "GAME_DATE",
    residual_method: ResidualScaleMethod = "std",
    calibration_methods: tuple[CalibrationMethod, ...] = ("isotonic", "sigmoid"),
    regression_threshold: float = 2.0,
    probability_edge_threshold: float = 0.02,
    over_odds_col: str | None = None,
    under_odds_col: str | None = None,
    odds_format: OddsFormat = "american",
    clip_eps: float = 1e-6,
    min_calibration_samples: int = 25,
    strict_nested_residuals: bool = True,
    push_tolerance: float = 1e-12,
) -> FoldProbabilityEvaluationResult:
    """Evaluate the raw and calibrated probability workflow for one outer fold."""
    train_splits = _resolve_splits(train_df, split_builder)
    residual_scale_estimate = estimate_residual_scale_from_oof(
        train_df,
        splits=train_splits,
        fit_predict_fn=fit_predict_fn,
        target_col=target_col,
        residual_method=residual_method,
    )

    calibration_oof_predictions = generate_nested_oof_probability_calibration_data(
        train_df,
        fit_predict_fn=fit_predict_fn,
        target_col=target_col,
        line_col=line_col,
        prediction_type=prediction_type,
        split_builder=split_builder,
        total_points_col=total_points_col,
        residual_method=residual_method,
        strict_nested_residuals=strict_nested_residuals,
        clip_eps=clip_eps,
        date_col=date_col,
        push_tolerance=push_tolerance,
    )

    y_pred_reg = np.asarray(fit_predict_fn(train_df, valid_df), dtype=float)
    return evaluate_probability_calibration_predictions(
        valid_df=valid_df,
        y_pred_reg=y_pred_reg,
        residual_scale_estimate=residual_scale_estimate,
        calibration_oof_predictions=calibration_oof_predictions,
        target_col=target_col,
        line_col=line_col,
        prediction_type=prediction_type,
        total_points_col=total_points_col,
        date_col=date_col,
        calibration_methods=calibration_methods,
        regression_threshold=regression_threshold,
        probability_edge_threshold=probability_edge_threshold,
        over_odds_col=over_odds_col,
        under_odds_col=under_odds_col,
        odds_format=odds_format,
        clip_eps=clip_eps,
        min_calibration_samples=min_calibration_samples,
        push_tolerance=push_tolerance,
    )


def evaluate_probability_calibration_predictions(
    *,
    valid_df: pd.DataFrame,
    y_pred_reg: np.ndarray | pd.Series,
    residual_scale_estimate: ResidualScaleEstimate,
    calibration_oof_predictions: pd.DataFrame,
    target_col: str,
    line_col: str,
    prediction_type: PredictionType,
    total_points_col: str = "TOTAL_POINTS",
    date_col: str = "GAME_DATE",
    calibration_methods: tuple[CalibrationMethod, ...] = ("isotonic", "sigmoid"),
    regression_threshold: float = 2.0,
    probability_edge_threshold: float = 0.02,
    over_odds_col: str | None = None,
    under_odds_col: str | None = None,
    odds_format: OddsFormat = "american",
    clip_eps: float = 1e-6,
    min_calibration_samples: int = 25,
    push_tolerance: float = 1e-12,
) -> FoldProbabilityEvaluationResult:
    """Score one validation fold from precomputed regression predictions."""
    y_pred_reg = np.asarray(y_pred_reg, dtype=float)
    if y_pred_reg.shape[0] != len(valid_df):
        raise ValueError(
            "y_pred_reg has the wrong length for the validation dataframe. "
            f"Expected {len(valid_df)}, got {y_pred_reg.shape[0]}."
        )

    y_true_reg = pd.to_numeric(valid_df[target_col], errors="coerce").to_numpy(dtype=float)
    line_values = pd.to_numeric(valid_df[line_col], errors="coerce").to_numpy(dtype=float)
    pred_margin = compute_predicted_margin(
        y_pred_reg=y_pred_reg,
        line_values=line_values,
        prediction_type=prediction_type,
    )
    actual_margin = resolve_actual_margin(
        valid_df,
        target_col=target_col,
        line_col=line_col,
        prediction_type=prediction_type,
        total_points_col=total_points_col,
    )
    actual_over, non_push_mask = make_over_labels_from_margin(
        actual_margin,
        push_tolerance=push_tolerance,
    )
    raw_prob_over = convert_regression_predictions_to_over_probabilities(
        y_pred_reg=y_pred_reg,
        line_values=line_values,
        residual_scale=residual_scale_estimate.scale,
        prediction_type=prediction_type,
        clip_eps=clip_eps,
    )

    predictions = pd.DataFrame(
        {
            "y_true_reg": y_true_reg,
            "y_pred_reg": y_pred_reg,
            "line_value": line_values,
            "pred_margin": pred_margin,
            "actual_margin": actual_margin,
            "raw_prob_over": raw_prob_over,
            "actual_over": actual_over,
            "is_non_push": non_push_mask,
        },
        index=valid_df.index,
    )
    if date_col in valid_df.columns:
        predictions[date_col] = pd.to_datetime(valid_df[date_col], errors="coerce")
    if total_points_col in valid_df.columns:
        predictions["actual_total_points"] = pd.to_numeric(
            valid_df[total_points_col], errors="coerce"
        )
    if over_odds_col is not None and over_odds_col in valid_df.columns:
        predictions["over_odds"] = pd.to_numeric(valid_df[over_odds_col], errors="coerce")
    if under_odds_col is not None and under_odds_col in valid_df.columns:
        predictions["under_odds"] = pd.to_numeric(valid_df[under_odds_col], errors="coerce")

    calibrators: dict[str, ProbabilityCalibrator] = {}
    for method in calibration_methods:
        calibrator = fit_probability_calibrator(
            calibration_oof_predictions.get("raw_prob_over", pd.Series(dtype=float)),
            calibration_oof_predictions.get("actual_over", pd.Series(dtype=float)),
            method=method,
            clip_eps=clip_eps,
            min_train_samples=min_calibration_samples,
        )
        predictions[f"{method}_prob_over"] = calibrator.transform(raw_prob_over)
        calibrators[method] = calibrator

    strategy_rows = [
        summarize_strategy_metrics(
            predictions=predictions,
            strategy_name="regression_threshold",
            regression_threshold=regression_threshold,
            probability_col=None,
            over_odds_col="over_odds" if "over_odds" in predictions.columns else None,
            under_odds_col="under_odds" if "under_odds" in predictions.columns else None,
            odds_format=odds_format,
        ),
        summarize_strategy_metrics(
            predictions=predictions,
            strategy_name="probability_raw",
            regression_threshold=regression_threshold,
            probability_col="raw_prob_over",
            probability_edge_threshold=probability_edge_threshold,
            over_odds_col="over_odds" if "over_odds" in predictions.columns else None,
            under_odds_col="under_odds" if "under_odds" in predictions.columns else None,
            odds_format=odds_format,
        ),
    ]

    for method in calibration_methods:
        strategy_rows.append(
            summarize_strategy_metrics(
                predictions=predictions,
                strategy_name=f"probability_{method}",
                regression_threshold=regression_threshold,
                probability_col=f"{method}_prob_over",
                probability_edge_threshold=probability_edge_threshold,
                over_odds_col="over_odds" if "over_odds" in predictions.columns else None,
                under_odds_col="under_odds" if "under_odds" in predictions.columns else None,
                odds_format=odds_format,
                calibrator=calibrators.get(method),
            )
        )

    strategy_summary = pd.DataFrame(strategy_rows)
    strategy_summary["residual_scale"] = residual_scale_estimate.scale
    strategy_summary["residual_method"] = residual_scale_estimate.method
    strategy_summary["n_residuals"] = residual_scale_estimate.n_residuals
    strategy_summary["n_calibration_samples"] = len(calibration_oof_predictions)

    return FoldProbabilityEvaluationResult(
        predictions=predictions.reset_index().rename(columns={"index": "row_index"}),
        strategy_summary=strategy_summary,
        calibration_oof_predictions=calibration_oof_predictions,
        residual_scale_estimate=residual_scale_estimate,
        calibrators=calibrators,
    )


def run_probability_calibration_backtest(
    df: pd.DataFrame,
    *,
    outer_splits: SplitInput,
    fit_predict_fn: FitPredictFunction,
    target_col: str,
    line_col: str,
    prediction_type: PredictionType,
    inner_split_builder: SplitInput,
    total_points_col: str = "TOTAL_POINTS",
    date_col: str = "GAME_DATE",
    residual_method: ResidualScaleMethod = "std",
    calibration_methods: tuple[CalibrationMethod, ...] = ("isotonic", "sigmoid"),
    regression_threshold: float = 2.0,
    probability_edge_threshold: float = 0.02,
    over_odds_col: str | None = None,
    under_odds_col: str | None = None,
    odds_format: OddsFormat = "american",
    clip_eps: float = 1e-6,
    min_calibration_samples: int = 25,
    strict_nested_residuals: bool = True,
    push_tolerance: float = 1e-12,
) -> ProbabilityBacktestResult:
    """Run the full regression-to-probability workflow across outer time-aware folds."""
    resolved_outer_splits = _resolve_splits(df, outer_splits)
    fold_prediction_rows: list[pd.DataFrame] = []
    fold_summary_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    residual_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(resolved_outer_splits, start=1):
        fold_result = evaluate_probability_calibration_fold(
            train_df=df.iloc[train_idx].copy(),
            valid_df=df.iloc[valid_idx].copy(),
            fit_predict_fn=fit_predict_fn,
            target_col=target_col,
            line_col=line_col,
            prediction_type=prediction_type,
            split_builder=inner_split_builder,
            total_points_col=total_points_col,
            date_col=date_col,
            residual_method=residual_method,
            calibration_methods=calibration_methods,
            regression_threshold=regression_threshold,
            probability_edge_threshold=probability_edge_threshold,
            over_odds_col=over_odds_col,
            under_odds_col=under_odds_col,
            odds_format=odds_format,
            clip_eps=clip_eps,
            min_calibration_samples=min_calibration_samples,
            strict_nested_residuals=strict_nested_residuals,
            push_tolerance=push_tolerance,
        )

        fold_predictions = fold_result.predictions.copy()
        fold_predictions["outer_fold"] = fold_number
        fold_prediction_rows.append(fold_predictions)

        fold_summary = fold_result.strategy_summary.copy()
        fold_summary["outer_fold"] = fold_number
        fold_summary_rows.append(fold_summary)

        calibration_predictions = fold_result.calibration_oof_predictions.copy()
        calibration_predictions["outer_fold"] = fold_number
        calibration_rows.append(calibration_predictions)

        residual_rows.append(
            {
                "outer_fold": fold_number,
                "residual_scale": fold_result.residual_scale_estimate.scale,
                "residual_method": fold_result.residual_scale_estimate.method,
                "n_residuals": fold_result.residual_scale_estimate.n_residuals,
            }
        )

    all_predictions = pd.concat(fold_prediction_rows, ignore_index=True)
    fold_strategy_summary = pd.concat(fold_summary_rows, ignore_index=True)
    calibration_oof_predictions = pd.concat(calibration_rows, ignore_index=True)
    residual_scale_by_fold = pd.DataFrame(residual_rows)

    overall_rows = []
    for strategy_name in fold_strategy_summary["strategy"].drop_duplicates():
        probability_col = None
        if strategy_name == "probability_raw":
            probability_col = "raw_prob_over"
        elif strategy_name.startswith("probability_") and strategy_name != "probability_raw":
            probability_col = f"{strategy_name.removeprefix('probability_')}_prob_over"

        overall_rows.append(
            summarize_strategy_metrics(
                predictions=all_predictions,
                strategy_name=strategy_name,
                regression_threshold=regression_threshold,
                probability_col=probability_col,
                probability_edge_threshold=probability_edge_threshold,
                over_odds_col="over_odds" if "over_odds" in all_predictions.columns else None,
                under_odds_col="under_odds" if "under_odds" in all_predictions.columns else None,
                odds_format=odds_format,
            )
        )

    overall_strategy_summary = pd.DataFrame(overall_rows)
    overall_strategy_summary["outer_fold"] = "ALL"
    overall_strategy_summary["residual_method"] = residual_method

    return ProbabilityBacktestResult(
        predictions=all_predictions,
        fold_strategy_summary=fold_strategy_summary,
        overall_strategy_summary=overall_strategy_summary,
        calibration_oof_predictions=calibration_oof_predictions,
        residual_scale_by_fold=residual_scale_by_fold,
    )


def summarize_strategy_metrics(
    *,
    predictions: pd.DataFrame,
    strategy_name: str,
    regression_threshold: float,
    probability_col: str | None,
    probability_edge_threshold: float = 0.02,
    over_odds_col: str | None = None,
    under_odds_col: str | None = None,
    odds_format: OddsFormat = "american",
    calibrator: ProbabilityCalibrator | None = None,
) -> dict[str, Any]:
    """Summarize regression, calibration, and betting metrics for one strategy."""
    y_true_reg = pd.to_numeric(predictions["y_true_reg"], errors="coerce").to_numpy(dtype=float)
    y_pred_reg = pd.to_numeric(predictions["y_pred_reg"], errors="coerce").to_numpy(dtype=float)

    regression_valid = np.isfinite(y_true_reg) & np.isfinite(y_pred_reg)
    mae = (
        float(mean_absolute_error(y_true_reg[regression_valid], y_pred_reg[regression_valid]))
        if np.any(regression_valid)
        else np.nan
    )
    rmse = (
        float(np.sqrt(mean_squared_error(y_true_reg[regression_valid], y_pred_reg[regression_valid])))
        if np.any(regression_valid)
        else np.nan
    )

    brier = np.nan
    ll = np.nan
    ece = np.nan
    n_probability = 0
    avg_probability = np.nan

    if probability_col is not None:
        probability_values = pd.to_numeric(
            predictions[probability_col], errors="coerce"
        ).to_numpy(dtype=float)
        actual_over = pd.to_numeric(predictions["actual_over"], errors="coerce").to_numpy(dtype=float)
        non_push = predictions["is_non_push"].to_numpy(dtype=bool)
        prob_valid = np.isfinite(probability_values) & np.isfinite(actual_over) & non_push

        if np.any(prob_valid):
            y_prob = np.clip(probability_values[prob_valid], 1e-6, 1.0 - 1e-6)
            y_true = actual_over[prob_valid].astype(int)
            brier = float(brier_score_loss(y_true, y_prob))
            ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
            ece = float(expected_calibration_error(y_true, y_prob))
            n_probability = int(prob_valid.sum())
            avg_probability = float(np.mean(y_prob))

    if probability_col is None:
        decisions = build_regression_threshold_betting_decisions(
            pred_margin=pd.to_numeric(predictions["pred_margin"], errors="coerce").to_numpy(dtype=float),
            threshold=regression_threshold,
        )
    else:
        over_implied = None
        under_implied = None
        if over_odds_col is not None and over_odds_col in predictions.columns:
            over_implied = odds_to_implied_probability(
                predictions[over_odds_col],
                odds_format=odds_format,
            )
        if under_odds_col is not None and under_odds_col in predictions.columns:
            under_implied = odds_to_implied_probability(
                predictions[under_odds_col],
                odds_format=odds_format,
            )
        decisions = build_probability_betting_decisions(
            prob_over=pd.to_numeric(predictions[probability_col], errors="coerce").to_numpy(dtype=float),
            edge_threshold=probability_edge_threshold,
            implied_prob_over=over_implied,
            implied_prob_under=under_implied,
        )

    betting_summary = summarize_betting_outcomes(
        predictions=predictions,
        decisions=decisions,
        over_odds_col=over_odds_col,
        under_odds_col=under_odds_col,
        odds_format=odds_format,
    )

    return {
        "strategy": strategy_name,
        "mae": mae,
        "rmse": rmse,
        "brier_score": brier,
        "log_loss": ll,
        "ece_10": ece,
        "n_probability_rows": n_probability,
        "avg_prob_over": avg_probability,
        "n_bets": betting_summary["n_bets"],
        "n_resolved_bets": betting_summary["n_resolved_bets"],
        "n_pushes": betting_summary["n_pushes"],
        "hit_rate": betting_summary["hit_rate"],
        "ou_betting_accuracy": betting_summary["hit_rate"],
        "avg_edge": betting_summary["avg_edge"],
        "roi": betting_summary["roi"],
        "n_bets_with_odds": betting_summary["n_bets_with_odds"],
        "calibrator_fitted": None if calibrator is None else calibrator.is_fitted,
        "calibrator_fallback_reason": None
        if calibrator is None
        else calibrator.fallback_reason,
    }


def build_regression_threshold_betting_decisions(
    *,
    pred_margin: np.ndarray | pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Create Over/Under betting decisions from a predicted regression margin."""
    margin = _coerce_numeric_array(pred_margin, name="pred_margin")
    decision = np.full(margin.shape[0], "", dtype=object)
    edge = np.abs(margin)

    over_mask = np.isfinite(margin) & (margin > threshold)
    under_mask = np.isfinite(margin) & (margin < -threshold)

    decision[over_mask] = "OVER"
    decision[under_mask] = "UNDER"

    return pd.DataFrame(
        {
            "decision": decision,
            "model_edge": edge,
            "edge_over": np.where(np.isfinite(margin), margin, np.nan),
            "edge_under": np.where(np.isfinite(margin), -margin, np.nan),
        }
    )


def build_probability_betting_decisions(
    *,
    prob_over: np.ndarray | pd.Series,
    edge_threshold: float = 0.02,
    implied_prob_over: np.ndarray | pd.Series | None = None,
    implied_prob_under: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Create Over/Under betting decisions from model probabilities."""
    prob = _coerce_numeric_array(prob_over, name="prob_over")
    prob_under = 1.0 - prob

    if implied_prob_over is None:
        edge_over = prob - 0.5
    else:
        edge_over = prob - _coerce_numeric_array(
            implied_prob_over,
            name="implied_prob_over",
        )

    if implied_prob_under is None:
        edge_under = prob_under - 0.5
    else:
        edge_under = prob_under - _coerce_numeric_array(
            implied_prob_under,
            name="implied_prob_under",
        )

    decision = np.full(prob.shape[0], "", dtype=object)
    over_mask = np.isfinite(edge_over) & (edge_over >= edge_threshold) & (edge_over >= edge_under)
    under_mask = np.isfinite(edge_under) & (edge_under >= edge_threshold) & (edge_under > edge_over)

    decision[over_mask] = "OVER"
    decision[under_mask] = "UNDER"

    selected_edge = np.where(
        decision == "OVER",
        edge_over,
        np.where(decision == "UNDER", edge_under, np.nan),
    )
    return pd.DataFrame(
        {
            "decision": decision,
            "model_edge": selected_edge,
            "edge_over": edge_over,
            "edge_under": edge_under,
        }
    )


def summarize_betting_outcomes(
    *,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    over_odds_col: str | None = None,
    under_odds_col: str | None = None,
    odds_format: OddsFormat = "american",
) -> dict[str, Any]:
    """Summarize betting outcomes for a precomputed decision set."""
    actual_margin = pd.to_numeric(predictions["actual_margin"], errors="coerce").to_numpy(dtype=float)
    actual_result = np.full(actual_margin.shape[0], "", dtype=object)
    actual_result[np.isfinite(actual_margin) & (actual_margin > 0)] = "OVER"
    actual_result[np.isfinite(actual_margin) & (actual_margin < 0)] = "UNDER"
    actual_result[np.isfinite(actual_margin) & (actual_margin == 0)] = "PUSH"

    decision = decisions["decision"].astype(str).to_numpy()
    placed_bets = decision != ""
    resolved_bets = placed_bets & (actual_result != "PUSH") & (actual_result != "")
    push_bets = placed_bets & (actual_result == "PUSH")

    hit_rate = (
        float(np.mean(decision[resolved_bets] == actual_result[resolved_bets]))
        if np.any(resolved_bets)
        else np.nan
    )
    avg_edge = (
        float(
            np.nanmean(
                pd.to_numeric(decisions.loc[placed_bets, "model_edge"], errors="coerce").to_numpy(dtype=float)
            )
        )
        if np.any(placed_bets)
        else np.nan
    )

    roi = np.nan
    n_bets_with_odds = 0
    if over_odds_col is not None and under_odds_col is not None:
        selected_odds = np.where(
            decision == "OVER",
            pd.to_numeric(predictions[over_odds_col], errors="coerce").to_numpy(dtype=float),
            np.where(
                decision == "UNDER",
                pd.to_numeric(predictions[under_odds_col], errors="coerce").to_numpy(dtype=float),
                np.nan,
            ),
        )
        settled_profit = payout_for_bet_result(
            odds=selected_odds,
            won=decision == actual_result,
            lost=resolved_bets & (decision != actual_result),
            pushed=push_bets,
            odds_format=odds_format,
        )
        odds_mask = placed_bets & np.isfinite(selected_odds)
        n_bets_with_odds = int(odds_mask.sum())
        if np.any(odds_mask):
            roi = float(np.nanmean(settled_profit[odds_mask]))

    return {
        "n_bets": int(placed_bets.sum()),
        "n_resolved_bets": int(resolved_bets.sum()),
        "n_pushes": int(push_bets.sum()),
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "roi": roi,
        "n_bets_with_odds": n_bets_with_odds,
    }


def resolve_actual_margin(
    df: pd.DataFrame,
    *,
    target_col: str,
    line_col: str,
    prediction_type: PredictionType,
    total_points_col: str = "TOTAL_POINTS",
) -> np.ndarray:
    """
    Resolve the realized margin versus the closing line.

    For ``prediction_type="line_error"``, the safest choice is to use the
    explicitly stored line-error target when available. Otherwise we derive it
    from ``TOTAL_POINTS - line``.
    """
    if prediction_type == "line_error":
        if target_col in df.columns:
            return pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
        if total_points_col not in df.columns:
            raise KeyError(
                f"{total_points_col} is required to derive actual margin when {target_col} is absent."
            )
        total_points = pd.to_numeric(df[total_points_col], errors="coerce").to_numpy(dtype=float)
        line_values = pd.to_numeric(df[line_col], errors="coerce").to_numpy(dtype=float)
        return total_points - line_values

    if total_points_col in df.columns:
        total_points = pd.to_numeric(df[total_points_col], errors="coerce").to_numpy(dtype=float)
    elif target_col in df.columns:
        total_points = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    else:
        raise KeyError(
            f"Could not resolve actual total points from {total_points_col!r} or {target_col!r}."
        )

    line_values = pd.to_numeric(df[line_col], errors="coerce").to_numpy(dtype=float)
    return total_points - line_values


def make_over_labels_from_margin(
    actual_margin: np.ndarray | pd.Series,
    *,
    push_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert realized margin into binary Over labels.

    Pushes are excluded from calibration and probability scoring because
    ``P(over)`` is not a binary event when the game lands exactly on the line.
    """
    margin = _coerce_numeric_array(actual_margin, name="actual_margin")
    non_push_mask = np.isfinite(margin) & (np.abs(margin) > push_tolerance)
    labels = np.full(margin.shape[0], np.nan, dtype=float)
    labels[non_push_mask] = (margin[non_push_mask] > 0).astype(float)
    return labels, non_push_mask


def odds_to_implied_probability(
    odds: np.ndarray | pd.Series,
    *,
    odds_format: OddsFormat = "american",
) -> np.ndarray:
    """Convert sportsbook odds to implied probabilities."""
    odds_array = _coerce_numeric_array(odds, name="odds")

    if odds_format == "american":
        implied = np.where(
            odds_array > 0,
            100.0 / (odds_array + 100.0),
            np.where(
                odds_array < 0,
                np.abs(odds_array) / (np.abs(odds_array) + 100.0),
                np.nan,
            ),
        )
    elif odds_format == "decimal":
        implied = np.where(odds_array > 1.0, 1.0 / odds_array, np.nan)
    else:
        raise ValueError(f"Unsupported odds format: {odds_format}")

    return np.asarray(implied, dtype=float)


def payout_for_bet_result(
    *,
    odds: np.ndarray | pd.Series,
    won: np.ndarray,
    lost: np.ndarray,
    pushed: np.ndarray,
    odds_format: OddsFormat = "american",
) -> np.ndarray:
    """Return profit per 1-unit stake for each bet outcome."""
    odds_array = _coerce_numeric_array(odds, name="odds")
    profit = np.full(odds_array.shape[0], np.nan, dtype=float)

    if odds_format == "american":
        win_profit = np.where(
            odds_array > 0,
            odds_array / 100.0,
            np.where(odds_array < 0, 100.0 / np.abs(odds_array), np.nan),
        )
    elif odds_format == "decimal":
        win_profit = np.where(odds_array > 1.0, odds_array - 1.0, np.nan)
    else:
        raise ValueError(f"Unsupported odds format: {odds_format}")

    profit[won & np.isfinite(win_profit)] = win_profit[won & np.isfinite(win_profit)]
    profit[lost & np.isfinite(odds_array)] = -1.0
    profit[pushed & np.isfinite(odds_array)] = 0.0
    return profit


def build_reliability_curve_dataframe(
    outcomes: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> pd.DataFrame:
    """Build reliability-diagram data for a probability column."""
    y_true = _coerce_numeric_array(outcomes, name="outcomes")
    y_prob = _coerce_numeric_array(probabilities, name="probabilities")
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid].astype(float)
    y_prob = y_prob[valid].astype(float)

    if y_true.size == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "bin_left",
                "bin_right",
                "n_obs",
                "mean_pred",
                "observed_freq",
                "abs_gap",
            ]
        )

    bin_ids, edges = _assign_probability_bins(y_prob, n_bins=n_bins, strategy=strategy)
    rows = []
    for bin_number in range(len(edges) - 1):
        mask = bin_ids == bin_number
        if not np.any(mask):
            continue
        mean_pred = float(np.mean(y_prob[mask]))
        observed_freq = float(np.mean(y_true[mask]))
        rows.append(
            {
                "bin": bin_number,
                "bin_left": float(edges[bin_number]),
                "bin_right": float(edges[bin_number + 1]),
                "n_obs": int(mask.sum()),
                "mean_pred": mean_pred,
                "observed_freq": observed_freq,
                "abs_gap": abs(mean_pred - observed_freq),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    outcomes: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> float:
    """Compute expected calibration error from a reliability curve."""
    curve_df = build_reliability_curve_dataframe(
        outcomes=outcomes,
        probabilities=probabilities,
        n_bins=n_bins,
        strategy=strategy,
    )
    if curve_df.empty:
        return np.nan
    weights = curve_df["n_obs"] / curve_df["n_obs"].sum()
    return float(np.sum(weights * curve_df["abs_gap"]))


def build_margin_over_rate_dataframe(
    pred_margin: np.ndarray | pd.Series,
    actual_over: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> pd.DataFrame:
    """Create binned diagnostics for predicted margin versus realized Over rate."""
    margin = _coerce_numeric_array(pred_margin, name="pred_margin")
    outcome = _coerce_numeric_array(actual_over, name="actual_over")
    valid = np.isfinite(margin) & np.isfinite(outcome)
    margin = margin[valid]
    outcome = outcome[valid]

    if margin.size == 0:
        return pd.DataFrame(
            columns=["bin", "n_obs", "mean_pred_margin", "observed_over_rate"]
        )

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(margin, quantiles))
    elif strategy == "uniform":
        edges = np.linspace(float(np.min(margin)), float(np.max(margin)), n_bins + 1)
    else:
        raise ValueError(f"Unsupported binning strategy: {strategy}")

    if edges.size < 2:
        edges = np.array([float(np.min(margin)), float(np.max(margin)) + 1e-9])

    bin_ids = np.digitize(margin, edges[1:-1], right=True)
    rows = []
    for bin_number in range(len(edges) - 1):
        mask = bin_ids == bin_number
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": bin_number,
                "bin_left": float(edges[bin_number]),
                "bin_right": float(edges[bin_number + 1]),
                "n_obs": int(mask.sum()),
                "mean_pred_margin": float(np.mean(margin[mask])),
                "observed_over_rate": float(np.mean(outcome[mask])),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability_diagram(
    predictions: pd.DataFrame,
    *,
    probability_columns: dict[str, str],
    actual_col: str = "actual_over",
    non_push_col: str = "is_non_push",
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
    ax=None,
):
    """Plot reliability curves for one or more probability columns."""
    axis = ax or plt.gca()
    actual = pd.to_numeric(predictions[actual_col], errors="coerce").to_numpy(dtype=float)
    mask = predictions[non_push_col].to_numpy(dtype=bool)

    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect")
    for column, label in probability_columns.items():
        prob = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        curve_df = build_reliability_curve_dataframe(
            outcomes=actual[mask],
            probabilities=prob[mask],
            n_bins=n_bins,
            strategy=strategy,
        )
        if curve_df.empty:
            continue
        axis.plot(curve_df["mean_pred"], curve_df["observed_freq"], marker="o", label=label)

    axis.set_xlabel("Predicted probability of Over")
    axis.set_ylabel("Observed Over frequency")
    axis.set_title("Reliability Diagram")
    axis.legend()
    return axis


def plot_probability_histograms(
    predictions: pd.DataFrame,
    *,
    probability_columns: dict[str, str],
    bins: int = 20,
    ax=None,
):
    """Plot histograms for one or more probability columns."""
    axis = ax or plt.gca()
    for column, label in probability_columns.items():
        prob = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        prob = prob[np.isfinite(prob)]
        if prob.size == 0:
            continue
        axis.hist(prob, bins=bins, alpha=0.4, label=label)
    axis.set_xlabel("Predicted probability of Over")
    axis.set_ylabel("Count")
    axis.set_title("Probability Distribution")
    axis.legend()
    return axis


def plot_margin_vs_over_rate(
    predictions: pd.DataFrame,
    *,
    pred_margin_col: str = "pred_margin",
    actual_col: str = "actual_over",
    non_push_col: str = "is_non_push",
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
    ax=None,
):
    """Plot binned predicted margin versus realized Over rate."""
    axis = ax or plt.gca()
    mask = predictions[non_push_col].to_numpy(dtype=bool)
    df_plot = build_margin_over_rate_dataframe(
        pred_margin=pd.to_numeric(predictions.loc[mask, pred_margin_col], errors="coerce"),
        actual_over=pd.to_numeric(predictions.loc[mask, actual_col], errors="coerce"),
        n_bins=n_bins,
        strategy=strategy,
    )
    if not df_plot.empty:
        axis.plot(
            df_plot["mean_pred_margin"],
            df_plot["observed_over_rate"],
            marker="o",
        )
    axis.axhline(0.5, linestyle="--", color="black", linewidth=1)
    axis.axvline(0.0, linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Predicted margin vs line")
    axis.set_ylabel("Observed Over frequency")
    axis.set_title("Predicted Margin vs Over Rate")
    return axis


def plot_raw_vs_calibrated_probabilities(
    predictions: pd.DataFrame,
    *,
    raw_col: str = "raw_prob_over",
    calibrated_columns: dict[str, str],
    sample_size: int | None = 2000,
    ax=None,
):
    """Scatter-plot raw versus calibrated probabilities."""
    axis = ax or plt.gca()
    raw = pd.to_numeric(predictions[raw_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw)
    if sample_size is not None and valid.sum() > sample_size:
        keep_idx = np.flatnonzero(valid)[:sample_size]
    else:
        keep_idx = np.flatnonzero(valid)

    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for column, label in calibrated_columns.items():
        calibrated = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        mask = keep_idx[np.isfinite(calibrated[keep_idx])]
        if mask.size == 0:
            continue
        axis.scatter(raw[mask], calibrated[mask], s=12, alpha=0.35, label=label)

    axis.set_xlabel("Raw probability")
    axis.set_ylabel("Calibrated probability")
    axis.set_title("Raw vs Calibrated Probabilities")
    axis.legend()
    return axis


def _resolve_splits(df: pd.DataFrame, splits: SplitInput) -> SplitList:
    if callable(splits):
        resolved = splits(df)
    else:
        resolved = splits

    if isinstance(resolved, tuple):
        resolved = resolved[0]

    if not isinstance(resolved, list):
        raise TypeError("splits must resolve to a list of (train_idx, valid_idx) tuples.")

    validated: SplitList = []
    for train_idx, valid_idx in resolved:
        train_arr = np.asarray(train_idx, dtype=int)
        valid_arr = np.asarray(valid_idx, dtype=int)
        if train_arr.size == 0 or valid_arr.size == 0:
            continue
        validated.append((train_arr, valid_arr))

    if not validated:
        raise ValueError("No valid time-aware splits were available.")

    max_position = len(df) - 1
    for train_arr, valid_arr in validated:
        if train_arr.min() < 0 or valid_arr.min() < 0:
            raise ValueError("Split indices must be non-negative.")
        if train_arr.max() > max_position or valid_arr.max() > max_position:
            raise ValueError("Split indices exceed dataframe bounds.")
        if np.intersect1d(train_arr, valid_arr).size > 0:
            raise ValueError("Train and validation indices overlap within a fold.")
        if train_arr.max() >= valid_arr.min():
            raise ValueError(
                "Time-aware split ordering is invalid: training rows must come strictly before validation rows."
            )

    return validated


def _coerce_numeric_array(values: np.ndarray | pd.Series, *, name: str) -> np.ndarray:
    array = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _assign_probability_bins(
    probabilities: np.ndarray,
    *,
    n_bins: int,
    strategy: BinStrategy,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(probabilities, quantiles))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unsupported binning strategy: {strategy}")

    if edges.size < 2:
        edges = np.array([0.0, 1.0])

    bin_ids = np.digitize(probabilities, edges[1:-1], right=True)
    return bin_ids, edges
