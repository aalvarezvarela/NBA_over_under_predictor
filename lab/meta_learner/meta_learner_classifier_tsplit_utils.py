from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)
from xgboost import XGBClassifier

try:
    from meta_learner_baselines import (
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
    )
    from meta_learner_feature_utils import (
        build_meta_learner_feature_frame,
        load_meta_learner_dataframe,
    )
except ModuleNotFoundError:
    from lab.meta_learner.meta_learner_baselines import (
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
    )
    from lab.meta_learner.meta_learner_feature_utils import (
        build_meta_learner_feature_frame,
        load_meta_learner_dataframe,
    )
from nba_ou.modeling.modeling import build_recency_sample_weights

DATE_COL = "GAME_DATE"
TARGET_ERROR_COL = "LINE_ERROR"
TARGET_CLASS_COL = "TARGET_IS_OVER"
DEFAULT_CONFIDENCE_THRESHOLDS = (0.52, 0.55, 0.58, 0.60, 0.65)


@dataclass(frozen=True)
class MetaClassifierDataset:
    dataframe: pd.DataFrame
    feature_frame: pd.DataFrame
    target: pd.Series
    feature_cols: list[str]
    dropped_feature_cols: list[str]
    baseline_cols: list[str]


class TemporalDecaySampleWeightClassifier(ClassifierMixin, BaseEstimator):
    """Classifier wrapper that computes recency weights inside each fold fit."""

    def __init__(
        self,
        estimator: Any,
        dates: pd.Series,
        *,
        lambda_: float = 0.01,
        date_col: str = DATE_COL,
    ) -> None:
        self.estimator = estimator
        self.dates = dates
        self.lambda_ = lambda_
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> "TemporalDecaySampleWeightClassifier":
        if not hasattr(X, "index"):
            raise TypeError("TemporalDecaySampleWeightClassifier requires X with an index.")

        raw_dates = pd.Series(self.dates)
        dates = pd.to_datetime(raw_dates.reindex(X.index), errors="coerce").dt.normalize()
        if dates.isna().any():
            if len(raw_dates) != len(X):
                raise ValueError("Classifier dates could not be aligned to X.")
            dates = pd.to_datetime(raw_dates.to_numpy(), errors="coerce")
            dates = pd.Series(dates, index=X.index).dt.normalize()
        if dates.isna().any():
            raise ValueError("Classifier dates could not be aligned to X.")

        temporal_weights = build_recency_sample_weights(dates, lambda_=self.lambda_)

        estimator = clone(self.estimator)
        existing_sample_weight = fit_params.pop("sample_weight", None)
        if existing_sample_weight is not None:
            existing_series = pd.Series(existing_sample_weight, index=X.index)
            temporal_weights = temporal_weights * pd.to_numeric(
                existing_series,
                errors="coerce",
            ).astype(float)

        estimator.fit(
            X,
            y,
            sample_weight=temporal_weights.to_numpy(dtype=float),
            **fit_params,
        )

        self.estimator_ = estimator
        self.sample_weight_ = temporal_weights
        self.classes_ = getattr(estimator, "classes_", None)
        self.n_features_in_ = getattr(estimator, "n_features_in_", None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimator_")


def error_to_binary(values: pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return (numeric > 0).astype(int)


def coerce_feature_column(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return series

    non_null = series.dropna()
    if non_null.empty:
        return pd.to_numeric(series, errors="coerce")

    normalized = non_null.astype(str).str.strip().str.lower()
    bool_mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "y": 1.0,
        "n": 0.0,
        "t": 1.0,
        "f": 0.0,
        "0": 0.0,
        "1": 1.0,
    }
    if normalized.isin(bool_mapping).all():
        mapped = series.astype("string").str.strip().str.lower().map(bool_mapping)
        return pd.to_numeric(mapped, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) == int(series.notna().sum()):
        return numeric

    return series


def coerce_feature_frame(df_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = df_features.copy()
    for column in output.columns:
        output[column] = coerce_feature_column(output[column])

    dropped_columns = [
        column
        for column in output.columns
        if not (is_numeric_dtype(output[column]) or is_bool_dtype(output[column]))
    ]
    if dropped_columns:
        output = output.drop(columns=dropped_columns)

    return output, dropped_columns


def build_meta_classifier_dataset(
    csv_path: str | Path,
    *,
    rolling_window_days: int = 5,
    include_baseline_features: bool = False,
) -> MetaClassifierDataset:
    raw_df = load_meta_learner_dataframe(csv_path)
    feature_result = build_meta_learner_feature_frame(
        raw_df,
        rolling_window_days=rolling_window_days,
    )
    return build_meta_classifier_dataset_from_feature_frame(
        feature_result.dataframe,
        include_baseline_features=include_baseline_features,
    )


def build_meta_classifier_dataset_from_feature_frame(
    feature_df: pd.DataFrame | Any,
    *,
    include_baseline_features: bool = False,
) -> MetaClassifierDataset:
    resolved_feature_df = getattr(feature_df, "dataframe", feature_df)
    if not isinstance(resolved_feature_df, pd.DataFrame):
        raise TypeError(
            "feature_df must be a pandas DataFrame or an object with a dataframe attribute."
        )

    df = resolved_feature_df.copy()

    df = df.loc[pd.to_numeric(df[TARGET_ERROR_COL], errors="coerce").ne(0)].copy()
    df[TARGET_CLASS_COL] = error_to_binary(df[TARGET_ERROR_COL])

    non_feature_cols = {
        "GAME_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "TEAM_ID_TEAM_HOME",
        "TEAM_ID_TEAM_AWAY",
        "TOTAL_POINTS",
        TARGET_ERROR_COL,
        TARGET_CLASS_COL,
    }
    if not include_baseline_features:
        non_feature_cols.update(
            [
                BASE_AVG_ALL_6_ERR_COL,
                BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
            ]
        )

    feature_cols = [column for column in df.columns if column not in non_feature_cols]
    X_raw = df[feature_cols].copy()
    X, dropped_feature_cols = coerce_feature_frame(X_raw)
    final_feature_cols = list(X.columns)

    df_model = df.loc[X.index].copy()
    y = df_model[TARGET_CLASS_COL].astype(int)

    return MetaClassifierDataset(
        dataframe=df_model,
        feature_frame=X,
        target=y,
        feature_cols=final_feature_cols,
        dropped_feature_cols=dropped_feature_cols,
        baseline_cols=[
            BASE_AVG_ALL_6_ERR_COL,
            BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
        ],
    )


def predict_positive_class_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return probabilities for both classes.")
        return np.asarray(probabilities[:, 1], dtype=float)
    raise TypeError("Estimator must implement predict_proba for this notebook.")


def brier_scorer(estimator, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba = predict_positive_class_probability(estimator, X)
    return -float(brier_score_loss(y_true_arr, y_proba))


def log_loss_scorer(estimator, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba = predict_positive_class_probability(estimator, X)
    return -float(log_loss(y_true_arr, y_proba, labels=[0, 1]))


def build_classifier_scoring() -> dict[str, Any]:
    return {
        "Accuracy": "accuracy",
        "Balanced_Accuracy": "balanced_accuracy",
        "Brier": brier_scorer,
        "LogLoss": log_loss_scorer,
    }


def print_classifier_cv_metrics(cv_results: dict[str, np.ndarray], scoring: dict[str, Any]) -> None:
    for metric_name in scoring:
        train_key = f"train_{metric_name}"
        test_key = f"test_{metric_name}"

        train_value = float(np.nanmean(cv_results[train_key]))
        test_value = float(np.nanmean(cv_results[test_key]))

        if metric_name in {"Brier", "LogLoss"}:
            train_value = -train_value
            test_value = -test_value
            print(f"Train {metric_name}: {train_value:.5f}")
            print(f"Validation {metric_name}: {test_value:.5f}")
        else:
            print(f"Train {metric_name}: {train_value:.2%}")
            print(f"Validation {metric_name}: {test_value:.2%}")
        print()


def summarize_classifier_predictions(
    *,
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    label: str,
) -> dict[str, float | str | int]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_pred_arr = (y_proba_arr >= 0.5).astype(int)
    confidence = np.maximum(y_proba_arr, 1.0 - y_proba_arr)

    return {
        "model": label,
        "n_games": int(len(y_true_arr)),
        "accuracy_pct": 100.0 * accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy_pct": 100.0 * balanced_accuracy_score(y_true_arr, y_pred_arr),
        "brier_score": float(brier_score_loss(y_true_arr, y_proba_arr)),
        "log_loss": float(log_loss(y_true_arr, y_proba_arr, labels=[0, 1])),
        "mean_predicted_over_pct": 100.0 * float(np.mean(y_proba_arr)),
        "mean_confidence_pct": 100.0 * float(np.mean(confidence)),
    }


def summarize_hard_class_predictions(
    *,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    label: str,
) -> dict[str, float | str | int]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    return {
        "model": label,
        "n_games": int(len(y_true_arr)),
        "accuracy_pct": 100.0 * accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy_pct": 100.0 * balanced_accuracy_score(y_true_arr, y_pred_arr),
    }


def evaluate_baseline_hard_class_over_splits(
    *,
    df: pd.DataFrame,
    target_col: str,
    baseline_error_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    y_all = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=int)
    rows: list[dict[str, float | str | int]] = []

    for baseline_col in baseline_error_cols:
        baseline_pred = error_to_binary(df[baseline_col])
        fold_accuracies: list[float] = []
        fold_balanced: list[float] = []

        for _, test_idx in splits:
            fold_accuracies.append(
                100.0 * accuracy_score(y_all[test_idx], baseline_pred[test_idx])
            )
            fold_balanced.append(
                100.0 * balanced_accuracy_score(y_all[test_idx], baseline_pred[test_idx])
            )

        rows.append(
            {
                "model": baseline_col,
                "validation_accuracy_pct": float(np.mean(fold_accuracies)),
                "validation_balanced_accuracy_pct": float(np.mean(fold_balanced)),
            }
        )

    return pd.DataFrame(rows)


def summarize_confidence_thresholds(
    *,
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    confidence = np.maximum(y_proba_arr, 1.0 - y_proba_arr)
    hard_pred = (y_proba_arr >= 0.5).astype(int)
    n_total = len(y_true_arr)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        take_mask = confidence >= threshold
        n_games = int(take_mask.sum())
        accuracy = (
            np.nan
            if n_games == 0
            else 100.0 * accuracy_score(y_true_arr[take_mask], hard_pred[take_mask])
        )
        rows.append(
            {
                "threshold_confidence_gte": threshold,
                "n_games": n_games,
                "pct_of_test": (n_games / n_total) if n_total else np.nan,
                "bet_accuracy_pct": accuracy,
                "mean_confidence_pct": (
                    np.nan if n_games == 0 else 100.0 * float(np.mean(confidence[take_mask]))
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_day_by_day_classifier_thresholds(
    predictions_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> pd.DataFrame:
    return summarize_confidence_thresholds(
        y_true=predictions_df["y_true"],
        y_proba=predictions_df["y_pred"],
        thresholds=thresholds,
    )


@dataclass
class ClassifierFoldMetrics:
    fold: int
    accuracy: float
    balanced_accuracy: float
    brier: float
    log_loss_value: float
    best_iteration: int
    n_train: int
    n_valid: int


def _coerce_sample_weight_dates(
    sample_weight_dates: pd.Series | np.ndarray | list[str] | None,
    index: pd.Index,
) -> pd.Series | None:
    if sample_weight_dates is None:
        return None

    if isinstance(sample_weight_dates, pd.Series):
        dates = sample_weight_dates.reindex(index)
    else:
        dates = pd.Series(sample_weight_dates, index=index)

    dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("sample_weight_dates contains missing or invalid datetime values.")
    return dates


def _resolve_sample_weight_lambda(
    trial: optuna.Trial,
    *,
    sample_weight_lambda: float | None,
    tune_sample_weight_lambda: bool,
    sample_weight_lambda_bounds: tuple[float, float],
) -> float | None:
    if not tune_sample_weight_lambda:
        if sample_weight_lambda is None:
            return None
        if sample_weight_lambda < 0:
            raise ValueError("sample_weight_lambda must be >= 0.")
        return float(sample_weight_lambda)

    low, high = sample_weight_lambda_bounds
    if low <= 0 or high <= 0 or low >= high:
        raise ValueError("sample_weight_lambda_bounds must be positive and ordered.")

    return float(trial.suggest_float("sample_weight_lambda", low, high, log=True))


def build_xgb_params_classifier(
    trial: optuna.Trial,
    *,
    random_state: int = 16,
) -> dict[str, Any]:
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 25.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "subsample": trial.suggest_float("subsample", 0.55, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.9),
        "learning_rate": trial.suggest_float("learning_rate", 0.0075, 0.05, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def _predict_best_classifier(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        try:
            return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]
        except TypeError:
            pass
    return model.predict_proba(X)[:, 1]


def evaluate_fold_classifier(
    model: XGBClassifier,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    fold: int,
    n_train: int,
) -> ClassifierFoldMetrics:
    y_true = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=int)
    y_proba = np.asarray(_predict_best_classifier(model, X_valid), dtype=float)
    y_pred = (y_proba >= 0.5).astype(int)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = model.get_params().get("n_estimators", 0) - 1

    return ClassifierFoldMetrics(
        fold=fold,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        brier=float(brier_score_loss(y_true, y_proba)),
        log_loss_value=float(log_loss(y_true, y_proba, labels=[0, 1])),
        best_iteration=int(best_iteration) + 1,
        n_train=int(n_train),
        n_valid=int(len(X_valid)),
    )


def objective_xgb_classifier_logloss(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight_dates: pd.Series | np.ndarray | list[str] | None = None,
    sample_weight_lambda: float | None = None,
    tune_sample_weight_lambda: bool = False,
    sample_weight_lambda_bounds: tuple[float, float] = (1e-4, 0.05),
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    params = build_xgb_params_classifier(trial, random_state=16)

    sample_weight_date_series = _coerce_sample_weight_dates(sample_weight_dates, X.index)
    resolved_sample_weight_lambda = _resolve_sample_weight_lambda(
        trial,
        sample_weight_lambda=sample_weight_lambda,
        tune_sample_weight_lambda=tune_sample_weight_lambda,
        sample_weight_lambda_bounds=sample_weight_lambda_bounds,
    )

    if sample_weight_date_series is not None and resolved_sample_weight_lambda is None:
        raise ValueError(
            "sample_weight_lambda must be provided when sample_weight_dates is used."
        )

    fold_metrics: list[ClassifierFoldMetrics] = []

    for fold_num, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        sample_weight_tr = None
        if sample_weight_date_series is not None:
            sample_weight_tr = build_recency_sample_weights(
                sample_weight_date_series.iloc[tr_idx],
                lambda_=float(resolved_sample_weight_lambda),
            ).to_numpy(dtype=float)

        model = XGBClassifier(**params)
        fit_kwargs: dict[str, Any] = {
            "eval_set": [(X_va, y_va)],
            "verbose": False,
        }
        if sample_weight_tr is not None:
            fit_kwargs["sample_weight"] = sample_weight_tr

        model.fit(X_tr, y_tr, **fit_kwargs)

        metrics = evaluate_fold_classifier(
            model=model,
            X_valid=X_va,
            y_valid=y_va,
            fold=fold_num,
            n_train=len(X_tr),
        )
        fold_metrics.append(metrics)

        mean_log_loss_so_far = float(np.mean([m.log_loss_value for m in fold_metrics]))
        trial.report(mean_log_loss_so_far, step=fold_num)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_accuracy = float(np.mean([m.accuracy for m in fold_metrics]))
    mean_balanced_accuracy = float(np.mean([m.balanced_accuracy for m in fold_metrics]))
    mean_brier = float(np.mean([m.brier for m in fold_metrics]))
    mean_log_loss_value = float(np.mean([m.log_loss_value for m in fold_metrics]))
    mean_best_iteration = int(round(np.mean([m.best_iteration for m in fold_metrics])))
    median_best_iteration = int(np.median([m.best_iteration for m in fold_metrics]))

    if resolved_sample_weight_lambda is not None:
        trial.set_user_attr("sample_weight_lambda", resolved_sample_weight_lambda)
    trial.set_user_attr("mean_accuracy", mean_accuracy)
    trial.set_user_attr("mean_balanced_accuracy", mean_balanced_accuracy)
    trial.set_user_attr("mean_brier", mean_brier)
    trial.set_user_attr("mean_log_loss", mean_log_loss_value)
    trial.set_user_attr("mean_best_iteration", mean_best_iteration)
    trial.set_user_attr("median_best_iteration", median_best_iteration)
    trial.set_user_attr(
        "fold_metrics",
        [
            {
                "fold": m.fold,
                "accuracy": m.accuracy,
                "balanced_accuracy": m.balanced_accuracy,
                "brier": m.brier,
                "log_loss": m.log_loss_value,
                "best_iteration": m.best_iteration,
                "n_train": m.n_train,
                "n_valid": m.n_valid,
            }
            for m in fold_metrics
        ],
    )

    return mean_log_loss_value


def tune_xgb_classifier_optuna(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight_dates: pd.Series | np.ndarray | list[str] | None = None,
    sample_weight_lambda: float | None = None,
    tune_sample_weight_lambda: bool = False,
    sample_weight_lambda_bounds: tuple[float, float] = (1e-4, 0.05),
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int = 80,
    timeout: int | None = None,
    study_name: str = "xgb_meta_classifier_logloss",
) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=16),
        pruner=MedianPruner(n_warmup_steps=5),
        study_name=study_name,
    )

    study.optimize(
        lambda trial: objective_xgb_classifier_logloss(
            trial,
            X=X,
            y=y,
            sample_weight_dates=sample_weight_dates,
            sample_weight_lambda=sample_weight_lambda,
            tune_sample_weight_lambda=tune_sample_weight_lambda,
            sample_weight_lambda_bounds=sample_weight_lambda_bounds,
            splits=splits,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=True,
    )

    return study


def summarize_classifier_optuna_trials(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        rows.append(
            {
                "trial": trial.number,
                "value_log_loss": trial.value,
                "mean_accuracy": trial.user_attrs.get("mean_accuracy"),
                "mean_balanced_accuracy": trial.user_attrs.get("mean_balanced_accuracy"),
                "mean_brier": trial.user_attrs.get("mean_brier"),
                "mean_log_loss": trial.user_attrs.get("mean_log_loss"),
                "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
                "median_best_iteration": trial.user_attrs.get("median_best_iteration"),
                **trial.params,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["value_log_loss", "mean_accuracy", "mean_brier"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def _get_completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed_trials:
        raise ValueError("No completed Optuna trials found.")
    return completed_trials


def _resolve_log_loss_cutoff(
    *,
    best_log_loss: float,
    log_loss_tolerance_abs: float | None,
    log_loss_tolerance_pct: float | None,
) -> float:
    if log_loss_tolerance_abs is not None and log_loss_tolerance_pct is not None:
        raise ValueError(
            "Provide only one of log_loss_tolerance_abs or log_loss_tolerance_pct."
        )
    if log_loss_tolerance_abs is not None:
        return float(best_log_loss + log_loss_tolerance_abs)
    if log_loss_tolerance_pct is not None:
        return float(best_log_loss * (1.0 + log_loss_tolerance_pct))
    return float(best_log_loss)


def select_best_trial_lexicographic_classifier(
    study: optuna.Study,
    *,
    log_loss_tolerance_abs: float | None = 0.01,
    log_loss_tolerance_pct: float | None = None,
) -> optuna.trial.FrozenTrial:
    completed_trials = _get_completed_trials(study)
    best_log_loss = min(float(trial.value) for trial in completed_trials)
    log_loss_cutoff = _resolve_log_loss_cutoff(
        best_log_loss=best_log_loss,
        log_loss_tolerance_abs=log_loss_tolerance_abs,
        log_loss_tolerance_pct=log_loss_tolerance_pct,
    )

    candidate_trials = [
        trial
        for trial in completed_trials
        if float(trial.user_attrs.get("mean_log_loss", trial.value)) <= log_loss_cutoff
    ]
    if not candidate_trials:
        raise ValueError("No candidate trials found within the log-loss tolerance.")

    return min(
        candidate_trials,
        key=lambda trial: (
            -float(trial.user_attrs.get("mean_accuracy", float("-inf"))),
            -float(trial.user_attrs.get("mean_balanced_accuracy", float("-inf"))),
            float(trial.user_attrs.get("mean_brier", float("inf"))),
            float(trial.user_attrs.get("mean_log_loss", trial.value)),
            trial.number,
        ),
    )


def summarize_classifier_lexicographic_candidates(
    study: optuna.Study,
    *,
    log_loss_tolerance_abs: float | None = 0.01,
    log_loss_tolerance_pct: float | None = None,
) -> pd.DataFrame:
    completed_trials = _get_completed_trials(study)
    best_log_loss = min(float(trial.value) for trial in completed_trials)
    log_loss_cutoff = _resolve_log_loss_cutoff(
        best_log_loss=best_log_loss,
        log_loss_tolerance_abs=log_loss_tolerance_abs,
        log_loss_tolerance_pct=log_loss_tolerance_pct,
    )

    rows: list[dict[str, Any]] = []
    for trial in completed_trials:
        mean_log_loss = float(trial.user_attrs.get("mean_log_loss", trial.value))
        if mean_log_loss > log_loss_cutoff:
            continue
        rows.append(
            {
                "trial": trial.number,
                "value_log_loss": float(trial.value),
                "mean_log_loss": mean_log_loss,
                "mean_accuracy": trial.user_attrs.get("mean_accuracy"),
                "mean_balanced_accuracy": trial.user_attrs.get("mean_balanced_accuracy"),
                "mean_brier": trial.user_attrs.get("mean_brier"),
                "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
                "median_best_iteration": trial.user_attrs.get("median_best_iteration"),
                "log_loss_cutoff": log_loss_cutoff,
                **trial.params,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_accuracy", "mean_balanced_accuracy", "mean_brier", "mean_log_loss", "trial"],
            ascending=[False, False, True, True, True],
        )
        .reset_index(drop=True)
    )


def get_trial_n_estimators_classifier(trial: optuna.trial.FrozenTrial) -> int:
    final_n_estimators = trial.user_attrs.get("median_best_iteration")
    if final_n_estimators is None:
        final_n_estimators = trial.user_attrs.get("mean_best_iteration")
    if final_n_estimators is None:
        final_n_estimators = trial.params.get("n_estimators", 75)
    return max(50, int(round(float(final_n_estimators))))


def fit_best_xgb_classifier(
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    sample_weight_dates: pd.Series | np.ndarray | list[str] | None = None,
    sample_weight_lambda: float | None = None,
    study: optuna.Study | None = None,
    trial: optuna.trial.FrozenTrial | None = None,
) -> XGBClassifier:
    if (study is None) == (trial is None):
        raise ValueError("Provide exactly one of study or trial.")

    selected_trial = trial if trial is not None else study.best_trial
    best_params = selected_trial.params.copy()
    final_n_estimators = get_trial_n_estimators_classifier(selected_trial)
    best_params.pop("sample_weight_lambda", None)

    final_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 16,
        "n_jobs": -1,
        "verbosity": 0,
        "n_estimators": final_n_estimators,
        **best_params,
    }

    model = XGBClassifier(**final_params)
    fit_kwargs: dict[str, Any] = {"verbose": False}

    if sample_weight_dates is not None:
        sample_weight_date_series = _coerce_sample_weight_dates(sample_weight_dates, X_dev.index)
        resolved_sample_weight_lambda = sample_weight_lambda
        if (
            resolved_sample_weight_lambda is None
            and "sample_weight_lambda" in selected_trial.params
        ):
            resolved_sample_weight_lambda = float(selected_trial.params["sample_weight_lambda"])
        if resolved_sample_weight_lambda is None:
            raise ValueError(
                "sample_weight_lambda must be provided or present in the selected trial."
            )
        sample_weight_series = build_recency_sample_weights(
            sample_weight_date_series,
            lambda_=float(resolved_sample_weight_lambda),
        )
        fit_kwargs["sample_weight"] = sample_weight_series.to_numpy(dtype=float)

    model.fit(X_dev, y_dev, **fit_kwargs)
    return model
