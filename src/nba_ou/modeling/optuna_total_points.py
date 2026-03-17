from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.scorers import over_under_betting_accuracy_total_points
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass
class FoldMetrics:
    fold: int
    mae: float
    rmse: float
    r2: float
    ou_accuracy: float
    best_iteration: int
    n_train: int
    n_valid: int


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _predict_best(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Predict using the best iteration when early stopping was used.
    """
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is not None:
        try:
            return model.predict(X, iteration_range=(0, best_iteration + 1))
        except TypeError:
            pass

    return model.predict(X)


def build_xgb_params_total_points(
    trial: optuna.Trial,
    *,
    random_state: int = 16,
    objective: str = "reg:squarederror",
) -> dict:
    """
    Conservative search space for noisy NBA totals.
    """
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": objective,
        "eval_metric": "mae",
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 5.0, 40.0, log=True
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "subsample": trial.suggest_float("subsample", 0.55, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.90),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 50.0, log=True),
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def evaluate_fold_total_points(
    model: XGBRegressor,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    line_col: str,
    *,
    fold: int,
    n_train: int,
) -> FoldMetrics:
    """
    Compute fold metrics after training.
    """
    y_true = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=float)
    y_pred = np.asarray(_predict_best(model, X_valid), dtype=float)
    betting_line = pd.to_numeric(X_valid[line_col], errors="coerce").to_numpy(
        dtype=float
    )

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    ou_acc = float(
        over_under_betting_accuracy_total_points(
            y_true=y_true,
            y_pred=y_pred,
            betting_line=betting_line,
        )
    )

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = model.get_params().get("n_estimators", 0) - 1

    return FoldMetrics(
        fold=fold,
        mae=mae,
        rmse=rmse,
        r2=r2,
        ou_accuracy=ou_acc,
        best_iteration=int(best_iteration) + 1,
        n_train=int(n_train),
        n_valid=int(len(X_valid)),
    )


def objective_total_points_mae(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    line_col: str,
    objective_name: str = "reg:squarederror",
) -> float:
    """
    Optuna objective: minimize mean validation MAE across time-aware folds.
    Secondary metrics are stored in user_attrs.
    """
    if line_col not in X.columns:
        raise KeyError(f"{line_col} not found in X")

    params = build_xgb_params_total_points(
        trial,
        random_state=16,
        objective=objective_name,
    )

    fold_metrics: list[FoldMetrics] = []

    for fold_num, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        metrics = evaluate_fold_total_points(
            model=model,
            X_valid=X_va,
            y_valid=y_va,
            line_col=line_col,
            fold=fold_num,
            n_train=len(X_tr),
        )
        fold_metrics.append(metrics)

        mean_mae_so_far = float(np.mean([m.mae for m in fold_metrics]))
        trial.report(mean_mae_so_far, step=fold_num)

        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_mae = float(np.mean([m.mae for m in fold_metrics]))
    mean_rmse = float(np.mean([m.rmse for m in fold_metrics]))
    mean_r2 = float(np.mean([m.r2 for m in fold_metrics]))
    mean_ou_acc = float(np.mean([m.ou_accuracy for m in fold_metrics]))
    mean_best_iteration = int(round(np.mean([m.best_iteration for m in fold_metrics])))

    trial.set_user_attr("mean_mae", mean_mae)
    trial.set_user_attr("mean_rmse", mean_rmse)
    trial.set_user_attr("mean_r2", mean_r2)
    trial.set_user_attr("mean_ou_acc", mean_ou_acc)
    trial.set_user_attr("mean_best_iteration", mean_best_iteration)
    trial.set_user_attr(
        "fold_metrics",
        [
            {
                "fold": m.fold,
                "mae": m.mae,
                "rmse": m.rmse,
                "r2": m.r2,
                "ou_accuracy": m.ou_accuracy,
                "best_iteration": m.best_iteration,
                "n_train": m.n_train,
                "n_valid": m.n_valid,
            }
            for m in fold_metrics
        ],
    )

    return mean_mae


def tune_xgb_total_points_optuna(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    line_col: str,
    n_trials: int = 80,
    timeout: int | None = None,
    objective_name: str = "reg:squarederror",
    study_name: str = "xgb_total_points_mae",
) -> optuna.Study:
    """
    Run Optuna tuning for total-points regression.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=16),
        pruner=MedianPruner(n_warmup_steps=5),
        study_name=study_name,
    )

    study.optimize(
        lambda trial: objective_total_points_mae(
            trial,
            X=X,
            y=y,
            splits=splits,
            line_col=line_col,
            objective_name=objective_name,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=True,
    )

    return study


def summarize_optuna_trials(study: optuna.Study) -> pd.DataFrame:
    """
    Build a tidy dataframe with primary and secondary metrics.
    """
    rows = []

    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue

        rows.append(
            {
                "trial": trial.number,
                "value_mae": trial.value,
                "mean_rmse": trial.user_attrs.get("mean_rmse"),
                "mean_r2": trial.user_attrs.get("mean_r2"),
                "mean_ou_acc": trial.user_attrs.get("mean_ou_acc"),
                "mean_best_iteration": trial.user_attrs.get("mean_best_iteration"),
                **trial.params,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["value_mae", "mean_rmse"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )


def fit_best_xgb_total_points(
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    study: optuna.Study,
    objective_name: str = "reg:squarederror",
) -> XGBRegressor:
    """
    Refit best params on all development data.

    Uses the average best_iteration from CV folds as the final n_estimators.
    This avoids carving out another validation chunk after tuning.
    """
    best_params = study.best_trial.params.copy()
    final_n_estimators = int(
        study.best_trial.user_attrs.get("mean_best_iteration", 300)
    )

    final_params = {
        "booster": "gbtree",
        "tree_method": "hist",
        "objective": objective_name,
        "eval_metric": "mae",
        "random_state": 16,
        "n_jobs": -1,
        "verbosity": 0,
        "n_estimators": max(50, final_n_estimators),
        **best_params,
    }

    model = XGBRegressor(**final_params)
    model.fit(X_dev, y_dev, verbose=False)
    return model
