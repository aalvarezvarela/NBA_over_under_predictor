from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.modeling.probability_calibration import (
    ResidualScaleEstimate,
    build_model_bundle_fit_predict_function,
    convert_regression_predictions_to_over_probabilities,
    evaluate_probability_calibration_predictions,
    estimate_residual_scale_from_oof,
    generate_nested_oof_probability_calibration_data,
    get_model_bundle_feature_names,
    prepare_model_bundle_feature_frame,
    predict_with_loaded_model_bundle,
    resolve_model_bundle_training_params,
)


def _mean_regression_fit_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    mean_value = float(pd.to_numeric(train_df["LINE_ERROR"], errors="coerce").mean())
    return np.full(len(valid_df), mean_value, dtype=float)


def _nested_split_builder(
    df: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_rows = len(df)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if n_rows >= 8:
        splits.append((np.arange(0, 6), np.arange(6, 8)))
    if n_rows >= 10:
        splits.append((np.arange(0, 8), np.arange(8, 10)))
    return splits


def test_convert_regression_predictions_to_probabilities_for_total_points() -> None:
    probabilities = convert_regression_predictions_to_over_probabilities(
        y_pred_reg=np.array([221.0, 219.0, 220.0]),
        line_values=np.array([220.0, 220.0, 220.0]),
        residual_scale=2.0,
        prediction_type="total_points",
    )

    assert probabilities.shape == (3,)
    assert probabilities[0] > 0.5
    assert probabilities[1] < 0.5
    assert np.isclose(probabilities[2], 0.5)


def test_estimate_residual_scale_from_oof_matches_expected_std() -> None:
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range("2024-01-01", periods=10, freq="D"),
            "LINE_ERROR": np.arange(10, dtype=float),
        }
    )
    splits = [
        (np.arange(0, 4), np.arange(4, 6)),
        (np.arange(0, 6), np.arange(6, 8)),
        (np.arange(0, 8), np.arange(8, 10)),
    ]

    estimate = estimate_residual_scale_from_oof(
        df,
        splits=splits,
        fit_predict_fn=_mean_regression_fit_predict,
        target_col="LINE_ERROR",
        residual_method="std",
    )

    expected_residuals = np.array([2.5, 3.5, 3.5, 4.5, 4.5, 5.5])
    expected_scale = float(np.std(expected_residuals, ddof=1))

    assert estimate.n_residuals == 6
    assert np.isclose(estimate.scale, expected_scale)


def test_nested_calibration_data_skips_folds_without_nested_history_in_strict_mode() -> None:
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range("2024-01-01", periods=10, freq="D"),
            "LINE_ERROR": np.array([-4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0]),
            "TOTAL_POINTS": np.array([216, 217, 218, 219, 219.5, 220.5, 221, 222, 223, 224]),
            "TOTAL_LINE_bet365": np.full(10, 220.0),
        }
    )

    calibration_df = generate_nested_oof_probability_calibration_data(
        df,
        fit_predict_fn=_mean_regression_fit_predict,
        target_col="LINE_ERROR",
        line_col="TOTAL_LINE_bet365",
        prediction_type="line_error",
        split_builder=_nested_split_builder,
        strict_nested_residuals=True,
    )

    # The first inner fold trains on only 6 rows, which intentionally produces
    # no nested split and must be skipped in strict mode.
    assert list(calibration_df["row_index"]) == [8, 9]
    assert np.all((calibration_df["raw_prob_over"] > 0.0) & (calibration_df["raw_prob_over"] < 1.0))


def test_resolve_model_bundle_training_params_uses_saved_metadata_only() -> None:
    metadata = {
        "schema": {"feature_names": ["feature_b", "feature_a"]},
        "training_metrics": {
            "best_params": {
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_weight": 1.0,
                "gamma": 0.0,
                "sample_weight_lambda": 0.01,
            },
            "median_best_iteration": 17,
            "mean_best_iteration": 21,
        },
    }

    feature_names = get_model_bundle_feature_names(metadata)
    params, sample_weight_lambda = resolve_model_bundle_training_params(metadata)

    assert feature_names == ["feature_b", "feature_a"]
    assert params["n_estimators"] == 50
    assert "sample_weight_lambda" not in params
    assert sample_weight_lambda == 0.01


def test_build_model_bundle_fit_predict_function_and_loaded_model_prediction() -> None:
    metadata = {
        "schema": {"feature_names": ["feature_b", "feature_a"]},
        "training_metrics": {
            "best_params": {
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "min_child_weight": 1.0,
                "gamma": 0.0,
                "sample_weight_lambda": 0.01,
            },
            "median_best_iteration": 5,
        },
    }
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range("2024-01-01", periods=8, freq="D"),
            "feature_a": np.linspace(0.0, 1.4, 8),
            "feature_b": np.linspace(1.0, 2.4, 8),
            "LINE_ERROR": np.linspace(-3.0, 4.0, 8),
        }
    )
    train_df = df.iloc[:6].copy()
    valid_df = df.iloc[6:].copy()

    fit_predict_fn = build_model_bundle_fit_predict_function(
        metadata,
        target_col="LINE_ERROR",
    )
    fold_predictions = fit_predict_fn(train_df, valid_df)

    assert fold_predictions.shape == (2,)

    class SumModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return X.sum(axis=1).to_numpy(dtype=float)

    loaded_predictions = predict_with_loaded_model_bundle(
        SumModel(),
        valid_df,
        metadata=metadata,
    )
    expected = valid_df.loc[:, ["feature_b", "feature_a"]].sum(axis=1).to_numpy(dtype=float)

    assert np.allclose(loaded_predictions, expected)


def test_prepare_model_bundle_feature_frame_fills_missing_saved_columns_with_nan() -> None:
    metadata = {
        "schema": {"feature_names": ["feature_b", "feature_missing", "feature_a"]},
        "training_metrics": {"best_params": {"max_depth": 2}},
    }
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )

    aligned = prepare_model_bundle_feature_frame(df, metadata=metadata)

    assert list(aligned.columns) == ["feature_b", "feature_missing", "feature_a"]
    assert aligned["feature_missing"].isna().all()


def test_evaluate_probability_calibration_predictions_supports_precomputed_fold_predictions() -> None:
    valid_df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range("2024-02-01", periods=4, freq="D"),
            "LINE_ERROR": np.array([-3.0, 2.0, -1.0, 4.0]),
            "TOTAL_POINTS": np.array([217.0, 224.0, 219.0, 228.0]),
            "TOTAL_LINE_bet365": np.array([220.0, 220.0, 220.0, 220.0]),
        }
    )
    calibration_oof_predictions = pd.DataFrame(
        {
            "raw_prob_over": np.array([0.2, 0.35, 0.6, 0.8]),
            "actual_over": np.array([0.0, 0.0, 1.0, 1.0]),
        }
    )
    residual_scale_estimate = ResidualScaleEstimate(
        scale=2.5,
        method="std",
        n_residuals=20,
        residuals=np.linspace(-2.0, 2.0, 20),
        oof_predictions=pd.DataFrame(),
    )

    result = evaluate_probability_calibration_predictions(
        valid_df=valid_df,
        y_pred_reg=np.array([-2.0, 1.5, -0.5, 3.5]),
        residual_scale_estimate=residual_scale_estimate,
        calibration_oof_predictions=calibration_oof_predictions,
        target_col="LINE_ERROR",
        line_col="TOTAL_LINE_bet365",
        prediction_type="line_error",
        calibration_methods=("isotonic", "sigmoid"),
        min_calibration_samples=2,
    )

    assert set(result.strategy_summary["strategy"]) == {
        "regression_threshold",
        "probability_raw",
        "probability_isotonic",
        "probability_sigmoid",
    }
    assert "raw_prob_over" in result.predictions.columns
    assert "isotonic_prob_over" in result.predictions.columns
    assert "sigmoid_prob_over" in result.predictions.columns
