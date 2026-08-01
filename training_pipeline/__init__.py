"""training_pipeline: a reusable, comparable training/experiment pipeline for
the NBA over/under predictor's XGBoost models (TOTAL_POINTS and LINE_ERROR).

See docstrings in config.py, pipeline.py, baseline.py, and leaderboard.py for
the main entry points.
"""

from training_pipeline.backtest import (
    DailyBacktestResult,
    run_daily_backtest,
    xgb_params_from_trial,
)
from training_pipeline.baseline import BaselineMetrics, compute_baseline_metrics
from training_pipeline.betting import (
    DECIMAL_ODDS_MINUS_110,
    BettingMetrics,
    betting_threshold_sweep,
    break_even_win_rate,
    evaluate_betting,
)
from training_pipeline.config import (
    BacktestConfig,
    BettingConfig,
    CVStrategy,
    ExperimentConfig,
    RefitStrategy,
    TargetFamily,
)
from training_pipeline.leaderboard import build_leaderboard, headline_leaderboard
from training_pipeline.pipeline import ExperimentResult, run_experiment

# training_pipeline.promote is deliberately NOT re-exported: it is a CLI
# module run via `python -m training_pipeline.promote`, and importing it
# here would make that emit a double-import RuntimeWarning. Import it
# directly: from training_pipeline.promote import ...
from training_pipeline.reuse import (
    RunHyperparameters,
    find_best_run_hyperparameters,
    load_run_hyperparameters,
)

__all__ = [
    "BacktestConfig",
    "DailyBacktestResult",
    "run_daily_backtest",
    "xgb_params_from_trial",
    "BaselineMetrics",
    "compute_baseline_metrics",
    "BettingConfig",
    "BettingMetrics",
    "DECIMAL_ODDS_MINUS_110",
    "betting_threshold_sweep",
    "break_even_win_rate",
    "evaluate_betting",
    "CVStrategy",
    "ExperimentConfig",
    "RefitStrategy",
    "TargetFamily",
    "build_leaderboard",
    "headline_leaderboard",
    "RunHyperparameters",
    "load_run_hyperparameters",
    "find_best_run_hyperparameters",
    "ExperimentResult",
    "run_experiment",
]
