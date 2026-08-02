"""Experiment configuration schema for training_pipeline.

`ExperimentConfig` is the single object that fully describes one training run:
which CSV to load, how to clean it, how to split it for CV/holdout, which
target to train, how to tune it with Optuna, how to refit the final model,
and where to save results. It is designed so that every axis observed to vary
across the repo's example notebooks (lab/total_points/regressor/*.ipynb,
lab/diff_from_line/regressor/*.ipynb) is a config field, not a hardcoded
notebook constant.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from nba_ou.config.constants import SEASON_TYPE_MAP
from pydantic import BaseModel, ConfigDict, Field, model_validator

from training_pipeline.betting import DECIMAL_ODDS_MINUS_110, DEFAULT_EDGE_THRESHOLDS


class TargetFamily(StrEnum):
    TOTAL_POINTS = "total_points"
    LINE_ERROR = "line_error"


class CVStrategy(StrEnum):
    TEST_ANCHORED = "test_anchored"
    LAST_N_SEASONS = "last_n_seasons"


class RefitStrategy(StrEnum):
    """How the production model is fitted once evaluation is done."""

    #: Last ``walk_forward.train_games`` games of the FULL dataset (dev+test).
    ROLLING_WINDOW = "rolling_window"
    #: Every available game, ignoring the window.
    FULL_DATASET = "full_dataset"


class HoldoutEvaluation(StrEnum):
    """How the held-out test period is scored.

    ``daily_walk_forward`` retrains once per test day on the games available
    strictly before it, so the test period is scored the way production would
    actually operate. It costs one model fit per game-day.

    ``single_shot`` fits one model on the dev window and predicts the whole
    test period at once. Much cheaper, and useful for smoke runs, but it
    understates how a daily-retrained model behaves because the model never
    absorbs completed test days.
    """

    DAILY_WALK_FORWARD = "daily_walk_forward"
    SINGLE_SHOT = "single_shot"


#: Season types kept when ``DataConfig.exclude_playoffs`` is True. Values must
#: match nba_ou.config.constants.SEASON_TYPE_MAP. Playoff basketball has
#: shorter rotations and slower pace -- in the current training data playoff
#: games average ~11.6 fewer total points than regular-season games -- so it is
#: a materially different distribution from what the models are used to predict.
DEFAULT_ALLOWED_SEASON_TYPES: tuple[str, ...] = (
    "Regular Season",
    "Play-In Tournament",
)


class DataConfig(BaseModel):
    csv_path: Path
    date_col: str = "GAME_DATE"
    season_col: str = "SEASON_YEAR"
    #: Season type is resolved from this column's 3-character prefix rather
    #: than from the SEASON_TYPE text column, which mislabels Play-In
    #: Tournament games as "Playoffs" (see training_pipeline.data).
    game_id_col: str = "GAME_ID"
    season_year_floor: int | None = None
    #: Drop games whose season type is not in ``allowed_season_types``.
    #: Defaults to True: regular season + play-in only.
    exclude_playoffs: bool = True
    allowed_season_types: tuple[str, ...] = DEFAULT_ALLOWED_SEASON_TYPES
    #: Human label for the dataset snapshot, e.g. "20260318-all-odds". Purely
    #: descriptive; the checksum below is what actually identifies the bytes.
    data_version: str | None = None
    #: Optional integrity assertion. When set, prepare_dataset verifies the
    #: CSV's sha256 matches and raises if it does not -- so regenerating a CSV
    #: in place is caught loudly instead of silently changing your results.
    #: The actual checksum is always recorded in run metadata regardless.
    expected_checksum: str | None = None

    @model_validator(mode="after")
    def _validate_allowed_season_types(self) -> DataConfig:
        if self.exclude_playoffs and not self.allowed_season_types:
            raise ValueError(
                "data.allowed_season_types must not be empty when "
                "data.exclude_playoffs is True (it would drop every row)."
            )
        unknown = set(self.allowed_season_types) - set(SEASON_TYPE_MAP.values())
        if unknown:
            raise ValueError(
                f"Unknown season types {sorted(unknown)}. Valid values: "
                f"{sorted(set(SEASON_TYPE_MAP.values()))}"
            )
        return self


class CleaningConfig(BaseModel):
    """Mirrors nba_ou.data_processing.missing_data.clean_df_for_training.clean_dataframe_for_training."""

    nan_threshold: float = 5.0
    corr_threshold: float = 0.995
    max_na_per_row: int = -1
    create_missing_flags: bool = False
    keep_columns: list[str] | None = None
    exclude_cols_containing: list[str] | None = None
    keep_all_cols: bool = False
    verbose: int = 1
    strict_mode: int = -1
    strict_mode_exclude_cols: list[str] | None = None


class HoldoutConfig(BaseModel):
    test_size: float | None = 0.15
    test_games: int | None = None

    @model_validator(mode="after")
    def _exactly_one_of_test_size_or_test_games(self) -> HoldoutConfig:
        if (self.test_size is None) == (self.test_games is None):
            raise ValueError(
                "Provide exactly one of holdout.test_size or holdout.test_games."
            )
        return self


class WalkForwardConfig(BaseModel):
    strategy: CVStrategy = CVStrategy.TEST_ANCHORED
    test_games: int = 30
    step_games_between_tests: int | None = None
    train_games: int | None = 5000
    train_seasons: int | None = None
    min_train_games: int = 300
    max_folds: int | None = 12
    fold_selection: str = "latest"
    exclude_test_months: tuple[int, ...] = (5, 6)
    require_same_season_test: bool = True
    verbose: int = 0

    @model_validator(mode="after")
    def _train_seasons_required_for_last_n_seasons(self) -> WalkForwardConfig:
        if self.strategy == CVStrategy.LAST_N_SEASONS:
            if self.train_seasons is None or self.train_seasons <= 0:
                raise ValueError(
                    "walk_forward.train_seasons must be a positive int when "
                    "walk_forward.strategy == 'last_n_seasons'."
                )
        return self


#: Weights are exp(-lambda * age_in_days), so lambda maps to a half-life:
#:
#:   lambda   half-life        weight of the oldest game in a 2500-game window
#:   0.0005   1386 d (3.8 yr)  68%    gentle -- barely tilts toward recency
#:   0.001     693 d (1.9 yr)  46%
#:   0.002     347 d (0.9 yr)  21%
#:   0.005     139 d (0.4 yr)   2%    moderate -- the practical ceiling
#:   0.010      69 d (0.2 yr)   0.05% window effectively discarded
#:   0.050      14 d           ~0%    absurd
#:
#: A 2500-game window spans roughly 770 calendar days, so anything above
#: ~0.005 shrinks the effective training set to a few weeks and makes the
#: configured window meaningless. The bounds stop there deliberately.
_DEFAULT_SAMPLE_WEIGHT_LAMBDA_BOUNDS = (0.0005, 0.005)


class SampleWeightConfig(BaseModel):
    """Exponential recency weighting: recent games count for more.

    Available to BOTH target families. Upstream's optuna_total_points.py has no
    sample-weight parameters, so training_pipeline supplies its own objective
    and final-fit path (see tuning.run_objective / tuning.fit_final_model) that
    apply weights identically for TOTAL_POINTS and LINE_ERROR.

    A weight of exp(-lambda * age_in_days) is applied per training row, so
    lambda=0.005 gives a game one year old roughly 16% of today's weight.
    """

    enabled: bool = False
    lambda_: float | None = None
    tune_lambda: bool = False
    lambda_bounds: tuple[float, float] = _DEFAULT_SAMPLE_WEIGHT_LAMBDA_BOUNDS
    #: When tuning, let Optuna also decide *whether* to weight at all, via a
    #: categorical use_sample_weight parameter. Without this the search is
    #: forced to weight, and "no weighting" can only be approximated by
    #: driving lambda toward zero -- which a log-uniform range can never
    #: actually reach, and which wastes sampler resolution near the floor.
    #: With it, "off" is one clean binary decision the sampler can evaluate,
    #: and you can read off how many trials preferred it.
    allow_unweighted: bool = True
    date_col: str = "GAME_DATE"

    @model_validator(mode="after")
    def _validate_bounds(self) -> SampleWeightConfig:
        low, high = self.lambda_bounds
        if low <= 0:
            raise ValueError(
                "sample_weight.lambda_bounds lower bound must be > 0 (the range "
                "is log-uniform). Use allow_unweighted=True to let Optuna turn "
                "weighting off instead of trying to sample zero."
            )
        if low >= high:
            raise ValueError("sample_weight.lambda_bounds must be (low, high) with low < high.")
        if self.lambda_ is not None and self.lambda_ < 0:
            raise ValueError("sample_weight.lambda_ must be >= 0.")
        return self

    def is_default(self) -> bool:
        return self == SampleWeightConfig()


class IntRange(BaseModel):
    low: int
    high: int
    log: bool = False

    @model_validator(mode="after")
    def _validate(self) -> IntRange:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        return self


class FloatRange(BaseModel):
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _validate(self) -> FloatRange:
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        if self.log and self.low <= 0:
            raise ValueError("log-scaled ranges require low > 0.")
        return self


class SearchSpaceConfig(BaseModel):
    """XGBoost hyperparameter search space for Optuna.

    Defaults are tuned for this problem's shape: ~2500 training games against
    1400-2100 features (p/n roughly 0.6-0.8), with a signal worth only ~3% of
    baseline MAE. Two ranges deviate deliberately from the legacy space in
    ``nba_ou.modeling.optuna_*.py`` (preserved as UPSTREAM_SEARCH_SPACE):

    * ``colsample_bytree`` reaches down to 0.05. At the old 0.35 floor a tree
      chose ~15 splits from ~720 candidate features on a noisy bootstrap,
      which mines noise rather than signal.
    * ``gamma`` spans 1-500 (log). For ``reg:squarederror`` with residual
      sigma ~19.5, a 500-row node's gradient sum fluctuates by ~436 from
      chance alone, so chance split gains are O(100s) -- the old 0.1-3.0 range
      pruned nothing at all. Gamma is the direct "only split on a real gain"
      control, which is exactly what a noise-dominated target needs.

    ``min_child_weight`` also starts at 20 rather than 5: with squared error
    the hessian is 1, so it is effectively minimum samples per leaf, and a
    leaf holding 5 of 2500 games is noise. ``max_depth`` reaches down to 1
    because additive stumps cannot manufacture spurious interactions.

    These are reasoned from the problem's structure, not measured -- run the
    A/B against UPSTREAM_SEARCH_SPACE on a fixed holdout before trusting them.
    Note a wider space needs more trials to search, so under a tight budget it
    can underperform a narrow one.

    Changing any range changes the config fingerprint, so a persistent study
    will start fresh rather than mixing incomparable trials.
    """

    max_depth: IntRange = IntRange(low=1, high=4)
    min_child_weight: FloatRange = FloatRange(low=20.0, high=250.0, log=True)
    gamma: FloatRange = FloatRange(low=1.0, high=500.0, log=True)
    subsample: FloatRange = FloatRange(low=0.55, high=0.95)
    colsample_bytree: FloatRange = FloatRange(low=0.05, high=0.8, log=True)
    learning_rate: FloatRange = FloatRange(low=0.0075, high=0.06, log=True)
    reg_alpha: FloatRange = FloatRange(low=1e-2, high=20.0, log=True)
    reg_lambda: FloatRange = FloatRange(low=1.0, high=500.0, log=True)
    #: Upper bound on boosting rounds; early stopping picks the real number.
    n_estimators: int = 1000
    early_stopping_rounds: int = 70


#: The search space hardcoded in nba_ou.modeling.optuna_*.py, kept verbatim so
#: pre-existing results stay reproducible and so the A/B against the new
#: defaults is one config change. A test asserts this still produces draws
#: identical to upstream's builders under a seeded sampler.
UPSTREAM_SEARCH_SPACE = SearchSpaceConfig(
    max_depth=IntRange(low=2, high=4),
    min_child_weight=FloatRange(low=5.0, high=60.0, log=True),
    gamma=FloatRange(low=0.1, high=3.0),
    subsample=FloatRange(low=0.55, high=0.95),
    colsample_bytree=FloatRange(low=0.35, high=0.8),
    learning_rate=FloatRange(low=0.0075, high=0.06, log=True),
    reg_alpha=FloatRange(low=1e-2, high=20.0, log=True),
    reg_lambda=FloatRange(low=1.0, high=50.0, log=True),
)


class OptunaConfig(BaseModel):
    """Optuna tuning knobs.

    Note on ``persistent_storage``: when enabled, the study is stored in a
    SQLite file keyed by the experiment name *and a fingerprint of the
    data/CV-affecting config* (see ExperimentConfig.fingerprint), so changing
    the CSV, cleaning thresholds, or fold layout starts a fresh study instead
    of silently resuming trials computed on different data.

    Note on ``n_trials`` when resuming: Optuna's ``study.optimize(n_trials=N)``
    runs N *additional* trials on a resumed study, it is not a target total.
    Re-running the same persistent config with n_trials=80 twice yields 160
    trials.
    """

    n_trials: int = 80
    timeout: int | None = None
    objective_name: str = "reg:squarederror"
    study_name: str | None = None
    persistent_storage: bool = False
    storage_filename: str = "optuna_study.db"
    load_if_exists: bool = True
    mae_tolerance_abs: float | None = 0.10
    mae_tolerance_pct: float | None = None
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)

    #: Skip tuning entirely and use these hyperparameters. Populate from a
    #: previous run with training_pipeline.reuse.load_run_hyperparameters(),
    #: which prints a ready-to-paste YAML block -- that is how you avoid
    #: re-running a long study just to reuse what it already found.
    fixed_params: dict[str, Any] | None = None
    fixed_n_estimators: int | None = None
    fixed_sample_weight_lambda: float | None = None

    @property
    def skip_tuning(self) -> bool:
        return self.fixed_params is not None

    @model_validator(mode="after")
    def _validate_fixed_params(self) -> OptunaConfig:
        if self.fixed_params is not None and not self.fixed_n_estimators:
            raise ValueError(
                "optuna.fixed_n_estimators is required alongside "
                "optuna.fixed_params (there is no study to infer it from)."
            )
        return self

    @model_validator(mode="after")
    def _at_most_one_mae_tolerance(self) -> OptunaConfig:
        if self.mae_tolerance_abs is not None and self.mae_tolerance_pct is not None:
            raise ValueError(
                "Provide at most one of optuna.mae_tolerance_abs or "
                "optuna.mae_tolerance_pct."
            )
        return self


class RefitConfig(BaseModel):
    """How the final model is fitted once tuning has chosen hyperparameters.

    There is deliberately no ``train_games`` here: the rolling-window size is
    ``walk_forward.train_games``, so the final model is trained on the same
    amount of history each CV fold used. Two separate knobs could silently
    diverge, which would mean selecting hyperparameters for one training-set
    size and then fitting on another.
    """

    strategy: RefitStrategy = RefitStrategy.ROLLING_WINDOW
    use_lexicographic_selection: bool = True
    #: Train (and save) the production model. Off by default: most experiments
    #: only want the evaluation. When on, the model is fitted on the freshest
    #: data available -- the last walk_forward.train_games games of dev AND
    #: test, with no split -- because in production nothing is held back.
    train_production_model: bool = False


class BettingConfig(BaseModel):
    """How model predictions get turned into bets and scored for profit.

    Post-hoc evaluation only -- these settings never change what a model or an
    Optuna trial *is*, so they are excluded from ExperimentConfig.fingerprint()
    and can be changed without forking a persistent study.
    """

    edge_thresholds: tuple[float, ...] = DEFAULT_EDGE_THRESHOLDS
    # The threshold whose metrics get promoted to the leaderboard. 2.0 points
    # is a reasonable default: it filters out coin-flip calls where the model
    # essentially agrees with the line.
    primary_edge_threshold: float = 2.0
    flat_decimal_odds: float = DECIMAL_ODDS_MINUS_110
    # Opt in to real per-book prices (decimal) instead of the flat price, e.g.
    # "total_bet365_price_over" / "total_bet365_price_under". Rows with missing
    # or invalid prices fall back to flat_decimal_odds.
    over_price_col: str | None = None
    under_price_col: str | None = None

    #: Also score the CV folds for profit, not just the holdout. Costs one extra
    #: fit per fold (~= one Optuna trial) and buys roughly 5x the bet volume:
    #: 12 folds x ~50 validation games ~= 600, versus ~290 in a 5% holdout. At
    #: these sample sizes volume is the binding constraint on being able to tell
    #: a real edge from a lucky one, so this is close to free power.
    #:
    #: Read it as a COMPARISON between configurations, not as an unbiased
    #: estimate of live ROI: the hyperparameters were selected on these same
    #: folds, so the number is optimistically biased. The holdout stays the
    #: out-of-sample estimate.
    evaluate_cv_folds: bool = True

    #: Extra total-line columns to re-score the very same predictions against,
    #: for information only -- nothing about training or selection changes.
    #:
    #: Why it matters: bets are settled here against the CLOSING line, which by
    #: definition is the last price quoted and therefore one you cannot actually
    #: take. Beating it is the strict test of "do I have information", but it is
    #: not the number you would have got. In this dataset the line moves 2.54
    #: points on average and moves at all in 93% of games -- larger than the
    #: default 2.0-point bet trigger -- so an edge measured against the close can
    #: correspond to a materially different bet against the open.
    #:
    #: "TOTAL_LINE_consensus_opener" is the natural counterpart. Columns absent
    #: from the data are skipped rather than raising, since line availability
    #: varies across CSV snapshots.
    comparison_line_cols: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> BettingConfig:
        if not self.edge_thresholds:
            raise ValueError("betting.edge_thresholds must not be empty.")
        if self.flat_decimal_odds <= 1.0:
            raise ValueError("betting.flat_decimal_odds must be > 1.0.")
        if (self.over_price_col is None) != (self.under_price_col is None):
            raise ValueError(
                "Provide both betting.over_price_col and betting.under_price_col, "
                "or neither."
            )
        if self.primary_edge_threshold not in self.edge_thresholds:
            raise ValueError(
                f"betting.primary_edge_threshold ({self.primary_edge_threshold}) must "
                f"be one of betting.edge_thresholds ({self.edge_thresholds})."
            )
        return self


#: Hyperparameters for the daily backtest when none are supplied. These are the
#: hand-tuned values from the repo's 3-season total_points notebook -- a known
#: working configuration rather than an invented one.
DEFAULT_BACKTEST_XGB_PARAMS: dict[str, float | int] = {
    "max_depth": 4,
    "learning_rate": 0.057,
    "subsample": 0.8,
    "colsample_bytree": 0.86,
    "reg_alpha": 0.57,
    "reg_lambda": 1.78,
    "min_child_weight": 5.48,
    "gamma": 1.77,
}
DEFAULT_BACKTEST_N_ESTIMATORS = 75


class BacktestConfig(BaseModel):
    """Daily walk-forward backtest: retrain every day, predict that day only.

    This simulates production operation. For each game-day in the backtest
    window the model is refitted on every game that finished strictly before
    that day (including earlier backtest days, which become training data once
    played), then predicts that day's games. No result is ever visible to the
    model before it would have been in real life.

    Hyperparameters are held fixed across all days -- re-tuning daily would be
    both prohibitively slow and unrealistic. Note that if those hyperparameters
    came from an Optuna study fitted on data overlapping the backtest window,
    the backtest inherits that selection bias; tune on data preceding the
    window to keep it clean.
    """

    #: ~200 games is one month of NBA regular season; 300 is roughly 6.5 weeks
    #: and ~40 retrains, which buys meaningfully more bets for significance.
    test_games: int = 300
    #: Rolling training window in games. None keeps every prior game
    #: (expanding window), which is closest to how retraining actually runs.
    train_games: int | None = None
    xgb_params: dict[str, Any] | None = None
    n_estimators: int | None = None
    show_progress: bool = True

    @model_validator(mode="after")
    def _validate(self) -> BacktestConfig:
        if self.test_games <= 0:
            raise ValueError("backtest.test_games must be > 0.")
        if self.train_games is not None and self.train_games <= 0:
            raise ValueError("backtest.train_games must be > 0 when provided.")
        if self.n_estimators is not None and self.n_estimators <= 0:
            raise ValueError("backtest.n_estimators must be > 0 when provided.")
        return self

    def resolved_xgb_params(self) -> dict[str, Any]:
        return dict(self.xgb_params or DEFAULT_BACKTEST_XGB_PARAMS)

    def resolved_n_estimators(self) -> int:
        return self.n_estimators or DEFAULT_BACKTEST_N_ESTIMATORS


class BaselineConfig(BaseModel):
    """Controls which column stands in for "the bookmaker's line" baseline.

    Resolution order (see training_pipeline.data.resolve_baseline_line_col):
    1. line_col, if set explicitly here -- e.g. set this to
       "odds_total_line_books_median" to use the engineered cross-book
       median instead of a single book's line, once you've verified that
       column is trustworthy in the specific CSV you're using (it was found
       empirically NOT to be points-scale in at least one archived CSV
       snapshot, so it is never used as a silent default).
    2. nba_ou.config.odds_columns.resolve_main_total_line_col(df, book=book)
       -- the same single-book line already used everywhere else in the
       pipeline for cleaning and OU-accuracy scoring.
    """

    line_col: str | None = None
    book: str | None = None


class ExperimentConfig(BaseModel):
    experiment_name: str
    #: Manually curated label for the training approach behind this run, e.g.
    #: "2.1" or "v3-style-matchup-features". Set it by hand whenever you change
    #: something you want to be able to group runs by later; it is recorded in
    #: the run metadata, surfaced on the leaderboard, and written into the saved
    #: model bundle's ``training_code_tag``.
    #:
    #: Deliberately excluded from fingerprint(): this is a human label, so
    #: renaming it must not fork a persistent Optuna study. Changes that
    #: genuinely alter what a trial means (data, cleaning, folds, objective)
    #: are already captured by the fingerprint on their own.
    training_version: str | None = None
    #: How the held-out test period is scored. See HoldoutEvaluation.
    holdout_evaluation: HoldoutEvaluation = HoldoutEvaluation.DAILY_WALK_FORWARD

    # --- research log: why this run exists and what it should be read next to.
    # All are labels, so all are excluded from fingerprint().
    #: Runs sharing a comparison_group are meant to be read against each other
    #: (same data, same holdout, one deliberate difference). This is the honest
    #: answer to "are these numbers comparable?" -- the leaderboard can group by
    #: it instead of ranking unrelated cohorts together.
    comparison_group: str | None = None
    #: What this run is supposed to tell you, e.g. "does a 5000-game window beat
    #: 2500 on ROI?". Costs one line now and saves guessing in six months.
    hypothesis: str | None = None
    tags: tuple[str, ...] = ()
    target_family: TargetFamily
    line_col: str | None = None

    data: DataConfig
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    holdout: HoldoutConfig = Field(default_factory=HoldoutConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    sample_weight: SampleWeightConfig = Field(default_factory=SampleWeightConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    refit: RefitConfig = Field(default_factory=RefitConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    betting: BettingConfig = Field(default_factory=BettingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    exclude_cols: list[str] = Field(
        default_factory=lambda: ["TOTAL_POINTS", "SEASON_YEAR", "GAME_DATE"]
    )

    window_dir_label: str | None = None
    window_name_label: str | None = None

    save_experiment_artifacts: bool = True
    experiment_root_dir: Path = Path("artifacts/experiments")
    model_output_root: Path = Path("models")
    overwrite_existing_model: bool = False
    #: The seed for the Optuna sampler AND every model fit. Threaded everywhere
    #: rather than hardcoded, which is what makes evaluation_seeds possible.
    random_state: int = 16

    #: Additional seeds to repeat the holdout evaluation under, holding the data,
    #: the split and the tuned hyperparameters fixed so the only thing that
    #: changes is XGBoost's own randomness (subsample / colsample draws).
    #:
    #: This measures the error bar you have otherwise been comparing experiments
    #: without. If two configurations differ by less than the spread across
    #: seeds, the difference between them is not evidence of anything. Empty by
    #: default because each extra seed re-runs the whole evaluation (~35 fits
    #: under daily_walk_forward); two or three extras is the useful range.
    #:
    #: random_state itself is always evaluated and is the reported headline; it
    #: is removed from this list if repeated, and duplicates are dropped.
    evaluation_seeds: tuple[int, ...] = ()

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _normalize_evaluation_seeds(self) -> ExperimentConfig:
        """Drop duplicates and the primary seed, preserving order.

        The primary seed is always run as the headline evaluation, so leaving it
        in the extras list would double the work and report the same fit twice
        as if it were independent evidence of stability.
        """
        seen: set[int] = {self.random_state}
        deduped: list[int] = []
        for seed in self.evaluation_seeds:
            if seed not in seen:
                seen.add(seed)
                deduped.append(seed)
        self.evaluation_seeds = tuple(deduped)
        return self

    @model_validator(mode="after")
    def _normalize_training_version(self) -> ExperimentConfig:
        """Treat a blank/whitespace label as unset rather than a real version."""
        if self.training_version is not None:
            stripped = self.training_version.strip()
            self.training_version = stripped or None
        return self

    @model_validator(mode="after")
    def _validate_target_family_constraints(self) -> ExperimentConfig:
        if self.target_family == TargetFamily.TOTAL_POINTS:
            if not self.line_col:
                raise ValueError(
                    "line_col is required when target_family == 'total_points' "
                    "(optuna_total_points.py scores against the betting line)."
                )
        else:  # LINE_ERROR
            if self.line_col:
                raise ValueError(
                    "line_col must be omitted when target_family == 'line_error' "
                    "(optuna_error_line.py never uses a line column)."
                )
            if "LINE_ERROR" not in self.exclude_cols:
                self.exclude_cols = [*self.exclude_cols, "LINE_ERROR"]

        if "TOTAL_POINTS" not in self.exclude_cols:
            self.exclude_cols = [*self.exclude_cols, "TOTAL_POINTS"]

        return self

    @property
    def resolved_window_dir_label(self) -> str:
        if self.window_dir_label:
            return self.window_dir_label
        train_games = self.walk_forward.train_games
        return f"{train_games}_games" if train_games else "full_dataset"

    @property
    def resolved_window_name_label(self) -> str:
        if self.window_name_label:
            return self.window_name_label
        return self.resolved_window_dir_label

    @property
    def resolved_study_name(self) -> str:
        if self.optuna.study_name:
            return self.optuna.study_name
        return f"xgb_{self.target_family.value}_{self.resolved_window_name_label}_mae"

    def fingerprint(self) -> str:
        """Short stable hash of everything that changes what a trial *means*.

        Used to key persistent Optuna storage so a study can only ever be
        resumed by a config that would produce comparable trials. Purely
        cosmetic/output-side fields (experiment name, labels, where artifacts
        and models are written, whether artifacts are saved at all) are
        excluded, so renaming an experiment does not throw away its study.
        """
        payload = self.model_dump(
            mode="json",
            exclude={
                "experiment_name": True,
                "training_version": True,
                # Research-log labels: they describe intent, not computation.
                "comparison_group": True,
                "hypothesis": True,
                "tags": True,
                # data_version is a label; expected_checksum IS included via
                # DataConfig, since different bytes mean incomparable trials.
                "data": {"data_version"},
                "window_dir_label": True,
                "window_name_label": True,
                "save_experiment_artifacts": True,
                "experiment_root_dir": True,
                "model_output_root": True,
                "overwrite_existing_model": True,
                # Repeating the evaluation under more seeds measures the result,
                # it does not change what a trial means. random_state is NOT
                # excluded: it does change every fit.
                "evaluation_seeds": True,
                # Post-hoc scoring settings; changing a bet threshold or a
                # backtest window must not invalidate an existing study's trials.
                "betting": True,
                "backtest": True,
                # n_trials/timeout change how *long* tuning runs, not what a
                # trial means, so resuming across them is legitimate.
                "optuna": {"n_trials", "timeout", "study_name"},
            },
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
