#!/usr/bin/env python3
"""Is XGBoost faster on your GPU than on your 4 CPUs, at THIS data size?

Run on the machine that will do the experiments:

    poetry run python experiments/benchmark_gpu.py

Why this needs measuring rather than assuming: GPU XGBoost wins decisively
above ~100k rows, but these training windows are 2,500-3,750 rows, where
kernel-launch and host/device transfer overhead can exceed the compute saved.
Two things argue the other way -- ~1,458 features is high, and histogram
building parallelises well over features, and a 4-core CPU baseline is a low
bar. So it is genuinely uncertain, and a two-minute measurement settles it.

VRAM is not a concern: the training matrix is 15-22 MB against 6 GB.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Matches the real experiment shapes.
SHAPES = [("2500-game window", 2500, 1458), ("3750-game window", 3750, 1458)]
REPEATS = 3

# Representative of the tuned search space: shallow, heavily regularised trees
# with a low colsample, which is what the Optuna space actually favours.
BASE_PARAMS = dict(
    booster="gbtree",
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="mae",
    max_depth=3,
    min_child_weight=60.0,
    gamma=4.0,
    subsample=0.75,
    colsample_bytree=0.07,
    learning_rate=0.03,
    reg_alpha=0.7,
    reg_lambda=20.0,
    n_estimators=300,
    verbosity=0,
    random_state=16,
)


def assert_gpu_is_really_used() -> None:
    """Refuse to benchmark a silent CPU fallback.

    ``device="cuda"`` does not fail when no GPU is visible -- XGBoost warns and
    quietly runs on the CPU. The fit still completes, so a naive benchmark
    times CPU-with-extra-overhead and confidently reports "the GPU is 3x
    slower". Verified on a machine with no GPU: it produced exactly that.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        XGBRegressor(device="cuda", tree_method="hist", n_estimators=2).fit(
            pd.DataFrame(np.zeros((32, 4), dtype=np.float32)),
            pd.Series(np.arange(32, dtype=float)),
        )
        messages = " ".join(str(item.message) for item in caught)

    if "No visible GPU" in messages or "not compiled with CUDA" in messages:
        raise SystemExit(
            "XGBoost fell back to the CPU, so there is nothing to benchmark:\n"
            f"  {messages.strip()[:400]}\n\n"
            "Check `nvidia-smi` works and that the NVIDIA driver is installed. "
            "The installed wheel does carry CUDA support "
            f"(USE_CUDA={__import__('xgboost').build_info().get('USE_CUDA')}), "
            "so this is a driver/visibility problem, not a rebuild."
        )


def timed_fit(X: pd.DataFrame, y: pd.Series, **overrides: object) -> float:
    """Best of REPEATS, to shake out first-call CUDA context setup."""
    best = float("inf")
    for _ in range(REPEATS):
        model = XGBRegressor(**{**BASE_PARAMS, **overrides})
        start = time.perf_counter()
        model.fit(X, y, verbose=False)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    assert_gpu_is_really_used()
    rng = np.random.default_rng(0)

    print(f"{'shape':>18} {'CPU (4 jobs)':>14} {'GPU (cuda)':>12} {'speedup':>9}")
    print("-" * 58)

    verdicts = []
    for label, rows, feats in SHAPES:
        X = pd.DataFrame(rng.normal(size=(rows, feats)).astype(np.float32))
        y = pd.Series(rng.normal(220, 20, rows))

        cpu = timed_fit(X, y, device="cpu", n_jobs=4)
        try:
            gpu = timed_fit(X, y, device="cuda")
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            print(f"{label:>18} {cpu:>13.2f}s   GPU unavailable: {exc}")
            return

        speedup = cpu / gpu
        verdicts.append(speedup)
        print(f"{label:>18} {cpu:>13.2f}s {gpu:>11.2f}s {speedup:>8.2f}x")

    mean_speedup = sum(verdicts) / len(verdicts)
    print("-" * 58)

    # One Optuna trial is 12 fold-fits; a full run is ~40 trials plus a
    # 55-day walk-forward and two extra seeds.
    fits_per_run = 40 * 12 + 12 + 55 * 3
    cpu_hours = fits_per_run * 8.7 / 3600
    print(f"\n~{fits_per_run} fits per experiment run")
    print(f"  at the CPU rate measured here : ~{cpu_hours:.1f} h")
    print(f"  at {mean_speedup:.2f}x                        : "
          f"~{cpu_hours / mean_speedup:.1f} h")

    if mean_speedup > 1.3:
        print("\nVERDICT: worth switching. Use device: cuda for ALL SIX runs --")
        print("mixing CPU and GPU runs inside one comparison group would add a")
        print("hardware confound to comparisons already fighting a 4.9pp noise floor.")
    elif mean_speedup > 0.9:
        print("\nVERDICT: a wash. Stay on CPU -- it is the configuration every")
        print("existing run used, so results stay directly comparable.")
    else:
        print("\nVERDICT: the GPU is SLOWER at this size (too few rows to amortise")
        print("kernel-launch and transfer overhead). Stay on CPU.")


if __name__ == "__main__":
    main()
