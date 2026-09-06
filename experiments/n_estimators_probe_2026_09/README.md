# `n_estimators` causal probe

This campaign isolates the change from fold-local early stopping to a single
Optuna-tuned number of boosting rounds.

## Modern-data controls

Cells `a`, `b`, `e`, and `f` are exact controls for the completed runs in
`legacy_cv_replication_2026_09`. They retain the modern data, 90-day holdout,
pooled objective, top-15% / 0.04 tie rule, seed 16, and historical 12x50 fold
layout. Their only intentional change is `tune_n_estimators: false`.

## Historical-data replay

Cells `c`/`d` and `g`/`h` use the July 4 training snapshot and the historical
60-day, season>=2021, max-NA 80, correlation 0.995, mean-fold objective and
fixed +0.10 selection rule. Each pair differs only in `tune_n_estimators`.

The source CSV predates the `ODDS_` naming invariant. Run
`scripts/prepare_legacy_odds_prefixed_csv.py` first. It verifies the original
checksum and changes only the header; every data row is copied byte-for-byte.

The historical line-error configuration requested 4,500 training games even
though its five earliest folds contain only 3,952--4,499 usable games. This is
the behavior of the original run, so part 1 passes the explicit preflight flag
`--allow-short-training-windows` and records a warning instead of silently
changing the historical geometry.

Run the line-error half and total-points half separately:

```bash
bash experiments/runners/run_n_estimators_probe_2026_09_part1.sh
bash experiments/runners/run_n_estimators_probe_2026_09_part2.sh
```
