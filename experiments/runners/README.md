# Experiment runners

Shell entry points that validate and run groups of experiment YAML files live
here. Run them from anywhere; each script resolves the repository root before
accessing configs, datasets, logs, or artifacts.

## Available campaigns

- `run_rolling_origin_campaign.sh`: the current protocol -- rolling-origin CV
  (train to a date, predict the next 4 game-days), with the training window and
  the boosting rounds both tuned by Optuna. 2 runs, one per regression target.
- `run_time_decay_2026_09_part1.sh` and `run_time_decay_2026_09_part2.sh`:
  parallel halves of the regenerated no-decay campaign. Together they run the
  three closing targets on normalized 2.2 data, spread error at T-30, T-360,
  and T-720, plus a closing spread-error comparison on schema 2.1.

## Archived campaigns

Everything that ran under the previous training protocol lives in
`../archived/runners/`, alongside the configs it drives and a frozen
`_base.yaml` that keeps those runs reproducing as they originally did. See
`../archived/README.md` for what changed and why the two sets of numbers are not
comparable.

Run a campaign in the foreground:

```bash
bash experiments/runners/run_rolling_origin_campaign.sh
```

Run it detached:

```bash
nohup bash experiments/runners/run_rolling_origin_campaign.sh > /dev/null 2>&1 &
```

Each runner documents its log paths and any supported resume options at the top
of the script.
