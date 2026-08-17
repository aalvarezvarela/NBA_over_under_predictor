# Experiment runners

Shell entry points that validate and run groups of experiment YAML files live
here. Run them from anywhere; each script resolves the repository root before
accessing configs, datasets, logs, or artifacts.

## Available campaigns

- `run_total_points_dataset_ab.sh`: compares the old and 2.0 training datasets
  with the same total-points experiment.
- `run_strategy_window_campaign.sh`: runs the three prediction strategies over
  the 2,500- and 3,750-game windows.
- `run_window_overtime_campaign.sh`: follow-up to the above, varying five axes
  against its winner (`line_error` at 3,750) -- training window, overtime rows,
  feature build, playoffs, and the per-row missing-data cap. 11 runs.
- `run_intermediate_line_campaign.sh`: runs the intermediate-line pooled
  snapshot models and their 12-hour controls. 4 runs.
- `../intermediate_line_2026_08/run_line_error_7snapshot_6h_4h.sh`: runs the
  unweighted, line-error-only seven-snapshot model and its independent 6h/4h
  controls. Its matching data-preparation runner is colocated with the configs
  so the complete follow-up campaign stays grouped in one folder.

Run a campaign in the foreground:

```bash
bash experiments/runners/run_strategy_window_campaign.sh
```

Run it detached:

```bash
nohup bash experiments/runners/run_strategy_window_campaign.sh > /dev/null 2>&1 &
```

Each runner documents its log paths and any supported resume options at the top
of the script.
