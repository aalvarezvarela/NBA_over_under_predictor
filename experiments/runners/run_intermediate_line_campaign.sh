#!/usr/bin/env bash
# Intermediate-line campaign: pooled snapshot models plus 12h controls.
#
#   bash experiments/runners/run_intermediate_line_campaign.sh
#
# Detached, so closing the terminal does not kill it:
#   nohup bash experiments/runners/run_intermediate_line_campaign.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/campaign_*/campaign.log      # the summary
#   tail -f artifacts/logs/campaign_*/<experiment>.log  # one run in detail
#
# Interrupted? Re-run with SKIP_EXISTING=1 to continue where it stopped:
#   SKIP_EXISTING=1 bash experiments/runners/run_intermediate_line_campaign.sh
#
# The pooled configs MUST use scripts/run_intermediate_snapshot_experiment.py so
# their artifacts include per-snapshot betting reports. The t720 controls are
# one row per game, so the ordinary CLI is the correct runner for those.
#
# NOT `set -e`: one failing run must not cancel the remaining campaign cells.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="intermediate_line_2026_08"
CONFIG_DIR="experiments/${CAMPAIGN}"
POOLED_DATASET="data/train_data/intermediate_line_data_20260412.csv"
T720_DATASET="data/train_data/intermediate_line_data_20260412_t720.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs/campaign_${STAMP}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

# Primary deliverables first. If the run is cut short, the pooled target
# comparison is still interpretable; controls only answer whether pooling is
# earning its complexity at the 12h horizon.
NAMES=(
  "intermediate_pooled_line_error"
  "intermediate_pooled_total_points"
  "intermediate_t720_control_line_error"
  "intermediate_t720_control_total_points"
)
CONFIGS=(
  "${CONFIG_DIR}/pooled_line_error.yaml"
  "${CONFIG_DIR}/pooled_total_points.yaml"
  "${CONFIG_DIR}/t720_control_line_error.yaml"
  "${CONFIG_DIR}/t720_control_total_points.yaml"
)
RUNNERS=(
  "snapshot"
  "snapshot"
  "cli"
  "cli"
)

# -u keeps stdout unbuffered so `tail -f` shows progress live rather than in
# buffered bursts.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
SNAPSHOT=("${PY[@]}" scripts/run_intermediate_snapshot_experiment.py)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log " Campaign : ${CAMPAIGN}"
log " Started  : $(date)"
log " Runs     : ${#CONFIGS[@]}   (2 pooled snapshot runs, 2 t720 controls)"
log " Logs     : ${LOG_DIR}"
log "=========================================================="

# --- required datasets must exist before committing GPU time ----------------
if [[ ! -s "$POOLED_DATASET" ]]; then
  log "ABORT: pooled dataset missing or empty: $POOLED_DATASET"
  log "  Regenerate it with:"
  log "    poetry run python scripts/create_train_data/create_intermediate_line_train_data.py"
  exit 1
fi

log ""
log "Dataset : $POOLED_DATASET ($(du -h "$POOLED_DATASET" | cut -f1))"
log "Checksum: $("${PY[@]}" -c \
  "from training_pipeline.data import compute_file_checksum as c; print(c('$POOLED_DATASET'))" \
  2>/dev/null)"

if ! head -1 "$POOLED_DATASET" | tr ',' '\n' | grep -qx "TIME_TO_MATCH_MIN"; then
  log ""
  log "ABORT: TIME_TO_MATCH_MIN is not in $POOLED_DATASET."
  log "  The pooled runner cannot produce per-snapshot reports without it."
  exit 1
fi

if [[ ! -s "$T720_DATASET" ]]; then
  log ""
  log "ABORT: t720 control dataset missing or empty: $T720_DATASET"
  log "  Build the 12-hour slice with:"
  log "    poetry run python scripts/create_train_data/slice_intermediate_snapshot.py --snapshot 720"
  exit 1
fi

log "Dataset : $T720_DATASET ($(du -h "$T720_DATASET" | cut -f1))"
log "Checksum: $("${PY[@]}" -c \
  "from training_pipeline.data import compute_file_checksum as c; print(c('$T720_DATASET'))" \
  2>/dev/null)"
log "  ^ paste these into data.expected_checksum in the four configs to be told"
log "    loudly if either CSV is ever rebuilt underneath this campaign."

# --- validate every config BEFORE committing the campaign -------------------
log ""
log "Validating configs..."
for i in "${!CONFIGS[@]}"; do
  name="${NAMES[$i]}"
  cfg="${CONFIGS[$i]}"
  runner="${RUNNERS[$i]}"

  if [[ ! -f "$cfg" ]]; then
    log "ABORT: missing config $cfg"
    exit 1
  fi

  if [[ "$runner" == "snapshot" ]]; then
    if ! "${SNAPSHOT[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
      log "ABORT: $cfg failed snapshot-wrapper validation. See $LOG"
      exit 1
    fi
  else
    if ! "${CLI[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
      log "ABORT: $cfg failed CLI validation. See $LOG"
      exit 1
    fi
  fi

  log "  ok  ${name}"
done

# --- run them, one after another -------------------------------------------
# Sequential on purpose: XGBoost already saturates the machine/GPU budget, so
# running two at once would only make both slower and muddy timings.
FAILED=0
COMPLETED=0
SKIPPED=0
CAMPAIGN_START=$SECONDS

for i in "${!CONFIGS[@]}"; do
  name="${NAMES[$i]}"
  cfg="${CONFIGS[$i]}"
  runner="${RUNNERS[$i]}"
  run_log="${LOG_DIR}/${name}.log"

  # Match ONLY the exact run-dir suffix `_YYYYMMDD_HHMMSS`. A loose prefix
  # check can skip a different experiment whose name merely starts the same way.
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] \
     && compgen -G "artifacts/experiments/${name}_20[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" > /dev/null; then
    log ""
    log "SKIP  ${name}  (artifacts already exist)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  ${name}   [$((COMPLETED + FAILED + SKIPPED + 1))/${#CONFIGS[@]}]"
  log "  config: ${cfg}"
  log "  runner: ${runner}"
  log "  detail: ${run_log}"
  log "----------------------------------------------------------"

  START=$SECONDS
  if [[ "$runner" == "snapshot" ]]; then
    if "${SNAPSHOT[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
      STATUS="OK"
      COMPLETED=$((COMPLETED + 1))
    else
      STATUS="FAILED"
      FAILED=$((FAILED + 1))
      log "  !! failed -- last 15 lines:"
      tail -15 "$run_log" | sed 's/^/     /' | tee -a "$LOG"
    fi
  else
    if "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
      STATUS="OK"
      COMPLETED=$((COMPLETED + 1))
    else
      STATUS="FAILED"
      FAILED=$((FAILED + 1))
      log "  !! failed -- last 15 lines:"
      tail -15 "$run_log" | sed 's/^/     /' | tee -a "$LOG"
    fi
  fi
  ELAPSED=$((SECONDS - START))

  printf 'END   %s  %-7s (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) "$name" \
    | tee -a "$LOG"
done

TOTAL=$((SECONDS - CAMPAIGN_START))
log ""
log "=========================================================="
log " Finished $(date)"
log " ${COMPLETED} ok, ${FAILED} failed, ${SKIPPED} skipped, in $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m"
log "=========================================================="
log ""
log "Read the pooled runs from their snapshot_cv_metrics.csv and"
log "snapshot_holdout_metrics.csv files, not from the pooled betting row."
log ""
log "Survey with, in experiments/notebooks/survey_experiments.ipynb:"
log "  SOURCES = [\"experiments/intermediate_line_2026_08\"]"
log ""
log "Primary read order:"
log "  1. pooled_line_error vs pooled_total_points by per-snapshot cv_roi"
log "  2. pooled_line_error at snapshot 720 vs t720_control_line_error cv_roi"
log "  3. holdout ROI only as a sanity check, because it is a smaller slice"

exit $FAILED
