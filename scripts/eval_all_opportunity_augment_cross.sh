#!/usr/bin/env bash
# Evaluate CrossDiff vs Mean-fill for ALL Opportunity tracks using
# the augment-cross classifiers. Does NOT overwrite real-only metrics.
#
# Outputs:
#   eval_outputs/opportunity/{track}_augment_cross_metrics.{json,csv}
# Logs:
#   logs/eval_opportunity_{track}_augment_cross.log
#
# Usage (inside nepaliboy container, from repo root):
#   nohup bash scripts/eval_all_opportunity_augment_cross.sh \
#     > logs/eval_all_opportunity_augment_cross.log 2>&1 &
set -euo pipefail

cd /workspace/RaResearch/cogage-vae-diffusion-
mkdir -p logs eval_outputs/opportunity

TRACKS=(
  locomotion
  hl_activity
  ll_left_arm
  ll_left_arm_object
  ll_right_arm
  ll_right_arm_object
  ml_both_arms
)

echo "============================================================"
echo "Opportunity CrossDiff eval (augment-cross classifier) — all tracks"
echo "Started: $(date -Is)"
echo "============================================================"

for track in "${TRACKS[@]}"; do
  ckpt="checkpoints/clstm_raw_opportunity_${track}_augment_cross/best_model.pt"
  metrics="eval_outputs/opportunity/${track}_augment_cross_metrics.csv"

  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] $track — missing classifier: $ckpt"
    continue
  fi

  if [[ -f "$metrics" ]]; then
    echo "[skip] $track — already has $metrics"
    continue
  fi

  echo
  echo "===== Evaluating track: $track (augment_cross) ====="
  echo "Started: $(date -Is)"
  python3 -m src.eval.opportunity.evaluate_signal_cross_diffusion \
    --track "$track" --augmented-cross \
    > "logs/eval_opportunity_${track}_augment_cross.log" 2>&1
  echo "Finished eval: $(date -Is)  -> $metrics"
done

echo
echo "============================================================"
echo "All augment-cross evals finished: $(date -Is)"
echo "============================================================"
