#!/usr/bin/env bash
# Train Opportunity raw CLSTM classifiers for all label tracks, then evaluate each.
# Locomotion classifier is skipped if a checkpoint already exists.
# Locomotion evaluation is skipped if its metrics file already exists.
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
echo "Opportunity CLSTM train + evaluate for all label tracks"
echo "Started: $(date -Is)"
echo "============================================================"

# Wait for any existing eval job that holds the GPU.
while pgrep -f 'src.eval.opportunity.evaluate_signal_cross_diffusion' >/dev/null 2>&1; do
  echo "[wait] evaluation still running... $(date -Is)"
  sleep 120
done

for track in "${TRACKS[@]}"; do
  ckpt="checkpoints/clstm_raw_opportunity_${track}/best_model.pt"
  metrics="eval_outputs/opportunity/${track}_metrics.csv"

  # ---- train classifier ----
  if [[ -f "$ckpt" && "$track" == "locomotion" ]]; then
    echo "[skip train] $track already has $ckpt"
  else
    echo
    echo "===== Training track: $track ====="
    echo "Started: $(date -Is)"
    python3 -m src.train.trainingonopportunitydataset.train_clstm_raw_opportunity --track "$track" \
      > "logs/train_clstm_${track}.log" 2>&1
    echo "Finished train: $(date -Is)  -> $ckpt"
  fi

  # ---- evaluate diffusion with that classifier ----
  if [[ -f "$metrics" && "$track" == "locomotion" ]]; then
    echo "[skip eval] $track already has $metrics"
  else
    echo
    echo "===== Evaluating track: $track ====="
    echo "Started: $(date -Is)"
    python3 -m src.eval.opportunity.evaluate_signal_cross_diffusion --track "$track" \
      > "logs/eval_opportunity_${track}.log" 2>&1
    echo "Finished eval: $(date -Is)  -> $metrics"
  fi
done

echo
echo "============================================================"
echo "All tracks train+eval finished: $(date -Is)"
echo "============================================================"
