#!/usr/bin/env bash
# Train Opportunity C-LSTM-A with CrossDiff augmentation for ALL label tracks.
# Does NOT evaluate. Does NOT overwrite real-only classifiers.
#
# Checkpoints:
#   checkpoints/clstm_raw_opportunity_{track}_augment_cross/best_model.pt
#
# Usage (inside nepaliboy container, from repo root):
#   nohup bash scripts/train_all_opportunity_clstm_augment_cross.sh \
#     > logs/train_all_opportunity_clstm_augment_cross.log 2>&1 &
set -euo pipefail

cd /workspace/RaResearch/cogage-vae-diffusion-
mkdir -p logs

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
echo "Opportunity CLSTM augment-cross train for all label tracks"
echo "Started: $(date -Is)"
echo "============================================================"

for track in "${TRACKS[@]}"; do
  ckpt="checkpoints/clstm_raw_opportunity_${track}_augment_cross/best_model.pt"

  if [[ -f "$ckpt" ]]; then
    echo "[skip] $track already has $ckpt"
    continue
  fi

  echo
  echo "===== Training track: $track (augment_cross) ====="
  echo "Started: $(date -Is)"
  python3 -m src.train.trainingonopportunitydataset.train_clstm_raw_opportunity_augment_cross --track "$track" \
    > "logs/train_clstm_${track}_augment_cross.log" 2>&1
  echo "Finished train: $(date -Is)  -> $ckpt"
done

echo
echo "============================================================"
echo "All augment-cross trains finished: $(date -Is)"
echo "============================================================"
