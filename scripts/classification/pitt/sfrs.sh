#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py sfrs \
  --data_dir "D:/raw/all" \
  --use_pretrained \
  --use_amp \
  --epochs 20 \
  --freeze_backbone_epochs 2 \
  --limit 0 \
  --save_model "weights/pitt_sfrs_best.pt" \
  --save_cm "cm/pitt_sfrs_cm.png" \
  2>&1 | tee "logs/pitt_sfrs.log"
