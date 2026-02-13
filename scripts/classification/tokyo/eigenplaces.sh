#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py eigenplaces \
  --data_dir "D:/raw/all" \
  --use_pretrained \
  --use_amp \
  --epochs 20 \
  --lambda_ortho 0.001 \
  --freeze_backbone_epochs 2 \
  --limit 20000 \
  --save_model "weights/tokyo_eigenplaces_best.pt" \
  --save_cm "cm/tokyo_eigenplaces_cm.png" \
  2>&1 | tee "logs/tokyo_eigenplaces.log"
