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
  --limit 0 \
  --save_model "weights/newcollege_eigenplaces_best.pt" \
  --save_cm "cm/newcollege_eigenplaces_cm.png" \
  2>&1 | tee "logs/newcollege_eigenplaces.log"
