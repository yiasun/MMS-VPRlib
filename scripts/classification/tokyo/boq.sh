#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py boq \
  --data_dir "D:/raw/all" \
  --use_pretrained \
  --freeze_backbone_epochs 2 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-3 \
  --limit 20000 \
  --save_model "weights/tokyo_boq_best.pt" \
  --save_cm "cm/tokyo_boq_cm.png" \
  2>&1 | tee "logs/tokyo_boq.log"
