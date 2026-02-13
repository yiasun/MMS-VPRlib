#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py mixvpr \
  --data_dir "D:/raw/all" \
  --use_pretrained \
  --use_amp \
  --epochs 20 \
  --freeze_backbone_epochs 2 \
  --limit 20000 \
  --save_model "weights/tokyo_mixvpr_best.pt" \
  --save_cm "cm/tokyo_mixvpr_cm.png" \
  2>&1 | tee "logs/tokyo_mixvpr.log"
