#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py salad \
  --data_dir "D:/raw/all" \
  --use_pretrained \
  --use_distill \
  --freeze_backbone_epochs 2 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-3 \
  --lambda_distill 5e-4 \
  --clip_grad_norm 5.0 \
  --limit 20000 \
  --save_model "weights/tokyo_salad_best.pt" \
  --save_cm "cm/tokyo_salad_cm.png" \
  2>&1 | tee "logs/tokyo_salad.log"
