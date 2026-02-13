#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py lr \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --img_data_dir "D:/raw/all" \
  --epochs 50 \
  --batch_size 128 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --test_size 0.3 \
  --seed 2025 \
  --image_size 32 \
  --hidden_dim 256 \
  --device auto \
  --limit 0 \
  --model_out "weights/kingscollege_lr_model.pth" \
  --save_cm "cm/kingscollege_lr_cm.png" \
  2>&1 | tee "logs/kingscollege_lr.log"
