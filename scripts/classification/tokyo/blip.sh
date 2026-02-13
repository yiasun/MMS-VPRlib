#!/usr/bin/env bash
set -e

# 回到 repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights

python -u run.py blip \
  --data_dir "D:/raw/all" \
  --batch_size 32 \
  --epochs 10 \
  --lr 3e-4 \
  --limit 20000 \
  --save_model "weights/tokyo_blip_best.pth" \
  2>&1 | tee "logs/tokyo_blip.log"
