#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights

python -u run.py blip \
  --data_dir "D:/raw/all" \
  --batch_size 32 \
  --epochs 10 \
  --lr 3e-4 \
  --limit 0 \
  --save_model "weights/pitt_blip_best.pth" \
  2>&1 | tee "logs/pitt_blip.log"
