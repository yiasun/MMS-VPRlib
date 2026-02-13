#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights

python -u run.py clip \
  --data_dir "D:/raw/all" \
  --backbone "ViT-B/32" \
  --batch_size 64 \
  --epochs 20 \
  --lr 5e-5 \
  --unfreeze_blocks 2 \
  --limit 0 \
  --save_model "weights/newcollege_clip_vitb32.pth" \
  2>&1 | tee "logs/newcollege_clip.log"
