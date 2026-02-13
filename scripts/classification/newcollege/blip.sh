#!/usr/bin/env bash
set -e

# 你现在在 /d/raw/scripts/classfication/newcollege
# 往上 3 层回到 /d/raw （repo root）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights

python -u run.py blip \
  --data_dir "D:/your_dataset/all" \
  --batch_size 32 \
  --epochs 10 \
  --lr 3e-4 \
  --limit 0 \
  --save_model "weights/newcollege_blip_best.pth" \
  2>&1 | tee "logs/newcollege_blip.log"
