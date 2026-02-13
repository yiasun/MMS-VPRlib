#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs cm

python -u run.py gnb \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_col "List of Store Names" \
  --img_data_dir "D:/raw/all" \
  --image_size 32 \
  --pca_components 128 \
  --test_size 0.3 \
  --seed 2025 \
  --bert_name "bert-base-chinese" \
  --device auto \
  --limit 20000 \
  --save_cm "cm/tokyo_gnb_cm.png" \
  2>&1 | tee "logs/tokyo_gnb.log"
