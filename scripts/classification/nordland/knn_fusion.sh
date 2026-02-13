#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs cm

python -u run.py knn \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_col "List of Store Names" \
  --img_data_dir "D:/raw/all" \
  --image_size 32 \
  --pca_components 128 \
  --pca_whiten \
  --test_size 0.5 \
  --seed 2025 \
  --bert_name "bert-base-chinese" \
  --device auto \
  --n_neighbors 7 \
  --weights distance \
  --metric minkowski \
  --p 2 \
  --scale \
  --limit 0 \
  --save_cm "cm/nordland_knn_cm.png" \
  2>&1 | tee "logs/nordland_knn.log"
