#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights

python -u run.py gcn \
  --data_dir "D:/raw/all" \
  --image_size 32 \
  --knn_k 10 \
  --knn_metric cosine \
  --test_size 0.2 \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_col "List of Store Names" \
  --row_index 0 \
  --bert_name "bert-base-chinese" \
  --hidden_dim 128 \
  --epochs 200 \
  --lr 1e-3 \
  --weight_decay 5e-4 \
  --seed 42 \
  --limit 0 \
  --save_model "weights/pitt_gcn_textplugin.pth" \
  2>&1 | tee "logs/pitt_gcn_textplugin.log"
