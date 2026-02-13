#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs cm

python -u run.py mlp \
  --text_data_path "Final Dataset-Texts.xlsx" \
  --text_col "List of Store Names" \
  --img_data_dir "D:/raw/all" \
  --image_size 32 \
  --pca_components 128 \
  --pca_whiten \
  --test_size 0.2 \
  --seed 2025 \
  --bert_name "bert-base-chinese" \
  --device auto \
  --scale \
  --hidden "512,256" \
  --activation relu \
  --alpha 0.0005 \
  --lr_init 0.001 \
  --solver adam \
  --batch_size 256 \
  --max_iter 400 \
  --early_stopping \
  --tol 1e-4 \
  --limit 0 \
  --save_cm "cm/newcollege_mlp_cm.png" \
  2>&1 | tee "logs/newcollege_mlp.log"
