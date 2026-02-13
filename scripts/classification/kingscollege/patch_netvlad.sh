#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs weights cm

python -u run.py patch_netvlad \
  --data_dir "D:/raw/all" \
  --use_pretrained \
  --use_amp \
  --epochs 20 \
  --freeze_backbone_epochs 2 \
  --vlad_K 32 \
  --proj_dim 128 \
  --grids 1 2 3 \
  --limit 0 \
  --save_model "weights/kingscollege_patch_netvlad_best.pt" \
  --save_cm "cm/kingscollege_patch_netvlad_cm.png" \
  2>&1 | tee "logs/kingscollege_patch_netvlad.log"
