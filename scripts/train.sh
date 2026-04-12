#!/bin/bash
# =============================================================================
# train.sh — run training only (assumes COLMAP already done)
#
# Usage:
#   bash scripts/train.sh
#   bash scripts/train.sh --resume models/checkpoints/checkpoint_010000.pkl
# =============================================================================

set -euo pipefail
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) RESUME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

RESUME_ARG=""
[[ -n "$RESUME" ]] && RESUME_ARG="--resume $RESUME"

echo "[train.sh] Starting Gaussian Splatting training…"

python -m scripts.train \
    --config     config/config.yaml \
    --colmap_dir data/colmap_output/sparse_text \
    --image_dir  data/processed \
    --output_dir models/gaussian \
    $RESUME_ARG

echo "[train.sh] Training complete ✅"
echo "[train.sh] PLY  files : models/gaussian/*.ply"
echo "[train.sh] Splat files: models/gaussian/*.splat"
echo "[train.sh] Checkpoints: models/checkpoints/*.pkl"