#!/bin/bash
# =============================================================================
# run_pipeline.sh
# MonoSplat — full pipeline: video → 3D Gaussian Splat
#
# Usage:
#   bash scripts/run_pipeline.sh --input data/raw/video.mp4
#   bash scripts/run_pipeline.sh --input data/raw/video.mp4 --fps 10 --skip_training
# =============================================================================

set -euo pipefail

INPUT=""
CONFIG="config/config.yaml"
SKIP_EXTRACTION=false
SKIP_COLMAP=false
SKIP_TRAINING=false
FPS=5           # Reduced from 10 → lighter COLMAP load on CPU
MAX_FRAMES=200  # Hard cap: keeps reconstruction tractable on CPU
USE_GPU=false   # Fix 2: CPU mode by default; pass --gpu to enable on Colab

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()    { echo -e "${GREEN}[pipeline]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[pipeline]${NC} $1"; }
log_error()   { echo -e "${RED}[pipeline]${NC} $1"; exit 1; }
log_section() { echo -e "\n${CYAN}══════════════════════════════════════${NC}"; echo -e "${CYAN} $1${NC}"; echo -e "${CYAN}══════════════════════════════════════${NC}"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)           INPUT="$2";           shift 2 ;;
        --config)          CONFIG="$2";          shift 2 ;;
        --fps)             FPS="$2";             shift 2 ;;
        --max_frames)      MAX_FRAMES="$2";      shift 2 ;;
        --skip_extraction) SKIP_EXTRACTION=true; shift   ;;
        --skip_colmap)     SKIP_COLMAP=true;     shift   ;;
        --skip_training)   SKIP_TRAINING=true;   shift   ;;
        --gpu)             USE_GPU=true;         shift   ;;
        *) log_error "Unknown argument: $1" ;;
    esac
done

[[ -z "$INPUT" ]] && log_error "Please provide --input <video.mp4 or image_folder>"

log_section "MonoSplat Pipeline"
log_info "Input       : $INPUT"
log_info "Extract FPS : $FPS  (max $MAX_FRAMES frames)"

# ─── Stage 1 — Frame extraction ───────────────────────────────────────────
log_section "Stage 1 — Frame Extraction (FFmpeg)"
if $SKIP_EXTRACTION; then
    log_warn "Skipping frame extraction."
else
    if ! command -v ffmpeg &>/dev/null; then
        log_error "FFmpeg not found.\n  Linux: sudo apt install ffmpeg\n  macOS: brew install ffmpeg\n  Windows: https://ffmpeg.org/download.html"
    fi
    python -m src.preprocessing.extract_frames \
        "$INPUT" \
        --output     data/processed \
        --fps        "$FPS" \
        --max_frames "$MAX_FRAMES"
    FRAME_COUNT=$(ls data/processed/output_*.png 2>/dev/null | wc -l)
    log_info "Extracted $FRAME_COUNT frames → data/processed/"
fi

# ─── Stage 2 — COLMAP ─────────────────────────────────────────────────────
log_section "Stage 2 — COLMAP Pose Estimation"
if $SKIP_COLMAP; then
    log_warn "Skipping COLMAP."
else
    COLMAP_GPU_FLAG=""
    $USE_GPU && COLMAP_GPU_FLAG="--gpu"
    python -m src.preprocessing.colmap_runner \
        --image_dir  data/processed \
        --output_dir data/colmap_output \
        --quality    medium \
        $COLMAP_GPU_FLAG
    log_info "COLMAP complete → data/colmap_output/sparse_text/"
fi

# ─── Stage 3 — Training ───────────────────────────────────────────────────
log_section "Stage 3 — Gaussian Splat Training"
if $SKIP_TRAINING; then
    log_warn "Skipping training."
else
    python -m scripts.train \
        --config     "$CONFIG" \
        --colmap_dir data/colmap_output/sparse_text \
        --image_dir  data/processed \
        --output_dir models/gaussian
    log_info "Training complete → models/gaussian/"
fi

# ─── Summary ──────────────────────────────────────────────────────────────
log_section "Pipeline Complete"
LATEST_PLY=$(ls -t models/gaussian/*.ply   2>/dev/null | head -1)
LATEST_SPLAT=$(ls -t models/gaussian/*.splat 2>/dev/null | head -1)
[[ -n "$LATEST_PLY"   ]] && log_info "PLY   : $LATEST_PLY"
[[ -n "$LATEST_SPLAT" ]] && log_info "Splat : $LATEST_SPLAT"
log_info "Start the viewer: bash scripts/start_dynamic.sh"