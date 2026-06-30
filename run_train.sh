#!/usr/bin/env bash
# ============================================================================
# SCR — Training  (③ train / fine-tune SCR)
# ============================================================================
# Default config (config/train.yaml) continues from ckpts/scr_3g.pth and writes
# model_epoch_{N}.pth into output/, evaluating zero-shot on all 39 datasets each
# epoch.
#
# Usage:
#   bash run_train.sh                                         # fine-tune from scr_3g.pth
#   bash run_train.sh -g 0 -s 1024                            # GPU + seed
#   bash run_train.sh -c config/train_scratch.yaml --ckpt none      # from scratch
#   bash run_train.sh -c config/train_ultra_ft.yaml --no-freeze    # from ULTRA backbone
#
# Options: -g GPU  -c CONFIG  -s SEED  --ckpt PATH  --no-freeze
# ============================================================================
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scr
cd "$(dirname "$0")"

CONFIG="config/train.yaml"
GPU=0
SEED=1024
CKPT=()
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g)         GPU="$2"; shift 2 ;;
        -c)         CONFIG="$2"; shift 2 ;;
        -s)         SEED="$2"; shift 2 ;;
        --ckpt)     CKPT=(--ckpt "$2"); shift 2 ;;
        --no-freeze) EXTRA+=(--no-freeze); shift ;;
        -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
        *)          echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "============================================"
echo " SCR Training — GPU $GPU  Seed $SEED"
echo " Config: $CONFIG"
echo "============================================"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p output results

START=$(date +%s)
python script/train.py -c "$CONFIG" -g "$GPU" -s "$SEED" "${CKPT[@]}" "${EXTRA[@]}"
ELAPSED=$(( $(date +%s) - START ))

echo ""
echo "Training completed in $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Saved checkpoints:"
find output -name "model_epoch_*.pth" -exec ls -lh {} \; 2>/dev/null || echo "  (none found)"
