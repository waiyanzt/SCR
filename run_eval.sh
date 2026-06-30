#!/usr/bin/env bash
# ============================================================================
# SCR — Reproduce Results  (② zero-shot evaluation on scr_3g.pth)
# ============================================================================
# Evaluates all 39 datasets with the released ckpts/scr_3g.pth checkpoint
# (config/eval_all.yaml). k/p are auto-selected per dataset.
#
# Usage:
#   bash run_eval.sh                                # GPU 0, default checkpoint
#   bash run_eval.sh -g 0
#   bash run_eval.sh --ckpt output/model_epoch_8.pth -g 0   # eval a trained ckpt
# ============================================================================
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scr
cd "$(dirname "$0")"

GPU=0
CKPT=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g)       GPU="$2"; shift 2 ;;
        --ckpt)   CKPT=(--ckpt "$2"); shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *)        echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

CKPT_NAME="${CKPT[1]:-ckpts/scr_3g.pth}"

echo "============================================"
echo " SCR Zero-shot Evaluation — GPU $GPU"
echo " Checkpoint: $CKPT_NAME"
echo "============================================"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p output results

START=$(date +%s)
python script/eval.py -c config/eval_all.yaml -g "$GPU" "${CKPT[@]}"
ELAPSED=$(( $(date +%s) - START ))

echo ""
echo "Completed in $((ELAPSED / 60))m $((ELAPSED % 60))s"

# Print results summary from the freshest log
LOG=$(find output -name "log.txt" -type f -newer config/eval_all.yaml 2>/dev/null | sort | tail -1)
if [[ -n "$LOG" ]]; then
    echo ""
    echo "--- KG Inductive (MRR) ---"
    grep -E "FB|WN|NELL|FBIngram|WKIngram|NLIngram" "$LOG" | grep -v dataset | grep "mrr" || true
    echo ""
    echo "--- Node Classification ---"
    grep "NF_" "$LOG" | grep "Acc" || true
    echo ""
    echo "--- Graph Classification ---"
    grep "GF_" "$LOG" | grep "Acc" || true
fi
