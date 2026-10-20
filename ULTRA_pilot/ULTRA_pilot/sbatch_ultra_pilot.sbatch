#!/bin/bash
# ULTRA representation-sensitivity pilot: zero-shot evaluation of the released checkpoints on every variant.
# One array task = one (dataset, checkpoint) pair; inside it every variant is evaluated, then compared.
# A second run of the reference variant ("<ref>__repeat") measures the hardware noise floor.
#
#   sbatch sbatch_ultra_pilot.sbatch                       # full pilot, 9 tasks (3 datasets x 3 checkpoints)
#   LIMIT=64 sbatch --array=0 -t 00:30:00 sbatch_ultra_pilot.sbatch   # smoke test: 64 queries, 1 task
#
# Writes only under results/ (ranks + 51 candidate scores per query; no full score matrices).
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -J ultra_pilot
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -t 12:00:00
#SBATCH --nice=50
#SBATCH -o /nfs/hpc/share/mousavij/gnn/new_code/INV-GNN/src/baselines/ULTRA_pilot/logs/pilot_%a_%j.out
#SBATCH --array=0-8

set -uo pipefail
PILOT=/nfs/hpc/share/mousavij/gnn/new_code/INV-GNN/src/baselines/ULTRA_pilot
cd "$PILOT"
source ~/hpc-share/gnn/env/bin/activate
module load gcc/12.2 cuda/12.4                     # C++17 compiler + nvcc for ULTRA's rspmm kernel
export CC=$(which gcc) CXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;9.0"       # one build for V100 / RTX 2080 / A100 / H100
export TORCH_EXTENSIONS_DIR="$PILOT/torch_extensions"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASETS=(WN18RR FB15k237 Hetionet)
CKPTS=(ultra_3g ultra_4g ultra_50g)
TASK=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$((TASK / 3))]}
CK=${CKPTS[$((TASK % 3))]}
case "$DS" in
  WN18RR)   VARIANTS=(no_changes all_inverse_edges transitive_edges universal_edges); REF=no_changes ;;
  FB15k237) VARIANTS=(original inverse reified_cvt_strong reified_cvt_light shortcut2_c5k shortcut2_c50k shortcut2_c200k); REF=original ;;
  Hetionet) VARIANTS=(original inverse reified_regulation shortcut2_c50k); REF=original ;;   # held out from 3g/4g pretraining
esac
LIMIT=${LIMIT:-0}
SUFFIX=""; [ "$LIMIT" != "0" ] && SUFFIX="_smoke$LIMIT"
OUT="$PILOT/results$SUFFIX/$DS/$CK"
mkdir -p "$OUT"
echo "task $TASK: $DS / $CK on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"

for V in "${VARIANTS[@]}" "${REF}__repeat"; do
  REAL=${V%__repeat}
  if [ -f "$OUT/$V/metrics.json" ]; then echo "skip $V (done)"; continue; fi
  python ultra_eval.py --dataset-dir "data/$DS" --variant "$REAL" --ckpt "upstream/ckpts/$CK.pth" \
      --split test --out "$OUT/$V" --limit "$LIMIT" || echo "FAILED $V"
done
python compare_variants.py --results "$OUT" --reference "$REF"
