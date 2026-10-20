#!/bin/bash
# "Declared after training": the UNCHANGED released checkpoint on every compiled variant (compile_back.py).
# One task per dataset; checkpoint via CK (default ultra_4g). Results: results_compiled/<DATASET>/<CK>/.
#   sbatch -p share,eecs --constraint=rtx2080 --exclude=cn-gpu3 sbatch_compiled.sbatch
#SBATCH -A eecs
#SBATCH -J ultra_compiled
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -t 06:00:00
#SBATCH -o /nfs/hpc/share/mousavij/gnn/new_code/INV-GNN/src/baselines/ULTRA_pilot/logs/compiled_%a_%j.out
#SBATCH --array=0-2
set -uo pipefail
PILOT=/nfs/hpc/share/mousavij/gnn/new_code/INV-GNN/src/baselines/ULTRA_pilot
cd "$PILOT"
source ~/hpc-share/gnn/env/bin/activate
module load gcc/12.2 cuda/12.4
export CC=$(which gcc) CXX=$(which g++) TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;9.0" TORCH_EXTENSIONS_DIR="$PILOT/torch_extensions"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1
DATASETS=(WN18RR FB15k237 Hetionet); DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}; CK=${CK:-ultra_4g}
case "$DS" in
  WN18RR)   REF=no_changes; VARIANTS=(no_changes all_inverse_edges transitive_edges universal_edges) ;;
  FB15k237) REF=original; VARIANTS=(original inverse reified_cvt_strong reified_cvt_light shortcut2_c5k shortcut2_c50k shortcut2_c200k) ;;
  Hetionet) REF=original; VARIANTS=(original inverse reified_regulation shortcut2_c50k) ;;
esac
OUT="$PILOT/results_compiled/$DS/$CK"; mkdir -p "$OUT"
for V in "${VARIANTS[@]}"; do
  [ -f "$OUT/${V}__compiled/metrics.json" ] && continue
  python ultra_eval.py --dataset-dir "data/$DS" --variant "${V}__compiled" --ckpt "upstream/ckpts/$CK.pth" --split test --out "$OUT/${V}__compiled" || echo "FAILED $V"
done
python compare_variants.py --results "$OUT" --reference "${REF}__compiled"
