#!/bin/bash
# Upstream reference check (see ref_check.py): ULTRA's own run.py, zero-shot, on its own dataset copies.
#SBATCH -A eecs
#SBATCH -J ultra_ref
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /nfs/hpc/share/mousavij/gnn/new_code/INV-GNN/src/baselines/ULTRA_pilot/logs/ref_%a_%j.out
#SBATCH --array=0-1
PILOT=/nfs/hpc/share/mousavij/gnn/new_code/INV-GNN/src/baselines/ULTRA_pilot
cd "$PILOT"
source ~/hpc-share/gnn/env/bin/activate
module load gcc/12.2 cuda/12.4
export CC=$(which gcc) CXX=$(which g++) TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;9.0" TORCH_EXTENSIONS_DIR="$PILOT/torch_extensions"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
CK=(ultra_3g ultra_4g); C=${CK[$SLURM_ARRAY_TASK_ID]}
python ref_check.py -c ref_check_inference.yaml --dataset WN18RR --epochs 0 --bpe null --gpus "[0]" --ckpt "$PILOT/upstream/ckpts/$C.pth"
