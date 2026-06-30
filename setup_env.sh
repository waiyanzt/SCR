#!/usr/bin/env bash
# ============================================================================
# SCR — Environment Setup  (① deploy environment)
# ============================================================================
# Creates the conda env "scr" (Python 3.11) and installs PyTorch, PyG, and all
# dependencies. Auto-detects CUDA. On AutoDL servers it installs onto the data
# disk (/root/autodl-tmp) to spare the system disk.
#
# Usage: bash setup_env.sh
# ============================================================================
set -euo pipefail

CONDA_ENV="scr"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.5.1"

# AutoDL data disk — used only when present
PIP_CACHE=()
if [[ -d /root/autodl-tmp ]]; then
    DATA_DISK="/root/autodl-tmp"
    echo "AutoDL data disk detected ($DATA_DISK); redirecting caches there."
    mkdir -p "$DATA_DISK/conda/envs" "$DATA_DISK/conda/pkgs" "$DATA_DISK/pip-cache"
    conda config --add envs_dirs "$DATA_DISK/conda/envs" 2>/dev/null || true
    conda config --add pkgs_dirs "$DATA_DISK/conda/pkgs" 2>/dev/null || true
    pip config set global.cache-dir "$DATA_DISK/pip-cache" 2>/dev/null || true
    PIP_CACHE=(--cache-dir "$DATA_DISK/pip-cache")
fi

echo "============================================"
echo " SCR Environment Setup  (env=$CONDA_ENV, py=$PYTHON_VERSION)"
echo "============================================"

# 1. Conda env
if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
    echo "[1/4] conda env '$CONDA_ENV' already exists."
else
    echo "[1/4] creating conda env '$CONDA_ENV'..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y -q
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "    python: $(which python)"

# 2. CUDA detection
if command -v nvidia-smi &>/dev/null; then
    ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    case "$ver" in 12.*) CUDA_VERSION="cu121" ;; 11.*) CUDA_VERSION="cu118" ;; *) CUDA_VERSION="cu121" ;; esac
else
    CUDA_VERSION="cu121"
fi
echo "[2/4] CUDA: $CUDA_VERSION"

# 3. PyTorch + PyG (need special wheel indexes; installed before requirements.txt)
echo "[3/4] installing PyTorch $PYTORCH_VERSION+$CUDA_VERSION + PyG..."
PYG_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html"
pip install "torch==$PYTORCH_VERSION+$CUDA_VERSION" \
    --index-url "https://download.pytorch.org/whl/$CUDA_VERSION" "${PIP_CACHE[@]}"
pip install torch-geometric==2.4.0 "${PIP_CACHE[@]}"
pip install torch-scatter -f "$PYG_URL" "${PIP_CACHE[@]}"
python -c "import torch; print(f'    torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# 4. Remaining deps from requirements.txt
echo "[4/4] installing remaining dependencies (requirements.txt)..."
pip install -r requirements.txt "${PIP_CACHE[@]}"

# Verify
echo ""
echo "Verifying key imports..."
python -c "
import torch_scatter;    print('  ✓ torch_scatter')
import torch_geometric;  print(f'  ✓ torch_geometric {torch_geometric.__version__}')
import faiss;            print('  ✓ faiss')
import sklearn;          print(f'  ✓ sklearn {sklearn.__version__}')
import transformers;     print(f'  ✓ transformers {transformers.__version__}')
"

echo ""
echo "Done. Activate with: conda activate $CONDA_ENV"
