#!/usr/bin/env bash
set -euo pipefail

module purge
module load python/3.11.7
module load cuda/12.2

PROJ_DIR="$PWD"
ENV_PATH="$CINECA_SCRATCH/agentic-rag"

cd "$PROJ_DIR"

if [ -d "$ENV_PATH" ] && [ -f "$ENV_PATH/bin/activate" ]; then
  echo "[env] existing venv"
  source "$ENV_PATH/bin/activate"
else
  echo "[env] creating venv"
  python3 -m venv "$ENV_PATH"
  source "$ENV_PATH/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel build ninja packaging
pip install -e .
pip uninstall "triton==3.4.0"

# python -m pip install --no-cache-dir --force-reinstall --upgrade "cardiologygenai-coordo @ git+https://github.com/cardiology-gen-ai/cardiology-gen-ai.git@main"


# export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
# export CUDACXX="$CUDA_HOME/bin/nvcc"
# export MAX_JOBS=8
# export TORCH_CUDA_ARCH_LIST="8.0"

# python -m pip install -U flash-attn --no-build-isolation -v

export XDG_CACHE_HOME=$WORK/.cache
export FASTEMBED_CACHE_PATH=$XDG_CACHE_HOME/fastembed
mkdir -p "$FASTEMBED_CACHE_PATH"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

echo "[OK] installation completed"