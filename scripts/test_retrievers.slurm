#!/usr/bin/env bash
#SBATCH -A IscrC_CardioAI
#SBATCH -p boost_usr_prod        # or dc_usr_prod if no GPU needed
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1             # REMOVE if CPU only
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test-retriever
#SBATCH --output=retriever_logs/%x.out
#SBATCH --error=retriever_logs/%x.err
#SBATCH --mail-user=your@email.com #ADD your email or remove this line
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail


#Setup project paths

PROJ_DIR="$WORK/agentic-rag"
ENV_DIR="$CINECA_SCRATCH/agentic-rag"

mkdir -p "$WORK/logs"

cd "$PROJ_DIR" || { echo "[ERR] Cannot access $PROJ_DIR"; exit 1; }
echo "[INFO] Running from: $(pwd)"


# Activate the Python environment

source "$ENV_DIR/bin/activate"

# Load environment variables
if [[ -f "$PROJ_DIR/.env.leonardo" ]]; then
  set -a
  source "$PROJ_DIR/.env.leonardo"
  set +a
else
  echo "[WARN] .env.leonardo not found in $PROJ_DIR"
fi

# Enable offline mode for Hugging Face / Transformers
export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"

# Ensure the package is importable
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"


# Run the retriever test
python -m agentic_rag.test_retrievers