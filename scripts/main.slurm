#!/usr/bin/env bash
#SBATCH -A IscrC_CardioAI
#SBATCH -p boost_usr_prod   # dc_usr_prod
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=agent-main
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

set -euo pipefail

SIF="$WORK/containers/postgres_16.sif"

PGDATA="${PGDATA:-$WORK/pg/pgdata16}"
PGSOCK="${PGSOCK:-$TMPDIR/pgsock}"

mkdir -p "$PGDATA" "$PGSOCK"


if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
  singularity exec -B "$WORK" -B "$TMPDIR" "$SIF" \
    initdb -D "$PGDATA" --encoding=UTF8 --locale=C
  {
    echo "listen_addresses = ''"
    echo "port = $PGPORT"
  } >> "$PGDATA/postgresql.conf"
  echo "local   all   all   trust" > "$PGDATA/pg_hba.conf"
fi

cleanup() {
  singularity exec -B "$WORK" -B "$TMPDIR" "$SIF" \
    pg_ctl -D "$PGDATA" stop || true
}
trap cleanup EXIT

singularity exec -B "$WORK" -B "$TMPDIR" "$SIF" \
  pg_ctl -D "$PGDATA" \
  -l "$TMPDIR/postgres.log" \
  -o "-c unix_socket_directories='$PGSOCK' -c listen_addresses='' -c port=$PGPORT" \
  start

singularity exec -B "$WORK" -B "$TMPDIR" "$SIF" \
  pg_isready -h "$PGSOCK" -p "$PGPORT" -t 60

singularity exec -B "$WORK" -B "$TMPDIR" "$SIF" \
  createdb -h "$PGSOCK" -p "$PGPORT" "$DB_NAME" || true

echo "PostgreSQL active."
echo "Log: $TMPDIR/postgres.log"
echo "Client env: PGHOST=$PGSOCK PGPORT=$PGPORT PGDATABASE=$DB_NAME"

PGHOST="$PGSOCK" PGPORT="$PGPORT" PGDATABASE=$DB_NAME psql -tc "SELECT version();"

export PGHOST="$PGSOCK"
export PGPORT=5432
export PGUSER="$DB_USER"
export PGDATABASE="$DB_NAME"

export DB_CONNECTION_STRING="postgresql+psycopg:///${DB_NAME}?host=${PGSOCK}&port=${DB_PORT}&user=${DB_USER}"
export POSTGRES_ADMIN_DSN="postgresql:///${DB_NAME}?host=${PGSOCK}&port=${DB_PORT}&user=${DB_USER}"

PROJ_DIR="$PWD"
ENV_PATH="${CINECA_SCRATCH:-$WORK}/agentic-rag"

# activate virtual environment
source "$ENV_PATH/bin/activate"

# load env file
[[ -f .env.leonardo ]] &&  { set -a; source .env.leonardo; set +a; }

: "${APP_CONFIG_PATH:?[ERR] APP_CONFIG_PATH not defined in env file}"
[[ -f "$APP_CONFIG_PATH" ]] || { echo "[ERR] config not found: $APP_CONFIG_PATH"; exit 3; }

USE_QDRANT=false
if jq -e '.cardiology_protocols
          | .indexing? // empty
          | .type? == "qdrant"' "$APP_CONFIG_PATH" >/dev/null; then
  USE_QDRANT=true
fi
echo "[INFO] use_qdrant = $USE_QDRANT"

QPID=""
if [[ "$USE_QDRANT" == "true" ]]; then
  : "${INDEX_ROOT:?ERROR: INDEX_ROOT not defined in env file}"
  INDEX_ROOT="${INDEX_ROOT%/}"

  PROJ_DIR="$PWD"
  QDRANT_IMG="$WORK/containers/qdrant.sif"
  QDRANT_STORE="${INDEX_ROOT}/qdrant_storage"
  QDRANT_CONF="${INDEX_ROOT}/qdrant_config"

  mkdir -p "$QDRANT_STORE" "$QDRANT_CONF"

  Q_HTTP=6333; Q_GRPC=6334

  export SINGULARITYENV_QDRANT__SERVICE__HTTP_PORT="$Q_HTTP"
  export SINGULARITYENV_QDRANT__SERVICE__GRPC_PORT="$Q_GRPC"
  export SINGULARITYENV_QDRANT__LOG_LEVEL=INFO
  export SINGULARITYENV_QDRANT__SERVICE__ENABLE_CORS=true
  export SINGULARITYENV_QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
  export SINGULARITYENV_QDRANT_INIT_FILE_PATH=/qdrant/storage/.qdrant-initialized

  singularity exec \
  --pwd /qdrant/storage \
  --bind "${QDRANT_STORE}:/qdrant/storage","${QDRANT_CONF}:/qdrant/config" \
  "$QDRANT_IMG" \
  /qdrant/qdrant &

  QPID=$!
  trap 'kill $QPID 2>/dev/null || true; wait $QPID 2>/dev/null || true' EXIT

  # Healthcheck Qdrant
  for i in {1..12}; do
    if curl -fsS "http://localhost:${Q_HTTP}/readyz" >/dev/null 2>&1; then
      echo "[OK] Qdrant READY"; break
    fi
    sleep 5
    kill -0 "$QPID" 2>/dev/null || { echo "[ERR] Qdrant terminato in avvio"; exit 3; }
    [[ $i -eq 12 ]] && { echo "[ERR] Qdrant non risponde a /readyz"; exit 4; }
  done

  export QDRANT_URL="http://localhost:${Q_HTTP}"
  export FASTEMBED_CACHE_PATH_ORIGINAL="$WORK/fastembed_cache"

else
  echo "[INFO] Using FAISS vectorstore."
fi

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export FASTEMBED_CACHE_PATH="$HUGGINGFACE_HUB_CACHE"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -d "$FASTEMBED_CACHE_PATH_ORIGINAL/models--Qdrant--bm25" ] && [ ! -e "$HUGGINGFACE_HUB_CACHE/models--Qdrant--bm25" ]; then
  ln -s "$FASTEMBED_CACHE_PATH_ORIGINAL/models--Qdrant--bm25" "$HUGGINGFACE_HUB_CACHE/models--Qdrant--bm25"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import os
from huggingface_hub import snapshot_download
cache = os.environ["HUGGINGFACE_HUB_CACHE"]
p = snapshot_download("Qdrant/bm25", cache_dir=cache, local_files_only=True, allow_patterns="*")
PY

# check GPU
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(),
      "GPUs:", torch.cuda.device_count(),
      "Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m src.agentic_rag.main
