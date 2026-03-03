ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

USER root
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e ".[prompting]"
# RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# ENV HF_HOME=/models
# ENV HF_HUB_CACHE=/models
# RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('zeroentropy/zerank-2', trust_remote_code=True);  \
#    CrossEncoder('jinaai/jina-reranker-v3', trust_remote_code=True);  \
#    CrossEncoder('nvidia/llama-nemotron-rerank-1b-v2', trust_remote_code=True);"

COPY src/ /app/src
# COPY configs/ /app/configs  # TODO: uncomment in production

RUN useradd -m user
USER user