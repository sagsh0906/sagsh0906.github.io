# Mat-MCP tool server image (CPU tools + MatGL/CHGNet; add CUDA base for GPU inference).
#   docker build -f deploy/Dockerfile -t mat-mcp .
#   docker run -p 8000:8000 -e MP_API_KEY=... -v $PWD/data/artifacts:/artifacts mat-mcp
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    MATBRAIN_ARTIFACT_DIR=/artifacts \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY matbrain ./matbrain
# CPU wheels for torch keep the image small; switch the index URL for CUDA builds.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install ".[agent,data,tools]"

EXPOSE 8000
CMD ["python", "-m", "matbrain.mcp.server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
