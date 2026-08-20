FROM python:3.11-slim AS base

# build-essential covers any deps (e.g. tokenizers) without a prebuilt wheel for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv venv "$VIRTUAL_ENV"

# sentence-transformers pulls in torch, and PyPI's default torch build bundles
# CUDA (~2-3GB across torch + a dozen nvidia-cu* packages) that's never used
# here — inference is CPU-only (faiss-cpu, no GPU in this image). Installing
# the CPU-only build first means later installs see torch already satisfied
# and skip pulling the GPU one.
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies. This reads pyproject.toml directly
# rather than `uv sync --locked`, so exact transitive versions may drift from
# uv.lock over time — acceptable for a CPU inference image; local dev still
# uses the full lockfile via `uv sync`.
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
