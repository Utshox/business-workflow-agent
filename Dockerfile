FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/
COPY fine_tuning/ fine_tuning/
COPY eval/ eval/

# Create data directory for ChromaDB
RUN mkdir -p /app/data/chroma

EXPOSE 8000

# --- Agent API server ---
FROM base AS api
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Fine-tuning runner ---
FROM base AS fine-tune
RUN pip install --no-cache-dir ".[fine-tuning]"
CMD ["python", "-m", "fine_tuning.train_lora"]
