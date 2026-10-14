FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

# keep the venv outside /app so the bind mount does not shadow it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ENV HF_HOME=/cache/huggingface

COPY pyproject.toml .

# cpu или gpu: сборка torch, см. extras в pyproject.toml
ARG TORCH=cpu

RUN uv sync --no-dev --extra "$TORCH"

ENV FACE_LANDMARKER_MODEL=/opt/models/face_landmarker.task
ADD --chmod=644 \
    --checksum=sha256:64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    "$FACE_LANDMARKER_MODEL"

# Иначе нет сертификатов для Hugging Face
# libxcb1/libgl1/libglib2.0-0: GUI-сборка OpenCV
# libgomp1: LiteRT (XNNPACK)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libxcb1 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# BuildKit cache keeps the download across image rebuilds. The copy lands in
# the image so a new huggingface_cache volume is seeded on first start.
RUN --mount=type=cache,target=/tmp/hf-cache \
    echo "import unigaze; unigaze.load('unigaze_b16_joint')" | HF_HOME=/tmp/hf-cache uv run --no-sync - \
    && mkdir -p "$HF_HOME" \
    && cp -a /tmp/hf-cache/. "$HF_HOME"/

COPY . .

CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py"]
