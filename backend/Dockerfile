FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

# keep the venv outside /app so the bind mount does not shadow it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ENV HF_HOME=/cache/huggingface

COPY pyproject.toml .

RUN uv sync --no-dev

# Колесо webeyetrack на PyPI собрано без model_weights; берём модель из репозитория
ADD --chmod=644 https://raw.githubusercontent.com/RedForestAI/WebEyeTrack/75fbd2f5f784f2eb3a39675a8dcbf1b01c697f1c/python/webeyetrack/model_weights/face_landmarker_v2_with_blendshapes.task \
    /opt/venv/lib/python3.13/site-packages/webeyetrack/model_weights/

# Иначе нет сертификатов для Hugging Face
# libxcb1/libgl1/libglib2.0-0: GUI-сборка OpenCV (тянется через mediapipe)
# libegl1/libgles2: mediapipe tasks грузит их даже с CPU-делегатом
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libxcb1 libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# BuildKit cache keeps the download across image rebuilds. The copy lands in
# the image so a new huggingface_cache volume is seeded on first start.
RUN --mount=type=cache,target=/tmp/hf-cache \
    echo "import unigaze; unigaze.load('unigaze_b16_joint')" | HF_HOME=/tmp/hf-cache uv run --no-sync - \
    && mkdir -p "$HF_HOME" \
    && cp -a /tmp/hf-cache/. "$HF_HOME"/

COPY . .

CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py"]
