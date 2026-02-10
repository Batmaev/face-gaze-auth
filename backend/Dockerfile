FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

# cv2 dependencies
RUN apt-get update && apt-get install -y python3-opencv && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN uv sync

COPY . .

CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py"]
