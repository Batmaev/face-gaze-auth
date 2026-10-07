FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

COPY pyproject.toml .

RUN uv sync --no-dev

COPY . .

CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py"]
