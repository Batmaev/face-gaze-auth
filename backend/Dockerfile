FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

COPY pyproject.toml .

RUN uv sync --no-dev

COPY . .

CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py"]
