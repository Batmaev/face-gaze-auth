FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

# cv2 dependencies
RUN apt-get update && apt-get install python3-opencv -y

COPY pyproject.toml .

RUN uv sync

# Download weigths
RUN echo 'from deepface import DeepFace; from deepface.modules.modeling import build_model; build_model("facial_recognition", "ArcFace")' | uv run -

COPY . .

CMD ["uv", "run", "fastapi", "run", "app.py"]
