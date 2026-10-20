# Лайвметр

Принцип работы:
1. Сервер генерирует случайную траекторию
2. Пользователю демонстрируется точка, которая движется по этой траектории
3. Движения глаз пользователя записываются с помощью фронтальной камеры
4. Сервер проверяет соответствие реакции пользователя стимулу. Возвращает результат (bool), конформный скор [0..1] и данные для отрисовки графиков.
5. На всякий случай сохраняется исходное видео. Предусмотрена регистрация пользователей, чтобы данные сохранялись в разные папки.

Запуск на сервере (Docker; для GPU — драйвер NVIDIA не ниже 580 и nvidia-container-toolkit):
1. `cp .env.example .env`, вписать домен (DNS должен указывать на сервер). Без NVIDIA убрать `compose.gpu.yml` из `COMPOSE_FILE`.
2. `docker compose up -d --build`

Caddy сам получит сертификат. Пересобрать один сервис: `docker compose up -d --build --no-deps backend` (или `frontend`). Видео и логи остаются на хосте в `backend/data` и `backend/logs`.

Локально без Docker: `cd backend && uv sync --extra cpu && uv run fastapi dev main.py`, в другом терминале `cd frontend && npm ci && npm run dev` — Vite проксирует `/api` на бекенд (адрес в `frontend/.env.example`).

Переменные окружения бекенда:
- `TORCH_DEVICE` — устройство для UniGaze, по умолчанию `cuda`, если доступна, иначе `cpu`;
- `UNIGAZE_FP16=0` — отключить fp16 на CUDA;
- `MEDIAPIPE_DEVICE` — `cpu`, `gpu` или `auto` (по умолчанию: GPU только при наличии CUDA);
- `FACE_LANDMARKER_MODEL` — путь к `face_landmarker.task`. В Docker модель уже в образе, без Docker она скачивается в `~/.cache/unified-liveness` при первом запуске.
- `WEB_CONCURRENCY` — число процессов uvicorn. В compose стоит `2`: на 4 vCPU тестового сервера третий и четвёртый уже делят ядра на декодировании видео.

Замеры: `bench_gaze.py` (пайплайн по видео) и `bench_api.py` (нагрузка на `/islive_video`).
