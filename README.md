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

`WEB_CONCURRENCY` в `compose.yml` — число процессов uvicorn. Стоит `2`: на 4 vCPU тестового сервера третий и четвёртый уже делят ядра на декодировании видео.

Замеры в `backend/bench/`: `python -m bench.gaze` (пайплайн по видео) и `python -m bench.api` (нагрузка на `/islive_video`), команды в docstring. Для сравнений бекенд читает `TORCH_DEVICE` (`cuda`/`cpu`/`mps`), `UNIGAZE_FP16=0` и `MEDIAPIPE_DEVICE` (`cpu`/`gpu`/`auto`); по умолчанию всё на CUDA, если она есть.
