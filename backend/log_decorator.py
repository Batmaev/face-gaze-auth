import asyncio
import json
import time
import traceback
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import aiofiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Контекстная переменная для хранения текущего Request
_current_request: ContextVar[Request | None] = ContextVar("current_request", default=None)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Лёгкий middleware, сохраняющий Request в contextvars."""
    async def dispatch(self, request: Request, call_next):
        _current_request.set(request)
        return await call_next(request)


def get_current_request() -> Request | None:
    """Получить текущий Request из контекста."""
    return _current_request.get()


def _serialize_arg(key: str, value: Any) -> Any:
    """Сериализует аргумент для логирования."""
    # Duck typing для UploadFile (starlette и fastapi)
    if hasattr(value, "filename") and hasattr(value, "size") and hasattr(value, "read"):
        return f"<FILE: {value.filename}, size={value.size}>"
    if hasattr(value, "model_dump"):  # Pydantic model
        return value.model_dump()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return f"<{type(value).__name__}>"
    return value


def _extract_request_info(request: Request | None) -> dict | None:
    """Извлекает информацию из Request."""
    if request is None:
        return None
    return {
        "client_ip": request.client.host if request.client else "unknown",
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "headers": {k: v for k, v in request.headers.items() if k.lower() not in ["authorization", "cookie"]},
    }


def _serialize_result(value: Any) -> Any:
    """Сериализует результат для логирования."""
    if hasattr(value, "model_dump"):  # Pydantic model
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return value
    return str(value)


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (int, float, bool, type(None)))


def _format_json_compact_arrays(obj: Any, indent: str = "") -> str:
    """Форматирует JSON с компактными массивами примитивов."""
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        new_indent = indent + "  "
        for key, value in obj.items():
            key_str = json.dumps(key, ensure_ascii=False)
            value_str = _format_json_compact_arrays(value, new_indent)
            items.append(f"{new_indent}{key_str}: {value_str}")
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    elif isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        if all(_is_primitive(item) for item in obj):
            return json.dumps(obj, ensure_ascii=False)
        items = []
        new_indent = indent + "  "
        for item in obj:
            items.append(new_indent + _format_json_compact_arrays(item, new_indent))
        return "[\n" + ",\n".join(items) + "\n" + indent + "]"
    else:
        return json.dumps(obj, ensure_ascii=False)


async def _write_log(log_entry: dict):
    """Записывает лог в файл асинхронно."""
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"handler_{date_str}.log"
    log_line = _format_json_compact_arrays(log_entry)
    log_line = f"\n{'='*80}\n{log_line}\n{'='*80}\n"
    async with aiofiles.open(log_file, "a", encoding="utf-8") as f:
        await f.write(log_line)


def log_handler(func):
    """Декоратор для логирования хендлеров FastAPI."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        request = get_current_request()
        request_info = _extract_request_info(request)
        
        logged_kwargs = {k: _serialize_arg(k, v) for k, v in kwargs.items()}
        
        log_entry = {
            "timestamp": timestamp,
            "handler": func.__name__,
        }
        if request_info:
            log_entry["request"] = request_info
        log_entry["args"] = logged_kwargs
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            log_entry["result"] = _serialize_result(result)
            log_entry["elapsed_time_seconds"] = round(elapsed, 4)
            
            asyncio.create_task(_write_log(log_entry))
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            log_entry["exception"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            log_entry["elapsed_time_seconds"] = round(elapsed, 4)
            
            asyncio.create_task(_write_log(log_entry))
            raise
    
    return wrapper
