import secrets
import string
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from cachetools import TTLCache


DATA_DIR = Path("data")

ASYNC_VERIFY_RESULTS: TTLCache = TTLCache(maxsize=1000, ttl=600)

GENERATED_STIMULI: TTLCache = TTLCache(maxsize=1000, ttl=180)

executor = ThreadPoolExecutor(max_workers=4)


def generate_short_id(length: int = 12) -> str:
    """Generate a short alphanumeric ID."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_name_from_time() -> str:
    """Generate a filename-safe timestamp."""
    return datetime.now().strftime("%y%m%d_%H%M%S_%f")

