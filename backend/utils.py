import secrets
import string
from pathlib import Path
from datetime import datetime


DATA_DIR = Path("data")


def generate_short_id(length: int = 12) -> str:
    """Generate a short alphanumeric ID."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_name_from_time() -> str:
    """Generate a filename-safe timestamp."""
    return datetime.now().strftime("%y%m%d_%H%M%S_%f")

