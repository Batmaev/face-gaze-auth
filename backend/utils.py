import secrets
import string
from pathlib import Path


DATA_DIR = Path("data")

USER_ID_LENGTH = 8
# id становится именем каталога в DATA_DIR, поэтому только ASCII-буквы и цифры.
USER_ID_PATTERN = r"^[A-Za-z0-9]+$"
USER_ID_MAX_LENGTH = 64


def generate_short_id(length: int = USER_ID_LENGTH) -> str:
    """Случайный id пользователя."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))

