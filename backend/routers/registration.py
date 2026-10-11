from pathlib import Path
import math

from fastapi import APIRouter, Form

from models import RegisterResponse
from utils import DATA_DIR, generate_short_id
from log_decorator import log_handler


router = APIRouter(tags=["Registration"])


def user_dir(user_id: str) -> Path:
    return DATA_DIR / user_id


@router.post("/register", response_model=RegisterResponse)
@log_handler
async def register(
    name: str = Form("Anonymous", description="User's name, defaults to 'Anonymous'"),
    user_id: str | None = Form(None, description="Optional user ID; if not provided, a random 12-character ID will be generated and returned. If ID is already used, the data will be overwritten."),
) -> RegisterResponse:
    """Register a new user."""
    person_id = user_id if user_id else generate_short_id()
    person_dir = user_dir(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    (person_dir / "name.txt").write_text(name)

    return RegisterResponse(user_id=person_id)


@router.get("/users", response_model=dict[str, str])
@log_handler
async def get_users() -> dict[str, str]:
    """Get all registered users in descending order of time of last use"""

    if not DATA_DIR.exists():
        return {}

    id_name_last_time = []
    for d in DATA_DIR.iterdir():
        if (d / "name.txt").exists():
            user_id = d.name
            name = (d / "name.txt").read_text()
            last_video_time = (d / 'video').stat().st_mtime if (d / 'video').exists() else math.inf
            last_csv_time = (d / 'csv').stat().st_mtime if (d / 'csv').exists() else math.inf
            last_time = max(last_video_time, last_csv_time)

            id_name_last_time.append({
                "user_id": user_id,
                "name": name,
                "last_time": last_time,
            })

    id_name_last_time.sort(key=lambda x: x['last_time'], reverse=True)

    return {user['user_id']: user['name'] for user in id_name_last_time}
