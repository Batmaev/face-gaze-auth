from pathlib import Path
import asyncio
from functools import partial

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from deepface import DeepFace

from models import RegisterResponse, VerifyResponse, AsyncVerifyResponse, VerifyResultResponse
from utils import DATA_DIR, ASYNC_VERIFY_RESULTS, executor, generate_short_id, generate_name_from_time


router = APIRouter(tags=["Face ID"])


def user_dir(user_id: str) -> Path:
    return DATA_DIR / user_id


def get_registration_photo(user_id: str) -> Path:
    """Get registration photo path or raise 404."""
    path = next(user_dir(user_id).glob("registration_photo.*"), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Registration photo not found")
    return path


async def save_auth_photo(image: UploadFile, user_id: str) -> Path:
    """Save auth attempt photo and return its path."""
    auth_dir = user_dir(user_id) / "auth_attempts"
    auth_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename).suffix if image.filename else ".jpg"
    path = auth_dir / (generate_name_from_time() + ext)
    path.write_bytes(await image.read())
    return path


@router.post("/register", response_model=RegisterResponse)
async def register(
    image: UploadFile = File(..., description="Face image for registration"),
    name: str = Form("Anonymous", description="User's name, defaults to 'Anonymous'"),
    user_id: str | None = Form(None, description="Optional user ID; if not provided, a random 6-character ID will be generated and returned. If ID is already used, the data will be overwritten."),
) -> RegisterResponse:
    """Register a new user with a face image."""
    person_id = user_id if user_id else generate_short_id()
    person_dir = user_dir(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename).suffix if image.filename else ".jpg"
    (person_dir / f"registration_photo{ext}").write_bytes(await image.read())
    (person_dir / "name.txt").write_text(name)

    return RegisterResponse(user_id=person_id)


@router.post("/verify-image", response_model=VerifyResponse)
async def verify(
    image: UploadFile = File(..., description="Face image to verify"),
    user_id: str = Form(..., description="User ID to verify against"),
    model_name: str = Form("Facenet512", description="One of: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet"),
) -> VerifyResponse:
    """Verify a face image against a registered user's face."""
    registration_path = get_registration_photo(user_id)
    auth_photo_path = await save_auth_photo(image, user_id)
    try:
        return DeepFace.verify(registration_path.as_posix(), auth_photo_path.as_posix(), model_name=model_name, align=False)
    except ValueError as e:
        which_image = "auth image" if "img2" in str(e) else "registration image"
        raise HTTPException(status_code=400, detail=f"No face detected in {which_image}")


async def _process_verification(result_id: str, registration_path: str, auth_photo_path: str, model_name: str):
    """Process verification in background and store result."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            partial(DeepFace.verify, registration_path, auth_photo_path, model_name=model_name, align=False),
        )
        ASYNC_VERIFY_RESULTS[result_id] = {"status": "completed", "result": result, "error": None}
    except ValueError as e:
        which_image = "auth image" if "img2" in str(e) else "registration image"
        ASYNC_VERIFY_RESULTS[result_id] = {"status": "error", "result": None, "error": f"No face detected in {which_image}"}
    except Exception as e:
        ASYNC_VERIFY_RESULTS[result_id] = {"status": "error", "result": None, "error": str(e)}


@router.post("/verify-image-async", response_model=AsyncVerifyResponse)
async def verify_async(
    image: UploadFile = File(..., description="Face image to verify"),
    user_id: str = Form(..., description="User ID to verify against"),
    model_name: str = Form("Facenet512", description="One of: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet"),
) -> AsyncVerifyResponse:
    """Start async face verification. Returns a result_id to retrieve the result later."""
    registration_path = get_registration_photo(user_id)
    auth_photo_path = await save_auth_photo(image, user_id)

    result_id = generate_short_id()
    ASYNC_VERIFY_RESULTS[result_id] = {"status": "pending", "result": None, "error": None}

    asyncio.create_task(
        _process_verification(result_id, registration_path.as_posix(), auth_photo_path.as_posix(), model_name)
    )
    return AsyncVerifyResponse(result_id=result_id)


@router.get("/verify-result/{result_id}", response_model=VerifyResultResponse)
async def get_verify_result(result_id: str) -> VerifyResultResponse:
    """Get the result of an async verification request."""
    if result_id not in ASYNC_VERIFY_RESULTS:
        raise HTTPException(status_code=404, detail="Result ID not found or expired")

    data = ASYNC_VERIFY_RESULTS[result_id]
    return VerifyResultResponse(status=data["status"], result=data["result"], error=data["error"])


@router.get("/users", response_model=dict[str, str])
async def get_users() -> dict[str, str]:
    """Get all registered users."""
    return {
        d.name: (d / "name.txt").read_text()
        for d in DATA_DIR.iterdir()
        if (d / "name.txt").exists()
    }
