import secrets
import string
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Body
from pydantic import BaseModel, Field
import pandas as pd
from deepface import DeepFace
from cachetools import TTLCache
import asyncio
from concurrent.futures import ThreadPoolExecutor

from stimulus.generate import fourier_ios_like, random_spline
from liveness.fit import align_gaze, find_blinks
from liveness.ml_model import predict
from liveness.example_values import ExampleValues


app = FastAPI(title="Face Gaze Auth API", version="1.0.0")


DATA_DIR = Path("data")

# Store async verification results (TTL 10 minutes)
ASYNC_VERIFY_RESULTS: TTLCache = TTLCache(maxsize=1000, ttl=600)

# Thread pool for running blocking DeepFace operations
_executor = ThreadPoolExecutor(max_workers=4)


def generate_short_id(length: int = 6) -> str:
    """Generate a short alphanumeric ID."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))

def generate_name_from_time() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S_%f")




class RegisterResponse(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the registered user")

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": "abc123"}]
        }
    }

@app.post("/register", response_model=RegisterResponse, tags=["Face ID"])
async def register(
    image: UploadFile = File(..., description="Face image for registration"),
    name: str = Form("Anonymous", description="User's name, defaults to 'Anonymous'"),
    user_id: str | None = Form(None, description="Optional user ID; if not provided, a random 6-character ID will be generated and returned. If ID is already used, the data will be overwritten."),
) -> RegisterResponse:
    """Register a new user with a face image."""
    person_id = user_id if user_id else generate_short_id()

    person_dir = DATA_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename).suffix if image.filename else ".jpg"

    image_path = person_dir / f"registration_photo{ext}"
    contents = await image.read()
    image_path.write_bytes(contents)

    name_path = person_dir / "name.txt"
    name_path.write_text(name)

    return RegisterResponse(user_id=person_id)





class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int
    left_eye: list[int] | None = None
    right_eye: list[int] | None = None

class FacialAreas(BaseModel):
    img1: FacialArea
    img2: FacialArea

class VerifyResponse(BaseModel):
    verified: bool = Field(..., description="Whether the face matches the registered user")
    distance: float = Field(..., description="Distance between face embeddings")
    threshold: float = Field(..., description="Threshold for verification")
    confidence: float = Field(..., description="Confidence score (inverse of distance)")
    model: str = Field(..., description="Face recognition model used")
    detector_backend: str = Field(..., description="Face detector backend used")
    similarity_metric: str = Field(..., description="Similarity metric used")
    facial_areas: FacialAreas = Field(..., description="Detected facial areas in both images")
    time: float = Field(..., description="Processing time in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "verified": False,
                    "distance": 0.97095,
                    "threshold": 0.68,
                    "confidence": 3.87,
                    "model": "ArcFace",
                    "detector_backend": "opencv",
                    "similarity_metric": "cosine",
                    "facial_areas": {
                        "img1": {
                            "x": 216,
                            "y": 318,
                            "w": 923,
                            "h": 923,
                            "left_eye": [836, 710],
                            "right_eye": [507, 682]
                        },
                        "img2": {
                            "x": 0,
                            "y": 0,
                            "w": 767,
                            "h": 919,
                            "left_eye": None,
                            "right_eye": None
                        }
                    },
                    "time": 1.45
                }
            ]
        }
    }


class VerifyErrorResponse(BaseModel):
    verified: bool = False
    error: str = Field(..., description="Error message")

    model_config = {
        "json_schema_extra": {
            "examples": [{"verified": False, "error": "Registration photo not found"}]
        }
    }


@app.post(
    "/verify-image",
    response_model=VerifyResponse,
    responses={
        200: {"model": VerifyResponse, "description": "Verification result"},
        404: {"model": VerifyErrorResponse, "description": "Registration photo not found"},
    },
    tags=["Face ID"],
)
async def verify(
    image: UploadFile = File(..., description="Face image to verify"),
    user_id: str = Form(..., description="User ID to verify against"),
    model_name: str = Form("ArcFace", description="Face recognition model (e.g., ArcFace, VGG-Face, Facenet)"),
) -> VerifyResponse | VerifyErrorResponse:
    """Verify a face image against a registered user's face."""
    person_dir = DATA_DIR / user_id
    auth_dir = person_dir / "auth_attempts"
    auth_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename).suffix if image.filename else ".jpg"
    file_name = generate_name_from_time() + ext
    auth_photo_path = auth_dir / file_name

    contents = await image.read()
    auth_photo_path.write_bytes(contents)

    registration_path = next(person_dir.glob("registration_photo.*"), None)
    if registration_path is None:
        return VerifyErrorResponse(verified=False, error="Registration photo not found")

    return DeepFace.verify(registration_path.as_posix(), auth_photo_path.as_posix(), model_name=model_name)


class AsyncVerifyResponse(BaseModel):
    result_id: str = Field(..., description="ID to retrieve the verification result")

    model_config = {
        "json_schema_extra": {
            "examples": [{"result_id": "abc123"}]
        }
    }


class VerifyResultResponse(BaseModel):
    status: Literal["pending", "completed", "error"] = Field(..., description="Status of the verification")
    result: VerifyResponse | None = Field(None, description="Verification result if completed")
    error: str | None = Field(None, description="Error message if failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"status": "pending", "result": None, "error": None},
                {"status": "completed", "result": {"verified": True, "distance": 0.3, "threshold": 0.68, "confidence": 2.3, "model": "ArcFace", "detector_backend": "opencv", "similarity_metric": "cosine", "facial_areas": {"img1": {"x": 0, "y": 0, "w": 100, "h": 100}, "img2": {"x": 0, "y": 0, "w": 100, "h": 100}}, "time": 1.2}, "error": None},
                {"status": "error", "result": None, "error": "Registration photo not found"},
            ]
        }
    }


def _run_verification(registration_path: str, auth_photo_path: str, model_name: str) -> dict:
    """Run DeepFace verification (blocking operation)."""
    return DeepFace.verify(registration_path, auth_photo_path, model_name=model_name)


async def _process_verification_async(
    result_id: str,
    registration_path: str,
    auth_photo_path: str,
    model_name: str,
):
    """Process verification in background and store result."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            _run_verification,
            registration_path,
            auth_photo_path,
            model_name,
        )
        ASYNC_VERIFY_RESULTS[result_id] = {"status": "completed", "result": result, "error": None}
    except Exception as e:
        ASYNC_VERIFY_RESULTS[result_id] = {"status": "error", "result": None, "error": str(e)}


@app.post(
    "/verify-image-async",
    response_model=AsyncVerifyResponse,
    responses={
        200: {"model": AsyncVerifyResponse, "description": "Result ID for async verification"},
        404: {"model": VerifyErrorResponse, "description": "Registration photo not found"},
    },
    tags=["Face ID"],
)
async def verify_async(
    image: UploadFile = File(..., description="Face image to verify"),
    user_id: str = Form(..., description="User ID to verify against"),
    model_name: str = Form("ArcFace", description="Face recognition model (e.g., ArcFace, VGG-Face, Facenet)"),
) -> AsyncVerifyResponse | VerifyErrorResponse:
    """Start async face verification. Returns a result_id to retrieve the result later."""
    person_dir = DATA_DIR / user_id
    auth_dir = person_dir / "auth_attempts"
    auth_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename).suffix if image.filename else ".jpg"
    file_name = generate_name_from_time() + ext
    auth_photo_path = auth_dir / file_name

    contents = await image.read()
    auth_photo_path.write_bytes(contents)

    registration_path = next(person_dir.glob("registration_photo.*"), None)
    if registration_path is None:
        return VerifyErrorResponse(verified=False, error="Registration photo not found")

    result_id = generate_short_id()
    ASYNC_VERIFY_RESULTS[result_id] = {"status": "pending", "result": None, "error": None}

    # Start background processing
    asyncio.create_task(
        _process_verification_async(
            result_id,
            registration_path.as_posix(),
            auth_photo_path.as_posix(),
            model_name,
        )
    )

    return AsyncVerifyResponse(result_id=result_id)


@app.get(
    "/verify-result/{result_id}",
    response_model=VerifyResultResponse,
    responses={
        200: {"model": VerifyResultResponse, "description": "Verification result or status"},
        404: {"description": "Result ID not found"},
    },
    tags=["Face ID"],
)
async def get_verify_result(result_id: str) -> VerifyResultResponse:
    """Get the result of an async verification request."""
    if result_id not in ASYNC_VERIFY_RESULTS:
        raise HTTPException(status_code=404, detail="Result ID not found or expired")

    data = ASYNC_VERIFY_RESULTS[result_id]
    return VerifyResultResponse(
        status=data["status"],
        result=data["result"],
        error=data["error"],
    )


@app.get(
    "/users",
    response_model=dict[str, str],
    responses={
        200: {
            "description": "Dictionary mapping user_id to user name",
            "content": {
                "application/json": {
                    "example": {"abc123": "John Doe", "xyz789": "Jane Smith"}
                }
            },
        }
    },
    tags=["Face ID"],
)
async def get_users() -> dict[str, str]:
    """Get all registered users."""
    return {
        user_id.name: (user_id / "name.txt").read_text()
        for user_id in DATA_DIR.iterdir()
        if (user_id / "name.txt").exists()
    }





GENERATED_STIMULI = TTLCache(maxsize=1000, ttl=180)

class StimulusResponse(BaseModel):
    type: Literal["moving-dot"] = Field("moving-dot", description="Type of stimulus")
    fps: int = Field(60, description="The stimulus should be displayed at this fps")
    token: str = Field(..., description="Stimulus will be saved on the server for 3 minutes with this token")
    x: List[float]
    y: List[float]


@app.get("/stim", tags=["Liveness"])
async def generate_stimulus(
    kind: Literal["fourier_ios_like", "random_spline"] = "random_spline",
) -> StimulusResponse:
    """Generate random stimulus.
    Generated (x, y) are guaranteed to be in [0, 1]×[0, 1].
    """

    if kind == "fourier_ios_like":
        stim_x, stim_y = fourier_ios_like()
    elif kind == "random_spline":
        stim_x, stim_y = random_spline()

    token = generate_short_id()
    GENERATED_STIMULI[token] = (stim_x, stim_y)

    return StimulusResponse(type="moving-dot", x=stim_x, y=stim_y, fps=60, token=token)



class LivenessResponse(BaseModel):
    is_live: bool = Field(..., example=True)
    score: float = Field(..., example=0.85)
    gaze_fit_x: List[float]
    gaze_fit_y: List[float]
    inlier: List[bool]
    lag: int = Field(
        ..., 
        description="Estimated lag between stimulus and gaze in frames",
        example=10
    )


LIVENESS_EXAMPLE = {
    "stim_x": ExampleValues.stim_x,
    "stim_y": ExampleValues.stim_y,
    "gaze_x": ExampleValues.gaze_x,
    "gaze_y": ExampleValues.gaze_y,
    "left_eye_blink": ExampleValues.left_eye_blink,
    "right_eye_blink": ExampleValues.right_eye_blink,
}


@app.post(
    "/islive",
    response_model=LivenessResponse,
    tags=["Liveness"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "default": {"value": LIVENESS_EXAMPLE}
                    }
                }
            }
        }
    },
)
async def check_liveness(
    token: Optional[str] = Body(None, description="Token returned by GET /stim. If not provided (or expired), stim_x, stim_y will be used."),
    user_id: str = Body("anon", description="Trajectories will be saved under this user_id"),
    stim_x: Optional[List[float]] = Body(None),
    stim_y: Optional[List[float]] = Body(None),
    gaze_x: List[float] = Body(...),
    gaze_y: List[float] = Body(...),
    left_eye_blink: Optional[List[float]] = Body(None),
    right_eye_blink: Optional[List[float]] = Body(None),
) -> LivenessResponse:
    if token:
        try:
            stim_x, stim_y = GENERATED_STIMULI[token]
        except KeyError:
            pass
    if stim_x is None or stim_y is None:
        raise HTTPException(status_code=400, detail="Stimulus not found. Maybe the token is expired.")

    try:
        min_len = min(len(stim_x), len(stim_y), len(gaze_x), len(gaze_y))
        df = pd.DataFrame({
            'stim_x': stim_x[:min_len],
            'stim_y': stim_y[:min_len],
            'gaze_x': gaze_x[:min_len],
            'gaze_y': gaze_y[:min_len],
        })

        USE_BLINKS = left_eye_blink is not None and right_eye_blink is not None
        if USE_BLINKS:
            df['blink'] = find_blinks(left_eye_blink, right_eye_blink)

        align_gaze(df, kind='independent', threshold=100.0, n_iterations=1, drop_blinks=USE_BLINKS)

        score, is_live = predict(df, USE_BLINKS)
        lag = df.lag.iloc[0]

        liveness_dir = DATA_DIR / user_id / "trajectories"
        liveness_dir.mkdir(parents=True, exist_ok=True)
        df.to_feather(liveness_dir / f"{generate_name_from_time()}.feather")

        return LivenessResponse(
            is_live=is_live,
            score=score,
            gaze_fit_x=df.gaze_fit_x,
            gaze_fit_y=df.gaze_fit_y,
            lag=lag,
            inlier=df.inlier,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}

