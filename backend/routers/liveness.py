from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
import pandas as pd

from models import StimulusResponse, LivenessResponse, LIVENESS_EXAMPLE
from utils import DATA_DIR, GENERATED_STIMULI, generate_short_id, generate_name_from_time
from stimulus.generate import fourier_ios_like, random_spline
from liveness.fit import align_gaze, find_blinks
from liveness.ml_model import predict
from log_decorator import log_handler


router = APIRouter(tags=["Liveness"])


@router.get("/stim")
@log_handler
async def generate_stimulus(
    kind: Literal["fourier_ios_like", "random_spline"] = "random_spline",
) -> StimulusResponse:
    """Generate random stimulus.
    Generated (x, y) are guaranteed to be in [0, 1]×[0, 1].
    """

    N_CTRL_POINTS = 6

    DURATION_SEC = 5

    FPS = 60
    N_POINTS = DURATION_SEC * FPS

    if kind == "fourier_ios_like":
        stim_x, stim_y = fourier_ios_like(N_POINTS)
    elif kind == "random_spline":
        stim_x, stim_y = random_spline(N_POINTS, N_CTRL_POINTS)

    token = generate_short_id()
    GENERATED_STIMULI[token] = (stim_x, stim_y)

    return StimulusResponse(type="moving-dot", x=stim_x, y=stim_y, fps=FPS, duration_sec=DURATION_SEC, token=token)


@router.post(
    "/islive",
    response_model=LivenessResponse,
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
@log_handler
async def check_liveness(
    token: Optional[str] = Body(None, description="Token returned by GET /stim. If not provided (or expired), stim_x, stim_y will be used."),
    user_id: str = Body("anon", description="Trajectories will be saved under this user_id"),
    quality: Optional[Literal["good", "bad"]] = Body(None, description="User's self-report of their cooperation"),
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

        if left_eye_blink:
            df['left_eye_blink'] = left_eye_blink[:min_len]
        if right_eye_blink:
            df['right_eye_blink'] = right_eye_blink[:min_len]

        USE_BLINKS = left_eye_blink is not None and right_eye_blink is not None
        if USE_BLINKS:
            df['blink'] = find_blinks(left_eye_blink, right_eye_blink)

        align_gaze(df, kind='independent', threshold=100.0, n_iterations=5, drop_blinks=USE_BLINKS)

        score, is_live = predict(df, USE_BLINKS)
        lag = df.lag.iloc[0]

        liveness_dir = DATA_DIR / user_id / "trajectories"
        liveness_dir.mkdir(parents=True, exist_ok=True)
        if quality:
            filename = f"{generate_name_from_time()}-{quality}.feather"
        else:
            filename = f"{generate_name_from_time()}.feather"
        df.to_feather(liveness_dir / filename)

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


@router.post("/csv")
@log_handler
async def upload_csv(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a CSV file for the given user."""
    csv_dir = DATA_DIR / user_id / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    ext = ".csv"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[1]

    filename = f"{generate_name_from_time()}{ext}"
    filepath = csv_dir / filename

    content = await file.read()
    filepath.write_bytes(content)

    return {"message": "File uploaded successfully", "path": str(filepath)}

