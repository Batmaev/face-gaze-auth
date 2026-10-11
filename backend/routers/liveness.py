import json

from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
import numpy as np
import pandas as pd
import unigaze

from models import StimulusResponse, LivenessResponse
from utils import DATA_DIR, GENERATED_STIMULI, generate_short_id, generate_name_from_time
from stimulus.generate import random_spline
from trajectory_comparison.fit import find_blinks
from trajectory_comparison.decide import align_and_compare_trajectories
from unigaze_helpers.trajectory import unigaze_gaze_from_video
from log_decorator import log_handler


unigaze_model = unigaze.load("unigaze_b16_joint", device="cpu")

STIM_FPS = 60

router = APIRouter(tags=["Liveness"])

@router.get("/stim")
@log_handler
async def generate_stimulus() -> StimulusResponse:
    """Generate random stimulus.
    Generated (x, y) are guaranteed to be in [0, 1]×[0, 1].
    """

    N_CTRL_POINTS = 6

    DURATION_SEC = 5

    N_FRAMES = DURATION_SEC * STIM_FPS

    stim_x, stim_y = random_spline(N_FRAMES, N_CTRL_POINTS)

    token = generate_short_id()
    GENERATED_STIMULI[token] = (stim_x, stim_y)

    return StimulusResponse(type="moving-dot", x=stim_x, y=stim_y, fps=STIM_FPS, duration_sec=DURATION_SEC, token=token)


@router.post(
    "/islive",
    response_model=LivenessResponse,
)
@log_handler
async def check_liveness(
    token: str | None = Body(None, description="Token returned by GET /stim. If not provided (or expired), stim_x, stim_y will be used."),
    stim_x: list[float] | None = Body(None),
    stim_y: list[float] | None = Body(None),
    gaze_x: list[float] = Body(...),
    gaze_y: list[float] = Body(...),
    left_eye_blink: list[float] = Body(...),
    right_eye_blink: list[float] = Body(...),
    fps: float = Body(60)
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
            'left_eye_blink': left_eye_blink[:min_len],
            'right_eye_blink': right_eye_blink[:min_len],
        })

        df['blink'] = find_blinks(left_eye_blink, right_eye_blink)

        score, is_live = align_and_compare_trajectories(df, fps)

        return LivenessResponse(
            is_live=is_live,
            score=score,
            gaze_fit_x=df.gaze_fit_x,
            gaze_fit_y=df.gaze_fit_y,
            fps=fps,
            lag=df.lag.iloc[0],
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
    """Upload a CSV with raw ARKit data"""
    csv_dir = DATA_DIR / user_id / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    ext = ".csv"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[1]

    filename = f"{generate_name_from_time()}{ext}"
    filepath = csv_dir / filename

    content = await file.read()
    filepath.write_bytes(content)

    return {"message": "CSV uploaded successfully"}


@router.post("/islive_video")
@log_handler
async def islive_video(
    user_id: str = Form(...),
    token: str | None = Form(None, description="Token returned by GET /stim. If not provided (or expired), stim_x, stim_y will be used."),
    stim_x: list[float] | None = Form(None),
    stim_y: list[float] | None = Form(None),
    file: UploadFile = File(...),
    metadata: str = Form("{}", description="JSON object with recording metadata"),
):
    """Check liveness of a video"""
    video_dir = DATA_DIR / user_id / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    ext = ".mp4"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[1]

    videopath = video_dir / (generate_name_from_time() + ext)

    content = await file.read()
    videopath.write_bytes(content)

    if token:
        try:
            stim_x, stim_y = GENERATED_STIMULI[token]
        except KeyError:
            stim_x = np.array(stim_x)
            stim_y = np.array(stim_y)
    if stim_x is None or stim_y is None:
        raise HTTPException(status_code=400, detail="Stimulus not found. Maybe the token is expired.")

    try:
        metadata_obj = json.loads(metadata)
        metadata_obj.update({"stim_x": stim_x.tolist(), "stim_y": stim_y.tolist()})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

    with open(videopath.with_suffix(".json"), "w") as f:
        json.dump(metadata_obj, f, ensure_ascii=False, indent=2)

    try:
        gaze = unigaze_gaze_from_video(videopath, unigaze_model, target_fps=15)

        if STIM_FPS % gaze.fps != 0:
            raise HTTPException(status_code=400, detail=f"Stimulus FPS {STIM_FPS} must be divisible by gaze FPS {gaze.fps} after downsampling.")

        step = int(STIM_FPS / gaze.fps)
        stim_x = stim_x[::step]
        stim_y = stim_y[::step]

        min_len = min(len(stim_x), len(stim_y), len(gaze.xy))

        df = pd.DataFrame({
            'stim_x': stim_x[-min_len:],
            'stim_y': stim_y[-min_len:],
            'gaze_x': gaze.xy[:, 0][:min_len],
            'gaze_y': gaze.xy[:, 1][:min_len],
            'blink': False,
        })

        score, is_live = align_and_compare_trajectories(df, gaze.fps)

        return LivenessResponse(
            is_live=is_live,
            score=score,
            gaze_fit_x=df.gaze_fit_x,
            gaze_fit_y=df.gaze_fit_y,
            fps=gaze.fps,
            lag=df.lag.iloc[0],
            inlier=df.inlier,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")