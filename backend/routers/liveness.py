import json
import os
import time

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
import numpy as np
from pydantic import TypeAdapter, ValidationError
import torch
import unigaze

import stimulus
from models import ArkitLivenessRequest, Stimulus, StimulusResponse, LivenessResponse
from utils import DATA_DIR, generate_name_from_time
from trajectory_comparison.decide import LivenessDecision, compare_trajectories, find_blinks
from unigaze_helpers.trajectory import unigaze_gaze_from_video
from log_decorator import log_handler, stage


TORCH_DEVICE = os.getenv("TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

unigaze_model = unigaze.load("unigaze_b16_joint", device=TORCH_DEVICE)

STIM_FPS = 60
STIM_DURATION_SEC = 5

# TypeAdapter, а не Stimulus.model_validate_json: продолжит работать, когда Stimulus станет Union
_stimulus_adapter = TypeAdapter(Stimulus)

router = APIRouter(tags=["Liveness"])


def _stage(name: str, started: float) -> None:
    stage(name, time.perf_counter() - started)


def _floats(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _liveness_response(decision: LivenessDecision, fps: float) -> LivenessResponse:
    for name, seconds in decision.seconds.items():
        stage(name, seconds)
    return LivenessResponse(
        is_live=decision.is_live,
        score=decision.score,
        gaze_x=_floats(decision.gaze_x),
        gaze_y=_floats(decision.gaze_y),
        inlier=decision.inlier.tolist(),
        fps=fps,
    )


def _parse_stimulus(raw: str) -> Stimulus:
    """Объект `stim` из GET /stim, пришедший строкой в multipart-поле."""
    try:
        return _stimulus_adapter.validate_json(raw)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json(include_url=False)))


def _sampled_stimulus(stim: Stimulus, gaze_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Стимул в моменты кадров взгляда."""
    if not np.isfinite(gaze_fps) or gaze_fps <= 0:
        raise HTTPException(status_code=400, detail=f"Gaze FPS must be positive, got {gaze_fps}")
    return stimulus.sample(stim, gaze_fps)


@router.get("/stim")
@log_handler
async def generate_stimulus() -> StimulusResponse:
    """Generate a random stimulus.

    `x`, `y` are the frames to show, in [0, 1]×[0, 1]. `stim` is the compact
    description to send back with the recording; the server keeps no state,
    so any instance can check it.
    """
    stim = stimulus.random_stimulus(STIM_FPS, STIM_DURATION_SEC)
    x, y = stimulus.frames(stim)
    return StimulusResponse(stim=stim, x=x.tolist(), y=y.tolist())


@router.post("/islive_arkit", response_model=LivenessResponse)
@log_handler
async def islive_arkit(body: ArkitLivenessRequest) -> LivenessResponse:
    """Check liveness of ARKit gaze series. Gaze starts together with the stimulus."""
    stim_x, stim_y = _sampled_stimulus(body.stim, body.fps)

    try:
        min_len = min(
            len(stim_x), len(body.gaze_x), len(body.gaze_y), len(body.left_eye_blink), len(body.right_eye_blink),
        )
        blink = find_blinks(body.left_eye_blink[:min_len], body.right_eye_blink[:min_len])
        decision = await compare_trajectories(
            body.gaze_x[:min_len],
            body.gaze_y[:min_len],
            stim_x[:min_len],
            stim_y[:min_len],
            blink,
            body.stim,
        )
        return _liveness_response(decision, body.fps)

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
    stim: str = Form(..., description="JSON: the `stim` object from GET /stim"),
    file: UploadFile = File(...),
    metadata: str = Form("{}", description="JSON object with recording metadata"),
):
    """Check liveness of a video. The recording may start before the stimulus: gaze is aligned to its end."""
    parsed_stim = _parse_stimulus(stim)

    video_dir = DATA_DIR / user_id / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    ext = ".mp4"
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[1]

    videopath = video_dir / (generate_name_from_time() + ext)

    started = time.perf_counter()
    content = await file.read()
    _stage("read", started)
    started = time.perf_counter()
    videopath.write_bytes(content)
    write_seconds = time.perf_counter() - started
    stage("write", write_seconds)

    try:
        metadata_obj = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    if not isinstance(metadata_obj, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    # Описание стимула для повторной проверки и готовые кадры для анализа записи
    metadata_obj["stim"] = parsed_stim.model_dump()
    stim_x, stim_y = stimulus.frames(parsed_stim)
    metadata_obj["stim_x"] = stim_x.tolist()
    metadata_obj["stim_y"] = stim_y.tolist()

    started = time.perf_counter()
    with open(videopath.with_suffix(".json"), "w") as f:
        json.dump(metadata_obj, f, ensure_ascii=False, indent=2)
    stage("write", write_seconds + time.perf_counter() - started)

    try:
        gaze = await run_in_threadpool(
            unigaze_gaze_from_video, videopath, unigaze_model, target_fps=15
        )
        for name, seconds in gaze.seconds.items():
            stage(name, seconds)

        stim_x, stim_y = _sampled_stimulus(parsed_stim, gaze.fps)

        min_len = min(len(stim_x), len(gaze.xy))
        decision = await compare_trajectories(
            gaze.xy[:min_len, 0],
            gaze.xy[:min_len, 1],
            stim_x[-min_len:],
            stim_y[-min_len:],
            np.zeros(min_len, dtype=bool),
            parsed_stim,
        )
        return _liveness_response(decision, gaze.fps)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")