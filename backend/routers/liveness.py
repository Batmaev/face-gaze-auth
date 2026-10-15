import json
import os
import time

from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
import numpy as np
import pandas as pd
import torch
import unigaze

from models import StimulusResponse, LivenessResponse
from utils import DATA_DIR, GENERATED_STIMULI, generate_short_id, generate_name_from_time
from stimulus.generate import eval_spline, random_spline, sample_at_fps
from trajectory_comparison.fit import find_blinks
from trajectory_comparison.decide import align_and_compare_trajectories
from unigaze_helpers.trajectory import unigaze_gaze_from_video
from log_decorator import log_handler, stage


TORCH_DEVICE = os.getenv("TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

unigaze_model = unigaze.load("unigaze_b16_joint", device=TORCH_DEVICE)

STIM_FPS = 60
STIM_DURATION_SEC = 5
N_CTRL_POINTS = 6
N_FRAMES = STIM_DURATION_SEC * STIM_FPS

router = APIRouter(tags=["Liveness"])


def _stage(name: str, started: float) -> None:
    stage(name, time.perf_counter() - started)


def _parse_ctrl(raw: str | None) -> np.ndarray | None:
    if raw is None or raw.strip() in ("", "null"):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="ctrl must be a JSON array of [x, y]")
    if data is None:
        return None
    return _as_ctrl(data)


def _as_ctrl(data) -> np.ndarray:
    ctrl = np.asarray(data, dtype=float)
    if ctrl.ndim != 2 or ctrl.shape[1] != 2 or len(ctrl) < 4 or not np.isfinite(ctrl).all():
        raise HTTPException(status_code=400, detail="ctrl must be an array of [x, y], at least 4 finite points")
    return ctrl


def _resolve_stimulus(
    token: str | None,
    ctrl: np.ndarray | None,
    stim_x: list[float] | np.ndarray | None,
    stim_y: list[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Кадры на частоте стимула и контрольные точки, если они есть.

    Сначала кэш этого воркера. Если токена нет (другой процесс, истёк TTL),
    траектория собирается из контрольных точек. Готовые stim_x, stim_y остаются
    для старых записей, где точек сплайна ещё не было.
    """
    if token:
        stored = GENERATED_STIMULI.get(token)
        if stored is not None:
            stim_x, stim_y, ctrl = stored
            return np.asarray(stim_x, dtype=float), np.asarray(stim_y, dtype=float), np.asarray(ctrl, dtype=float)
    if ctrl is not None:
        x, y = eval_spline(ctrl, np.linspace(0, 1, N_FRAMES), N_FRAMES)
        return x, y, ctrl
    if stim_x is not None and stim_y is not None:
        return np.asarray(stim_x, dtype=float), np.asarray(stim_y, dtype=float), None
    raise HTTPException(status_code=400, detail="Stimulus not found. Send spline control points or a token from this worker.")


def _match_gaze_fps(
    stim_x: np.ndarray,
    stim_y: np.ndarray,
    ctrl: np.ndarray | None,
    gaze_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Отсчёты стимула в те же моменты, что и кадры взгляда."""
    if not np.isfinite(gaze_fps) or gaze_fps <= 0:
        raise HTTPException(status_code=400, detail=f"Gaze FPS must be positive, got {gaze_fps}")
    ratio = STIM_FPS / gaze_fps
    if ratio >= 1 and abs(ratio - round(ratio)) < 1e-6:
        return stim_x[:: int(round(ratio))], stim_y[:: int(round(ratio))]
    if ctrl is None:
        raise HTTPException(
            status_code=400,
            detail=f"Gaze FPS {gaze_fps} does not divide stimulus FPS {STIM_FPS}; spline control points are required",
        )
    return sample_at_fps(ctrl, len(stim_x), STIM_FPS, gaze_fps)


@router.get("/stim")
@log_handler
async def generate_stimulus() -> StimulusResponse:
    """Generate random stimulus.
    Generated (x, y) are guaranteed to be in [0, 1]×[0, 1].
    """
    stim_x, stim_y, ctrl = random_spline(N_FRAMES, N_CTRL_POINTS)

    token = generate_short_id()
    GENERATED_STIMULI[token] = (stim_x, stim_y, ctrl)

    return StimulusResponse(
        type="moving-dot",
        x=stim_x,
        y=stim_y,
        ctrl=ctrl.tolist(),
        fps=STIM_FPS,
        duration_sec=STIM_DURATION_SEC,
        token=token,
    )


@router.post(
    "/islive",
    response_model=LivenessResponse,
)
@log_handler
async def check_liveness(
    token: str | None = Body(None, description="Token returned by GET /stim. If this worker does not have it, ctrl is used."),
    ctrl: list[list[float]] | None = Body(None, description="Spline control points, shape (n, 2)"),
    stim_x: list[float] | None = Body(None),
    stim_y: list[float] | None = Body(None),
    gaze_x: list[float] = Body(...),
    gaze_y: list[float] = Body(...),
    left_eye_blink: list[float] = Body(...),
    right_eye_blink: list[float] = Body(...),
    fps: float = Body(60)
) -> LivenessResponse:
    parsed_ctrl = _as_ctrl(ctrl) if ctrl is not None else None
    stim_x, stim_y, _parsed_ctrl = _resolve_stimulus(token, parsed_ctrl, stim_x, stim_y)

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

        started = time.perf_counter()
        score, is_live = await align_and_compare_trajectories(df, fps)
        _stage("align", started)

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
    token: str | None = Form(None, description="Token returned by GET /stim. If this worker does not have it, ctrl is used."),
    ctrl: str | None = Form(None, description="JSON array of spline control points [[x, y], ...]"),
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

    started = time.perf_counter()
    content = await file.read()
    _stage("read", started)
    started = time.perf_counter()
    videopath.write_bytes(content)
    write_seconds = time.perf_counter() - started
    stage("write", write_seconds)

    stim_x, stim_y, parsed_ctrl = _resolve_stimulus(token, _parse_ctrl(ctrl), stim_x, stim_y)

    try:
        metadata_obj = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    if not isinstance(metadata_obj, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    metadata_obj["stim_x"] = stim_x.tolist()
    metadata_obj["stim_y"] = stim_y.tolist()
    if parsed_ctrl is not None:
        metadata_obj["ctrl"] = parsed_ctrl.tolist()

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

        stim_x, stim_y = _match_gaze_fps(stim_x, stim_y, parsed_ctrl, gaze.fps)

        min_len = min(len(stim_x), len(stim_y), len(gaze.xy))

        df = pd.DataFrame({
            'stim_x': stim_x[-min_len:],
            'stim_y': stim_y[-min_len:],
            'gaze_x': gaze.xy[:, 0][:min_len],
            'gaze_y': gaze.xy[:, 1][:min_len],
            'blink': False,
        })

        started = time.perf_counter()
        score, is_live = await align_and_compare_trajectories(df, gaze.fps)
        _stage("align", started)

        return LivenessResponse(
            is_live=is_live,
            score=score,
            gaze_fit_x=df.gaze_fit_x,
            gaze_fit_y=df.gaze_fit_y,
            fps=gaze.fps,
            lag=df.lag.iloc[0],
            inlier=df.inlier,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")