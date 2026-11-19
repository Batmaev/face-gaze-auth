import json
import os
from datetime import datetime
from pathlib import Path
import string
import time

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.concurrency import run_in_threadpool
import numpy as np
from pydantic import TypeAdapter, ValidationError
import torch
import unigaze

import stimulus
from models import Stimulus, StimulusResponse, LivenessResponse
from utils import DATA_DIR, USER_ID_MAX_LENGTH, USER_ID_PATTERN
from trajectory_comparison.decide import LivenessDecision, compare_trajectories
from unigaze_helpers.trajectory import LANDMARK_IDS, FrameTrace, GazeTrajectory, unigaze_gaze_from_video
from log_decorator import format_json_compact_arrays, log_handler, stage


TORCH_DEVICE = os.getenv("TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

unigaze_model = unigaze.load("unigaze_b16_joint", device=TORCH_DEVICE)

STIM_FPS = 60
STIM_DURATION_SEC = 5
STIM_N_CTRL = 6
STIM_SCORE_THRESHOLD = 0.99
# После неудачной попытки: стимул длиннее и сложнее, порог строже.
RETRY_DURATION_SEC = 8
RETRY_N_CTRL = 10
RETRY_SCORE_THRESHOLD = 0.9975
ATTEMPT_ID_LENGTH = 6
# id попытки — секунды от 2026-01-01 UTC в base36. Шести символов хватает до конца 2094 года.
_ATTEMPT_EPOCH = 1767225600
_BASE36 = string.digits + string.ascii_lowercase

# TypeAdapter, а не Stimulus.model_validate_json: продолжит работать, когда Stimulus станет Union
_stimulus_adapter = TypeAdapter(Stimulus)

router = APIRouter(tags=["Liveness"])


def _floats(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _num(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _vec(value) -> list[float] | None:
    if value is None:
        return None
    flat = np.asarray(value, dtype=float).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        return None
    return [float(item) for item in flat]


def _matrix(value: np.ndarray | None) -> list[list[float]] | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        return None
    return matrix.tolist()


def _points(value: np.ndarray | None) -> list[float] | None:
    flat = _vec(value)
    return None if flat is None else [round(item, 2) for item in flat]


def _frame_record(trace: FrameTrace, gaze: GazeTrajectory, compared_frames: int) -> dict:
    """Кадр без пустых полей: пропуск легче читать, чем пачку null."""
    record: dict = {
        "index": trace.index,
        "t_ms": round(float(gaze.timestamps_ms[trace.index]), 3),
        # Используется ли кадр для сравнения после прореживания и отбрасывания хвоста.
        "compared": trace.row is not None and trace.row < compared_frames,
        "status": trace.status,
    }
    fields = {
        "xy": _vec(trace.xy),
        "ear": _num(trace.ear),
        "ear_left": _num(trace.ear_left),
        "ear_right": _num(trace.ear_right),
        "unigaze_pitchyaw": _vec(trace.unigaze_pitchyaw),
        "detector_score": _num(trace.detector_score),
        "presence": _num(trace.presence),
        "bbox": _vec(trace.bbox),
        "landmarks_xy_flat": _points(trace.landmarks),
        "crop": list(trace.crop) if trace.crop else None,
        "rotation": _matrix(trace.rotation),
        "origin": _vec(trace.origin),
        "rvec": _vec(trace.rvec),
        "tvec": _vec(trace.tvec),
        "head_pitch": _num(trace.head_pitch),
        "head_yaw": _num(trace.head_yaw),
    }
    record.update({key: value for key, value in fields.items() if value is not None})
    return record


def _processing_record(gaze: GazeTrajectory, compared_frames: int) -> dict:
    return {
        "fps": gaze.fps,
        "source_fps": gaze.source_fps,
        "stride": gaze.stride,
        "fps_from_timestamps": gaze.fps_from_timestamps,
        # Метки времени всех кадров ролика, в том числе не прогнанных через сети.
        # Индекс — index кадра в frames.
        "timestamps_ms": [round(float(value), 3) for value in gaze.timestamps_ms],
        "frame_size": list(gaze.frame_size) if gaze.frame_size else None,
        # Номера точек сетки MediaPipe в том порядке, в каком они лежат в landmarks_xy_flat кадра.
        "landmark_ids": list(LANDMARK_IDS),
        # unigaze_pitchyaw и head_pitch/head_yaw — радианы. xy — плоскость камеры до min-max.
        # bbox — пиксели кадра, xmin ymin xmax ymax. rotation — 3×3 в СК камеры кропа.
        # landmarks_xy_flat — x0 y0 x1 y1 … в пикселях кадра, reshape(-1, 2) даёт точки
        # в порядке landmark_ids. Плоский, чтобы кадр занимал одну строку.
        # crop — x0 y0 x1 y1 кропа, по нему строится фиктивная камера позы.
        "frames": [_frame_record(trace, gaze, compared_frames) for trace in gaze.frames],
    }


def _decision_record(
    decision: LivenessDecision,
    stim_x: np.ndarray,
    stim_y: np.ndarray,
) -> dict:
    compared = int(decision.gaze_x.shape[0])
    return {
        "is_live": decision.is_live,
        "score": decision.score,
        "score_threshold": decision.score_threshold,
        "raw_logit": decision.raw_logit,
        "features": decision.features,
        # Взгляд и стимул сравниваются с первого кадра, хвост длиннее этого числа отброшен.
        "compared_frames": compared,
        "stim_x": stim_x[:compared].tolist(),
        "stim_y": stim_y[:compared].tolist(),
        "gaze_x": _floats(decision.gaze_x),
        "gaze_y": _floats(decision.gaze_y),
        "inlier": decision.inlier.tolist(),
    }


def _video_ext(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ".mp4"
    ext = filename.rsplit(".", 1)[1].lower()
    if ext.isascii() and ext.isalnum() and 1 <= len(ext) <= 4:
        return f".{ext}"
    return ".mp4"


def _base36(value: int, width: int) -> str:
    digits = []
    for _ in range(width):
        value, rest = divmod(value, 36)
        digits.append(_BASE36[rest])
    return "".join(reversed(digits))


def _allocate_attempt(directory: Path, ext: str) -> tuple[str, Path, Path]:
    """Короткий возрастающий id попытки: `attempts/<id>.webm` и `attempts/<id>.json`.

    Ширина постоянная, а цифры в ASCII идут раньше букв, поэтому сортировка
    по имени совпадает с порядком создания. Пустой json создаётся с O_EXCL:
    если секунда занята, берётся следующая.
    """
    directory.mkdir(parents=True, exist_ok=True)
    seconds = int(time.time()) - _ATTEMPT_EPOCH
    while True:
        attempt_id = _base36(seconds, ATTEMPT_ID_LENGTH)
        json_path = directory / f"{attempt_id}.json"
        try:
            fd = os.open(json_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            seconds += 1
            continue
        os.close(fd)
        return attempt_id, directory / f"{attempt_id}{ext}", json_path


async def _write_bytes(path: Path, content: bytes) -> None:
    async with aiofiles.open(path, "wb") as f:
        await f.write(content)


async def _write_record(path: Path, record: dict) -> None:
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(format_json_compact_arrays(record))


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        return detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    return f"{type(exc).__name__}: {exc}"


def _mark(timings: dict[str, float], name: str, seconds: float) -> None:
    timings[name] = seconds
    stage(name, seconds)


def _liveness_response(decision: LivenessDecision, fps: float, processing_ms: int) -> LivenessResponse:
    return LivenessResponse(
        is_live=decision.is_live,
        score=decision.score,
        gaze_x=_floats(decision.gaze_x),
        gaze_y=_floats(decision.gaze_y),
        inlier=decision.inlier.tolist(),
        fps=fps,
        processing_ms=processing_ms,
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
async def generate_stimulus(
    retry: bool = Query(False, description="After a failed attempt: longer, more complex stimulus and a stricter threshold"),
) -> StimulusResponse:
    """Generate a random stimulus.

    `x`, `y` are the frames to show, in [0, 1]×[0, 1]. `stim` is the compact
    description to send back with the recording; the server keeps no state,
    so any instance can check it.
    """
    if retry:
        stim = stimulus.random_stimulus(STIM_FPS, RETRY_DURATION_SEC, RETRY_N_CTRL)
    else:
        stim = stimulus.random_stimulus(STIM_FPS, STIM_DURATION_SEC, STIM_N_CTRL)
    x, y = stimulus.frames(stim)
    return StimulusResponse(stim=stim, x=x.tolist(), y=y.tolist())


@router.post("/islive_video")
@log_handler
async def islive_video(
    user_id: str = Form(..., pattern=USER_ID_PATTERN, max_length=USER_ID_MAX_LENGTH, description="ASCII letters and digits"),
    stim: str = Form(..., description="JSON: the `stim` object from GET /stim"),
    file: UploadFile = File(...),
    metadata: str = Form("{}", description="JSON object with recording metadata"),
    retry: bool = Form(False, description="The stimulus came from GET /stim?retry=true: stricter threshold"),
):
    """Check liveness of a video. Recording and stimulus start together; extra frames at the end are dropped.

    The recording is stored at `data/<user_id>/attempts/<id>.<ext>`. The sidecar
    `<id>.json` keeps client metadata, the stimulus, and the intermediate values
    of the pipeline, so a failed attempt can be inspected without the video.
    """
    parsed_stim = _parse_stimulus(stim)
    try:
        metadata_obj = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    if not isinstance(metadata_obj, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

    processing_started = time.perf_counter()
    shown_x, shown_y = stimulus.frames(parsed_stim)
    attempt_id, videopath, json_path = _allocate_attempt(DATA_DIR / user_id / "attempts", _video_ext(file.filename))
    record = {
        **metadata_obj,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "user_id": user_id,
        "attempt_id": attempt_id,
        "retry": retry,
        "file": videopath.name,
        "stim": parsed_stim.model_dump(),
        "stim_x": shown_x.tolist(),
        "stim_y": shown_y.tolist(),
    }

    timings: dict[str, float] = {}
    started = time.perf_counter()
    content = await file.read()
    _mark(timings, "read", time.perf_counter() - started)
    started = time.perf_counter()
    await _write_bytes(videopath, content)
    await _write_record(json_path, record)
    _mark(timings, "write", time.perf_counter() - started)

    try:
        gaze = await run_in_threadpool(
            unigaze_gaze_from_video, videopath, unigaze_model, target_fps=15
        )
        for name, seconds in gaze.seconds.items():
            _mark(timings, name, seconds)

        stim_x, stim_y = _sampled_stimulus(parsed_stim, gaze.fps)
        # Фронтенд запускает стимул по событию start у MediaRecorder и останавливает
        # запись после последнего кадра стимула: лишние кадры бывают только в хвосте.
        n = min(len(stim_x), len(gaze.xy))
        decision = await compare_trajectories(
            gaze.xy[:n, 0],
            gaze.xy[:n, 1],
            stim_x[:n],
            stim_y[:n],
            np.zeros(n, dtype=bool),
            parsed_stim,
            RETRY_SCORE_THRESHOLD if retry else STIM_SCORE_THRESHOLD,
        )
        for name, seconds in decision.seconds.items():
            _mark(timings, name, seconds)
        record["seconds"] = {name: round(value, 4) for name, value in timings.items()}
        record["processing"] = _processing_record(gaze, n)
        record["decision"] = _decision_record(decision, stim_x, stim_y)
        await _write_record(json_path, record)
        processing_ms = round((time.perf_counter() - processing_started) * 1000)
        return _liveness_response(decision, gaze.fps, processing_ms)

    except HTTPException as exc:
        record["error"] = _error_text(exc)
        record["seconds"] = {name: round(value, 4) for name, value in timings.items()}
        await _write_record(json_path, record)
        raise
    except Exception as exc:
        record["error"] = _error_text(exc)
        record["seconds"] = {name: round(value, 4) for name, value in timings.items()}
        await _write_record(json_path, record)
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")