"""Траектория точки взгляда по видео.

UniGaze отдаёт направление. Точка — пересечение этого луча с плоскостью камеры.
Масштаб произвольный: у ролика нет калибровки, её снимает min-max при сравнении со стимулом.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_HELPERS = Path(__file__).resolve().parent
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))

from geometry import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    camera_plane_hit,
    crop_face,
    dummy_camera,
    estimate_head_pose,
    eye_aspect_ratios,
    face_center_camera,
    gaze_in_camera,
    head_angles,
    head_pose_rejected,
    normalize_face,
)

# Те же шесть точек 3D-модели в сетке MediaPipe: уголки глаз и крылья носа.
_MP_POINTS = (33, 133, 362, 263, 98, 327)
_MP_LEFT_EAR = (362, 385, 387, 263, 373, 380)
_MP_RIGHT_EAR = (133, 158, 160, 33, 144, 153)
# Точки сетки, которые попадают в json попытки. Видео оттуда не достать, а по
# этим точкам и кропу пересчитываются EAR, поза и всё после UniGaze.
# Сверх точек позы и EAR: центры радужек 468 и 473, кончик носа 1, лоб 10,
# подбородок 152, края лица 234 и 454, уголки рта 61 и 291, губы 13 и 14.
LANDMARK_IDS = tuple(
    sorted(
        {*_MP_POINTS, *_MP_LEFT_EAR, *_MP_RIGHT_EAR, 468, 473, 1, 10, 152, 234, 454, 61, 291, 13, 14}
    )
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

_PREDICT_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()

# Без средней частоты в контейнере (WebM из MediaRecorder) OpenCV отдаёт
# единицу времени дорожки, обычно 1000. Камеры быстрее не снимают.
_MAX_CONTAINER_FPS = 240.0


@dataclass
class FrameTrace:
    """Один прогнанный через сети кадр. Числа, которых не было, остаются None.

    index — номер кадра в ролике.
    row — строка кадра в xy и ear траектории. None, если кадр отброшен прореживанием.
    status:
    no_face — детектор не нашёл лицо;
    mesh — лицо было, сетка его не отдала;
    crop — кроп лица выродился;
    pose — solvePnP не сошёлся;
    normalize — не собралась матрица варпа;
    head — поза больше 80°;
    ray — луч взгляда параллелен плоскости камеры;
    ok — точка взгляда посчитана.
    xy — точка на плоскости камеры до min-max, в осях стимула.
    ear — меньший EAR двух глаз. ear_left и ear_right — по глазам,
    «левый» — глаз человека, на кадре он справа.
    unigaze_pitchyaw — выход UniGaze в радианах, взгляд в СК нормализованного лица.
    detector_score, presence — уверенности детектора и сетки.
    bbox — xmin, ymin, xmax, ymax детектора в пикселях кадра.
    landmarks — точки LANDMARK_IDS в пикселях кадра, массив N×2.
    crop — кроп для позы и UniGaze, x0, y0, x1, y1 в пикселях кадра.
    rotation — матрица 3×3, которой взгляд возвращается в СК камеры кропа.
    origin — центр лица в СК камеры кропа.
    rvec, tvec — поза головы solvePnP.
    head_pitch, head_yaw — радианы, те же углы, что сравниваются с порогом.
    """

    status: str
    index: int = -1
    row: int | None = None
    xy: np.ndarray | None = None
    ear: float | None = None
    ear_left: float | None = None
    ear_right: float | None = None
    unigaze_pitchyaw: np.ndarray | None = None
    detector_score: float | None = None
    presence: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    landmarks: np.ndarray | None = None
    crop: tuple[int, int, int, int] | None = None
    rotation: np.ndarray | None = None
    origin: np.ndarray | None = None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    head_pitch: float | None = None
    head_yaw: float | None = None


@dataclass(frozen=True)
class GazeTrajectory:
    """Один ряд xy и ear на оставленный кадр, в порядке кадров.

    xy: точка на плоскости камеры, в осях стимула. x растёт вправо по экрану,
    y — вниз. Масштаб произвольный.
    NaN, если лица нет, поза отвергнута или луч параллелен плоскости.
    ear: степень открытости глаз, меньший из двух. Около 0.2 и ниже — моргание.
    UniGaze сам моргания не помечает. NaN, если лица на кадре нет.
    fps: частота оставленных кадров после целого прореживания.
    source_fps: частота ролика, stride: шаг прореживания.
    fps_from_timestamps: в контейнере не было правдоподобной частоты,
    source_fps посчитан по меткам времени кадров.
    timestamps_ms: метки времени всех кадров ролика, в мс.
    frame_size: ширина и высота кадра в пикселях, None если кадров не было.
    frames: прогнанные через сети кадры с промежуточными величинами. Все кадры
    ролика, если UNIGAZE_ALL_FRAMES не 0, иначе только оставленные. Оставленные
    — те, у кого row не None. Оставленные
    — те, у кого row не None.
    seconds: длительности этапов в секундах: decode, detector, mesh, prepare,
    unigaze и, если предсказание было, geometry. detector — кроп и детектор
    лица, mesh — сетка.
    """

    xy: np.ndarray
    ear: np.ndarray
    fps: float
    seconds: dict[str, float] = field(default_factory=dict)
    source_fps: float = 0.0
    stride: int = 1
    fps_from_timestamps: bool = False
    timestamps_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    frame_size: tuple[int, int] | None = None
    frames: tuple[FrameTrace, ...] = ()


def _sample_stride(
    source_fps: float, target_fps: float | None, *, round_fps: bool = True
) -> tuple[int, float]:
    """Шаг прореживания и fps оставленных кадров.

    round_fps округляет номинальный fps до целого: контейнер часто пишет 30.03
    вместо 30. Средний fps по меткам времени кадров не округляется, иначе
    к концу ролика накапливается сдвиг относительно стимула.
    Берётся наибольший целый шаг, при котором частота не ниже target_fps.
    Кадры остаются через равные промежутки. Если ролик и так медленнее цели,
    остаётся каждый кадр.
    """
    if source_fps <= 0:
        nominal: float = 30
    elif round_fps:
        nominal = int(round(source_fps)) or 30
    else:
        nominal = source_fps
    if target_fps is None:
        return 1, float(nominal)
    if target_fps <= 0:
        raise ValueError(f"target_fps должен быть положительным, получено {target_fps}")
    stride = max(1, int(nominal // target_fps))
    return stride, nominal / stride


def unigaze_gaze_from_video(
    video: str | Path,
    model,
    *,
    batch_size: int = 32,
    target_fps: float | None = None,
) -> GazeTrajectory:
    """Посчитать траекторию взгляда для видеофайла.

    model — уже загруженный unigaze.load(...). Кадры без лица остаются NaN
    и не сдвигают остальные точки. Точки лица считают сети Face Landmarker
    пакетом: каждый кадр с нуля, без трекинга режима VIDEO и без фильтра One Euro.

    target_fps — нижняя граница частоты оставленных кадров. Берётся каждый N-й
    кадр, N целое, так что итог не медленнее target_fps. None оставляет все кадры.
    UNIGAZE_ALL_FRAMES, если не 0, прогоняет через сети и прореженные кадры:
    они попадают в frames, но не в xy.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"не удалось открыть видео: {video}")

    images = []
    pending: list[int] = []
    traces: list[FrameTrace] = []
    seconds: dict[str, float] = {}

    try:
        import torch

        container_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_from_timestamps = not 0 < container_fps <= _MAX_CONTAINER_FPS
        all_frames = _infer_all_frames()
        # Шаг заранее известен, только если частоте контейнера можно верить.
        decode_stride = 1
        if not (all_frames or fps_from_timestamps):
            decode_stride = _sample_stride(container_fps, target_fps)[0]
        started = time.perf_counter()
        frames, indices, timestamps_ms = _decoded_frames(capture, decode_stride)
        seconds["decode"] = time.perf_counter() - started
        frame_size = (int(frames[0].shape[1]), int(frames[0].shape[0])) if frames else None

        source_fps = _fps_from_timestamps(timestamps_ms) if fps_from_timestamps else container_fps
        stride, fps = _sample_stride(source_fps, target_fps, round_fps=not fps_from_timestamps)
        if not all_frames and decode_stride != stride:
            frames = [frame for frame, index in zip(frames, indices) if _kept(index, stride)]
            indices = [index for index in indices if _kept(index, stride)]

        faces, seconds["detector"], seconds["mesh"] = _detect_faces(frames, batch_size)
        started = time.perf_counter()
        rows = 0
        for frame, face, index in zip(frames, faces, indices):
            trace, image = _prepare_frame(face, frame)
            trace.index = index
            if _kept(index, stride):
                trace.row = rows
                rows += 1
            traces.append(trace)
            if image is not None:
                images.append(image)
                pending.append(len(traces) - 1)
        seconds["prepare"] = time.perf_counter() - started
        del frames

        if images:
            device = next(model.parameters()).device
            model.eval()
            # Параллельные вызовы torch на CPU делят одни ядра и тормозят в десятки раз.
            started = time.perf_counter()
            with _PREDICT_LOCK:
                pitchyaw = _predict(model, images, batch_size, device, torch)
            seconds["unigaze"] = time.perf_counter() - started
            started = time.perf_counter()
            _apply_gaze(traces, pending, pitchyaw)
            seconds["geometry"] = time.perf_counter() - started
    finally:
        capture.release()

    xy, ear = _trajectory_arrays([trace for trace in traces if trace.row is not None])
    return GazeTrajectory(
        xy=xy,
        ear=ear,
        fps=fps,
        seconds=seconds,
        source_fps=source_fps,
        stride=stride,
        fps_from_timestamps=fps_from_timestamps,
        timestamps_ms=timestamps_ms,
        frame_size=frame_size,
        frames=tuple(traces),
    )


def _decoded_frames(capture, stride: int) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Каждый stride-й кадр, номера этих кадров и метки времени всех кадров в мс."""
    frames = []
    indices = []
    timestamps = []
    while capture.grab():
        index = len(timestamps)
        if _kept(index, stride):
            ok, frame = capture.retrieve()
            if not ok:
                break
            frames.append(frame)
            indices.append(index)
        timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
    return frames, indices, np.asarray(timestamps, dtype=np.float64)


def _kept(index: int, stride: int) -> bool:
    return index % stride == 0


def _fps_from_timestamps(timestamps_ms: np.ndarray) -> float:
    """Средняя частота по первой и последней метке. 0, если считать не по чему."""
    if len(timestamps_ms) < 2:
        return 0.0
    span = float(timestamps_ms[-1] - timestamps_ms[0])
    return (len(timestamps_ms) - 1) * 1000.0 / span if span > 0 else 0.0


def _detect_faces(frames: list[np.ndarray], batch_size: int):
    from unigaze_helpers.mediapipe_batch import face_landmarks_batch

    return face_landmarks_batch(frames, batch_size=batch_size)


def _apply_gaze(traces: list[FrameTrace], pending: list[int], pitchyaw: np.ndarray) -> None:
    for row, frame_index in enumerate(pending):
        trace = traces[frame_index]
        gaze_py = np.asarray(pitchyaw[row], dtype=np.float64).reshape(2)
        trace.unigaze_pitchyaw = gaze_py
        gaze = gaze_in_camera(gaze_py, trace.rotation)
        hit = camera_plane_hit(trace.origin, gaze)
        if hit is None or not np.isfinite(hit).all():
            trace.status = "ray"
            continue
        trace.status = "ok"
        # x: кадр не зеркальный, право экрана слева на изображении.
        # y: и в кадре, и в стимуле ось смотрит вниз.
        trace.xy = np.array((-float(hit[0]), float(hit[1])), dtype=np.float64)


def _trajectory_arrays(traces: list[FrameTrace]) -> tuple[np.ndarray, np.ndarray]:
    xy = np.full((len(traces), 2), np.nan, dtype=np.float64)
    ear = np.full(len(traces), np.nan, dtype=np.float64)
    for index, trace in enumerate(traces):
        if trace.xy is not None:
            xy[index] = trace.xy
        if trace.ear is not None:
            ear[index] = trace.ear
    return xy, ear


def _prepare_frame(face, frame: np.ndarray) -> tuple[FrameTrace, np.ndarray | None]:
    """След кадра и картинка для UniGaze, если до сети дошли."""
    trace = FrameTrace(
        status="no_face",
        detector_score=face.detector_score,
        presence=face.presence,
        bbox=face.bbox,
    )
    if face.points is None:
        if face.bbox is not None:
            trace.status = "mesh"
        return trace, None

    trace.landmarks = face.points[list(LANDMARK_IDS)]
    trace.ear_left, trace.ear_right = eye_aspect_ratios(face.points, _MP_LEFT_EAR, _MP_RIGHT_EAR)
    trace.ear = min(trace.ear_left, trace.ear_right)
    cropped = crop_face(frame, face.points)
    if cropped is None:
        trace.status = "crop"
        return trace, None
    crop, shifted, trace.crop = cropped
    points = shifted[list(_MP_POINTS)]
    camera, distortion = dummy_camera(crop)
    pose = estimate_head_pose(points, camera, distortion)
    if pose is None:
        trace.status = "pose"
        return trace, None
    rvec, tvec = pose
    trace.rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    trace.tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    origin = face_center_camera(rotation_matrix, tvec)
    trace.origin = origin.reshape(3)
    normalized = normalize_face(crop, origin, rvec, camera)
    if normalized is None:
        trace.status = "normalize"
        return trace, None
    image, rotation, head_norm = normalized
    trace.rotation = rotation
    trace.head_pitch, trace.head_yaw = head_angles(head_norm)
    if head_pose_rejected(head_norm):
        trace.status = "head"
        return trace, None
    return trace, image


def _predict(model, images, batch_size: int, device, torch) -> np.ndarray:
    """images — кадры 224×224 BGR uint8. Нормализация идёт уже на устройстве."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    half = device.type == "cuda" and _use_fp16()
    predictions = []
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.float16, enabled=half),
    ):
        for start in range(0, len(images), batch_size):
            pixels = torch.from_numpy(np.stack(images[start : start + batch_size]))
            pixels = pixels.to(device, non_blocking=True)
            batch = pixels.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
            gaze = model((batch - mean) / std)["pred_gaze"]
            predictions.append(gaze.float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _use_fp16() -> bool:
    return os.getenv("UNIGAZE_FP16", "1") != "0"


def _infer_all_frames() -> bool:
    return os.getenv("UNIGAZE_ALL_FRAMES", "1") != "0"


def _mediapipe_model() -> Path:
    """FACE_LANDMARKER_MODEL, если задан. Иначе модель качается в кеш при первом вызове."""
    override = os.getenv("FACE_LANDMARKER_MODEL")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    cache = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    path = cache / "unified-liveness" / "face_landmarker.task"
    with _MODEL_LOCK:
        if not path.is_file():
            _download_model(path)
    return path


def _download_model(path: Path) -> None:
    with urllib.request.urlopen(_MODEL_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _MODEL_SHA256:
        raise RuntimeError(f"sha256 модели {digest}, ожидался {_MODEL_SHA256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(payload)
    partial.replace(path)
