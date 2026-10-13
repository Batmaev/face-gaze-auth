"""Траектория точки взгляда по видео.

UniGaze отдаёт направление. Точка — пересечение этого луча с плоскостью камеры.
Масштаб произвольный: у ролика нет калибровки, её закрывает последующий МНК.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import urllib.request
from dataclasses import dataclass
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
    eye_aspect_ratio,
    face_center_camera,
    gaze_in_camera,
    head_pose_rejected,
    normalize_face,
)

# Те же шесть точек 3D-модели в сетке MediaPipe: уголки глаз и крылья носа.
_MP_POINTS = (33, 133, 362, 263, 98, 327)
_MP_LEFT_EAR = (362, 385, 387, 263, 373, 380)
_MP_RIGHT_EAR = (133, 158, 160, 33, 144, 153)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

_PREDICT_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class GazeTrajectory:
    """Один ряд на кадр видео, в том же порядке.

    xy: точка на плоскости камеры, в осях стимула. x растёт вправо по экрану,
    y — вниз. Масштаб произвольный.
    NaN, если лица нет, поза отвергнута или луч параллелен плоскости.
    ear: степень открытости глаз, меньший из двух. Около 0.2 и ниже — моргание.
    UniGaze сам моргания не помечает. NaN, если лица на кадре нет.
    fps: частота оставленных кадров после целого прореживания.
    """

    xy: np.ndarray
    ear: np.ndarray
    fps: float


def _sample_stride(source_fps: float, target_fps: float | None) -> tuple[int, float]:
    """Шаг прореживания и fps оставленных кадров.

    Номинальный fps округляется до целого: контейнер часто пишет 30.03 вместо 30.
    Берётся наибольший целый шаг, при котором частота не ниже target_fps.
    Кадры остаются через равные промежутки. Если ролик и так медленнее цели,
    остаётся каждый кадр.
    """
    nominal = int(round(source_fps)) if source_fps > 0 else 30
    if nominal <= 0:
        nominal = 30
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
    и не сдвигают остальные точки. Точки лица даёт MediaPipe Face Landmarker.

    target_fps — нижняя граница частоты оставленных кадров. Берётся каждый N-й
    кадр, N целое, так что итог не медленнее target_fps. None оставляет все кадры.
    """
    reader = _MediaPipeFaces()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"не удалось открыть видео: {video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    xy_rows: list[np.ndarray] = []
    ear_rows: list[float] = []
    images = []
    samples = []

    try:
        import torch
        from tqdm import tqdm

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        stride, fps = _sample_stride(raw_fps, target_fps)
        progress = tqdm(total=frame_count or None, desc="gaze")
        source_index = 0
        while True:
            keep = source_index % stride == 0
            if keep:
                ok, frame = capture.read()
            else:
                ok = capture.grab()
            if not ok:
                break
            source_index += 1
            progress.update(1)
            if not keep:
                continue
            frame_index = len(xy_rows)
            xy_rows.append(np.full(2, np.nan, dtype=np.float64))
            ear_rows.append(np.nan)
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            prepared = _prepare_frame(reader, frame, timestamp_ms)
            if prepared is not None:
                ear_score, sample = prepared
                ear_rows[frame_index] = ear_score
                if sample is not None:
                    image, rotation, origin = sample
                    images.append(image)
                    samples.append((frame_index, rotation, origin))
        progress.close()

        if not images:
            return GazeTrajectory(
                xy=np.vstack(xy_rows) if xy_rows else np.zeros((0, 2)),
                ear=np.asarray(ear_rows, dtype=np.float64),
                fps=fps,
            )

        device = next(model.parameters()).device
        model.eval()
        # Параллельные вызовы torch на CPU делят одни ядра и тормозят в десятки раз.
        with _PREDICT_LOCK:
            pitchyaw = _predict(model, images, batch_size, device, torch)
    finally:
        capture.release()
        # close() у FaceLandmarker иногда ждёт секунду и дольше; GIL он при этом не держит.
        threading.Thread(target=reader.close, daemon=True).start()

    xy = np.vstack(xy_rows)
    for row, (frame_index, rotation, origin) in enumerate(samples):
        gaze = gaze_in_camera(pitchyaw[row], rotation)
        hit = camera_plane_hit(origin, gaze)
        if hit is not None:
            # x: кадр не зеркальный, право экрана слева на изображении.
            # y: и в кадре, и в стимуле ось смотрит вниз.
            xy[frame_index] = (-hit[0], hit[1])
    return GazeTrajectory(xy=xy, ear=np.asarray(ear_rows, dtype=np.float64), fps=fps)


def _prepare_frame(reader, frame: np.ndarray, timestamp_ms: int):
    found = reader.find(frame, timestamp_ms)
    if found is None:
        return None

    landmarks, ear = found
    cropped = crop_face(frame, landmarks)
    if cropped is None:
        return ear, None
    crop, shifted = cropped
    points = shifted[list(_MP_POINTS)]
    camera, distortion = dummy_camera(crop)
    pose = estimate_head_pose(points, camera, distortion)
    if pose is None:
        return ear, None
    rvec, tvec = pose
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    origin = face_center_camera(rotation_matrix, tvec)
    normalized = normalize_face(crop, origin, rvec, camera)
    if normalized is None or head_pose_rejected(normalized[2]):
        return ear, None
    image, rotation, _head_norm = normalized
    return ear, (image, rotation, origin)


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


class _MediaPipeFaces:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # На Mac сборка MediaPipe роняет CPU-делегат в детекторе лица.
        # GPU принимает кадр только как RGBA.
        self._gpu = _mediapipe_delegate() == "gpu"
        delegate = mp_python.BaseOptions.Delegate
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(_mediapipe_model()),
                delegate=delegate.GPU if self._gpu else delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._previous_ms = -1

    def find(self, frame: np.ndarray, timestamp_ms: int):
        if timestamp_ms <= self._previous_ms:
            timestamp_ms = self._previous_ms + 1
        self._previous_ms = timestamp_ms
        if self._gpu:
            pixels = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
            image_format = self._mp.ImageFormat.SRGBA
        else:
            pixels = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_format = self._mp.ImageFormat.SRGB
        image = self._mp.Image(image_format=image_format, data=pixels)
        detection = self._landmarker.detect_for_video(image, timestamp_ms)
        if not detection.face_landmarks:
            return None
        height, width = frame.shape[:2]
        mesh = detection.face_landmarks[0]
        landmarks = np.array(
            [[point.x * width, point.y * height] for point in mesh],
            dtype=np.float64,
        )
        ear = eye_aspect_ratio(landmarks, _MP_LEFT_EAR, _MP_RIGHT_EAR)
        return landmarks, ear

    def close(self):
        self._landmarker.close()


def _mediapipe_delegate() -> str:
    delegate = os.getenv("MEDIAPIPE_DELEGATE") or ("gpu" if sys.platform == "darwin" else "cpu")
    if delegate not in ("cpu", "gpu"):
        raise ValueError(f"MEDIAPIPE_DELEGATE должен быть cpu или gpu, получено {delegate}")
    return delegate


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
