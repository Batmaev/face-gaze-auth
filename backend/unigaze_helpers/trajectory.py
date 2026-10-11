"""Траектория точки взгляда по видео.

UniGaze отдаёт направление. Точка — пересечение этого луча с плоскостью камеры.
Масштаб произвольный: у ролика нет калибровки, её закрывает последующий МНК.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

_HELPERS = Path(__file__).resolve().parent
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))

from geometry import (
    FACE_POINT_INDICES,
    camera_plane_hit,
    crop_face,
    dummy_camera,
    estimate_head_pose,
    eye_aspect_ratio,
    face_center_camera,
    gaze_in_camera,
    head_pose_rejected,
    imagenet_tensor,
    normalize_face,
)

_DETECT_SCALE = 0.5

# Те же шесть точек 3D-модели в сетке MediaPipe: уголки глаз и крылья носа.
_MP_POINTS = (33, 133, 362, 263, 98, 327)
_MP_LEFT_EAR = (362, 385, 387, 263, 373, 380)
_MP_RIGHT_EAR = (133, 158, 160, 33, 144, 153)

LandmarksBackend = Literal["mediapipe", "face_alignment"]


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
    batch_size: int = 4,
    landmarks: LandmarksBackend = "mediapipe",
    target_fps: float | None = None,
) -> GazeTrajectory:
    """Посчитать траекторию взгляда для видеофайла.

    model — уже загруженный unigaze.load(...). Кадры без лица остаются NaN
    и не сдвигают остальные точки.

    landmarks — откуда брать точки лица. ``mediapipe`` размечает кадр целиком
    своей сеткой. ``face_alignment`` — прежний путь: BlazeFace и 68 точек FAN.

    target_fps — нижняя граница частоты оставленных кадров. Берётся каждый N-й
    кадр, N целое, так что итог не медленнее target_fps. None оставляет все кадры.
    """
    reader = _make_reader(landmarks)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"не удалось открыть видео: {video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    xy_rows: list[np.ndarray] = []
    ear_rows: list[float] = []
    tensors = []
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
                    tensor, rotation, origin = sample
                    tensors.append(tensor)
                    samples.append((frame_index, rotation, origin))
        progress.close()

        if not tensors:
            return GazeTrajectory(
                xy=np.vstack(xy_rows) if xy_rows else np.zeros((0, 2)),
                ear=np.asarray(ear_rows, dtype=np.float64),
                fps=fps,
            )

        device = next(model.parameters()).device
        model.eval()
        pitchyaw = _predict(model, tensors, batch_size, device, torch)
    finally:
        capture.release()
        reader.close()

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
    points = shifted[list(reader.point_indices)]
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
    return ear, (imagenet_tensor(image), rotation, origin)


def _face_area(landmarks: np.ndarray) -> float:
    return float(np.ptp(landmarks[:, 0]) * np.ptp(landmarks[:, 1]))


def _predict(model, tensors, batch_size: int, device, torch) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            gaze = model(batch)["pred_gaze"]
            predictions.append(gaze.detach().float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _make_reader(landmarks: LandmarksBackend):
    if landmarks == "mediapipe":
        return _MediaPipeFaces()
    if landmarks == "face_alignment":
        return _FaceAlignmentFaces()
    raise ValueError(f"неизвестный источник точек: {landmarks}")


class _MediaPipeFaces:
    point_indices = _MP_POINTS

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # На Mac сборка MediaPipe роняет CPU-делегат в детекторе лица.
        # GPU принимает кадр только как RGBA. На сервере GPU нет.
        self._gpu = sys.platform == "darwin"
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


class _FaceAlignmentFaces:
    point_indices = FACE_POINT_INDICES

    def __init__(self):
        self._aligner = _make_aligner()

    def find(self, frame: np.ndarray, timestamp_ms: int):
        del timestamp_ms
        small = cv2.resize(
            frame,
            dsize=None,
            fx=_DETECT_SCALE,
            fy=_DETECT_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        preds = self._aligner.get_landmarks(rgb)
        if not preds:
            return None
        landmarks = max(preds, key=_face_area).astype(np.float64) / _DETECT_SCALE
        return landmarks, eye_aspect_ratio(landmarks)

    def close(self):
        return None


def _mediapipe_model() -> Path:
    spec = importlib.util.find_spec("webeyetrack")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("не найден face_landmarker; нужен пакет webeyetrack")
    path = Path(spec.origin).parent / "model_weights" / "face_landmarker_v2_with_blendshapes.task"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _make_aligner():
    import face_alignment

    try:
        kind = face_alignment.LandmarksType.TWO_D
    except AttributeError:
        kind = face_alignment.LandmarksType._2D
    # S3FD на полном кадре занимает около секунды. BlazeFace ищет только
    # прямоугольник, 68 точек по-прежнему считает FAN.
    return face_alignment.FaceAlignment(
        kind,
        flip_input=False,
        device="cpu",
        compile=False,
        face_detector="blazeface",
        face_detector_kwargs={"back_model": True},
    )
