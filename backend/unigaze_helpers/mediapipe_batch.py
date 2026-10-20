"""Те же сети Face Landmarker, но пакетом, без трекинга между кадрами.

Детектор и сетка — файлы из face_landmarker.task. Каждый кадр считается
отдельно: детектор по всему кадру, затем сетка по найденному лицу. Это
покадровый режим MediaPipe, без прямоугольника с прошлого кадра и без
фильтра One Euro, который включён в режиме видео.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

_DETECTOR_SIZE = 128
_LANDMARK_SIZE = 256
_NUM_LANDMARKS = 478
_MIN_SCORE = 0.5
# Сетка больше не отбрасывает кадр по presence. Порог детектора остаётся _MIN_SCORE.
_MIN_PRESENCE = 0.0
_MIN_IOU = 0.5
_RECT_SCALE = 1.5


def face_landmarks_batch(
    frames_bgr: list[np.ndarray],
    *,
    batch_size: int = 32,
) -> tuple[list[np.ndarray | None], float, float]:
    """Точки лица для каждого кадра, в пикселях кадра. None, если лица нет.

    Вместе с точками возвращаются секунды детектора и сетки. В детектор входит
    загрузка его сети и кроп: перевод цвета и оба варпа. Сетка — пакетное
    предсказание и разбор точек.

    Сетка считается одним пакетом. На NVIDIA она идёт через LiteRT GPU
    (Vulkan/WebGPU), в float32: так точки совпадают с CPU. Детектор остаётся
    на CPU, он и так около миллисекунды на кадр. На Apple GPU пакет был
    медленнее и разъезжался, поэтому без CUDA остаётся CPU.
    MEDIAPIPE_DEVICE=cpu|gpu|auto.
    """
    if not frames_bgr:
        return [], 0.0, 0.0
    started = time.perf_counter()
    runner = _Runner()
    setup = time.perf_counter() - started
    try:
        points, detector_s, mesh_s = runner.landmarks(frames_bgr, batch_size)
        return points, detector_s + setup, mesh_s
    finally:
        runner.close()


class _Runner:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter

        detector, landmarks = _model_files()
        self._landmarks_path = landmarks
        self._landmarks = None
        self._detector = Interpreter(model_path=str(detector))
        self._detector.allocate_tensors()

    def _cpu_landmarks(self):
        if self._landmarks is None:
            from ai_edge_litert.interpreter import Interpreter

            self._landmarks = Interpreter(model_path=str(self._landmarks_path))
            self._landmarks.allocate_tensors()
        return self._landmarks

    def close(self):
        return None

    def landmarks(
        self, frames: list[np.ndarray], batch_size: int
    ) -> tuple[list[np.ndarray | None], float, float]:
        started = time.perf_counter()
        crops: list[np.ndarray | None] = []
        contexts = []
        for frame in frames:
            prepared = self._crop(frame)
            if prepared is None:
                crops.append(None)
                contexts.append(None)
            else:
                crop, context = prepared
                crops.append(crop)
                contexts.append(context)
        detector_s = time.perf_counter() - started

        ready = [(index, crop) for index, crop in enumerate(crops) if crop is not None]
        points: list[np.ndarray | None] = [None] * len(frames)
        if not ready:
            return points, detector_s, 0.0

        started = time.perf_counter()
        predicted = self._predict_tflite([crop for _, crop in ready], batch_size)

        for row, (index, _) in enumerate(ready):
            decoded = _decode_landmarks(predicted[row], contexts[index])
            if decoded is not None:
                points[index] = decoded
        return points, detector_s, time.perf_counter() - started

    def _crop(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        roi = _pad_roi(_full_image_roi(width, height), _DETECTOR_SIZE, _DETECTOR_SIZE)
        tensor = _warp(rgb, roi.padded, _DETECTOR_SIZE)
        tensor = (tensor.astype(np.float32) - 127.5) / 127.5
        boxes, scores = self._detect(tensor)
        if len(scores) == 0:
            return None
        best = _best_face(_project_detections(boxes, scores, roi.matrix(width, height)))
        if best is None:
            return None
        rect = _expanded_rect(best, width, height)
        padded = _pad_roi(rect, _LANDMARK_SIZE, _LANDMARK_SIZE)
        crop = _warp(rgb, padded.padded, _LANDMARK_SIZE)
        crop = crop.astype(np.float32) / 255.0
        return crop, (padded.padding, rect, width, height)

    def _detect(self, tensor: np.ndarray):
        details = {item["name"]: item for item in self._detector.get_input_details()}
        self._detector.set_tensor(details["input"]["index"], tensor[None])
        self._detector.invoke()
        outputs = {
            item["name"]: self._detector.get_tensor(item["index"])
            for item in self._detector.get_output_details()
        }
        return _decode_blazeface(outputs["regressors"][0], outputs["classificators"][0, :, 0])

    def _predict_tflite(self, crops: list[np.ndarray], batch_size: int) -> np.ndarray:
        gpu = _gpu_landmarks(batch_size)
        if gpu is not None:
            return gpu.predict(crops)
        landmarks = self._cpu_landmarks()
        predicted = []
        input_index = landmarks.get_input_details()[0]["index"]
        for start in range(0, len(crops), batch_size):
            batch = np.stack(crops[start : start + batch_size])
            landmarks.resize_tensor_input(input_index, batch.shape)
            landmarks.allocate_tensors()
            landmarks.set_tensor(input_index, batch)
            landmarks.invoke()
            output = _landmark_output(landmarks)
            predicted.append(output)
        return np.concatenate(predicted, axis=0)


_GPU_LOCK = threading.Lock()
_GPU_NETS: dict[int, "_GpuLandmarks"] = {}
_GPU_FAILED = False
_GPU_CHOICE: bool | None = None


def _want_gpu() -> bool:
    """auto: GPU только там, где есть CUDA. Apple GPU для этой сетки хуже CPU."""
    global _GPU_CHOICE
    if _GPU_CHOICE is None:
        choice = os.getenv("MEDIAPIPE_DEVICE", "auto")
        if choice == "cpu":
            _GPU_CHOICE = False
        elif choice == "gpu":
            _GPU_CHOICE = True
        else:
            import torch

            _GPU_CHOICE = bool(torch.cuda.is_available())
    return _GPU_CHOICE


def _gpu_landmarks(batch_size: int) -> "_GpuLandmarks | None":
    global _GPU_FAILED
    if _GPU_FAILED or not _want_gpu():
        return None
    with _GPU_LOCK:
        net = _GPU_NETS.get(batch_size)
        if net is not None:
            return net
        try:
            net = _GpuLandmarks(batch_size)
        except Exception as exc:
            _GPU_FAILED = True
            logging.getLogger(__name__).warning("MediaPipe GPU недоступен, сетка на CPU: %s", exc)
            return None
        _GPU_NETS[batch_size] = net
        return net


class _GpuLandmarks:
    """Сетка Face Landmarker на GPU. Размер пакета фиксирован: смена размера заново компилирует граф."""

    def __init__(self, batch_size: int):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import GpuOptions, Options

        _, landmarks = _model_files()
        # float32 обязателен: fp16 уезжает на пару пикселей кропа.
        options = Options(
            hardware_accelerators=HardwareAccelerator.GPU | HardwareAccelerator.CPU,
            gpu_options=GpuOptions(enforce_f32=True),
        )
        self.batch = batch_size
        self.model = CompiledModel.from_file(str(landmarks), options=options)
        if not self.model.is_fully_accelerated():
            raise RuntimeError("сетка не целиком на GPU")
        self.model.resize_input_tensor(
            0, [batch_size, _LANDMARK_SIZE, _LANDMARK_SIZE, 3], strict=False
        )
        self.inputs = self.model.create_input_buffers(0)
        self.outputs = self.model.create_output_buffers(0)
        self.sizes = [
            self.model.get_output_buffer_requirements(index)["buffer_size"] // 4
            for index in range(len(self.outputs))
        ]
        self.shapes = [buffer.get_tensor_details()["shape"] for buffer in self.outputs]
        dummy = np.zeros((batch_size, _LANDMARK_SIZE, _LANDMARK_SIZE, 3), np.float32)
        self._invoke(dummy)

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        with _GPU_LOCK:
            rows = []
            for start in range(0, len(crops), self.batch):
                chunk = crops[start : start + self.batch]
                batch = np.zeros((self.batch, _LANDMARK_SIZE, _LANDMARK_SIZE, 3), np.float32)
                batch[: len(chunk)] = np.stack(chunk)
                rows.append(self._invoke(batch)[: len(chunk)])
            return np.concatenate(rows, axis=0)

    def _invoke(self, batch: np.ndarray) -> np.ndarray:
        self.inputs[0].write(np.ascontiguousarray(batch))
        self.model.run_by_index(0, self.inputs, self.outputs)
        outputs = [
            buffer.read(size, np.float32).reshape(shape)
            for buffer, size, shape in zip(self.outputs, self.sizes, self.shapes)
        ]
        return _landmark_rows(outputs)


_FILES_LOCK = threading.Lock()


def _model_files() -> tuple[Path, Path]:
    from unigaze_helpers.trajectory import _mediapipe_model

    task = _mediapipe_model()
    cache = task.parent / "face_landmarker"
    detector = cache / "face_detector.tflite"
    landmarks = cache / "face_landmarks_detector.tflite"
    with _FILES_LOCK:
        if detector.is_file() and landmarks.is_file():
            return detector, landmarks
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(task) as archive:
            _write_extracted(detector, archive.read("face_detector.tflite"))
            _write_extracted(landmarks, archive.read("face_landmarks_detector.tflite"))
    return detector, landmarks


def _write_extracted(path: Path, payload: bytes) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(payload)
    partial.replace(path)


def _full_image_roi(width: int, height: int) -> "_Roi":
    return _Roi(width / 2, height / 2, float(width), float(height), 0.0)


class _Roi:
    def __init__(self, center_x: float, center_y: float, width: float, height: float, rotation: float):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.rotation = rotation


class _PaddedRoi:
    def __init__(self, original: _Roi, padded: _Roi, padding: tuple[float, float, float, float]):
        self.original = original
        self.padded = padded
        self.padding = padding

    def matrix(self, image_width: int, image_height: int) -> np.ndarray:
        return _roi_matrix(self.padded, image_width, image_height)


def _pad_roi(roi: _Roi, tensor_width: int, tensor_height: int) -> _PaddedRoi:
    tensor_aspect = tensor_height / tensor_width
    roi_aspect = roi.height / roi.width
    if tensor_aspect > roi_aspect:
        width = roi.width
        height = roi.width * tensor_aspect
        vertical = (1.0 - roi_aspect / tensor_aspect) / 2.0
        horizontal = 0.0
    else:
        height = roi.height
        width = roi.height / tensor_aspect
        horizontal = (1.0 - tensor_aspect / roi_aspect) / 2.0
        vertical = 0.0
    padded = _Roi(roi.center_x, roi.center_y, width, height, roi.rotation)
    return _PaddedRoi(roi, padded, (horizontal, vertical, horizontal, vertical))


def _roi_matrix(roi: _Roi, image_width: int, image_height: int) -> np.ndarray:
    """[0, 1] выровненного кропа -> нормированные координаты кадра."""
    cosine = float(np.cos(roi.rotation))
    sine = float(np.sin(roi.rotation))
    scale_x = 1.0 / image_width
    scale_y = 1.0 / image_height
    matrix = np.zeros(16, dtype=np.float64)
    matrix[0] = roi.width * cosine * scale_x
    matrix[1] = -roi.height * sine * scale_x
    matrix[3] = (-0.5 * roi.width * cosine + 0.5 * roi.height * sine + roi.center_x) * scale_x
    matrix[4] = roi.width * sine * scale_y
    matrix[5] = roi.height * cosine * scale_y
    matrix[7] = (-0.5 * roi.height * cosine - 0.5 * roi.width * sine + roi.center_y) * scale_y
    return matrix


def _warp(image: np.ndarray, roi: _Roi, size: int) -> np.ndarray:
    """Кроп в квадрат. Сэмпл берётся из центра пикселя, как у GPU-делегата MediaPipe."""
    height, width = image.shape[:2]
    matrix = _roi_matrix(roi, width, height)
    to_source = np.array(
        [
            [matrix[0] * width, matrix[1] * width, matrix[3] * width],
            [matrix[4] * height, matrix[5] * height, matrix[7] * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pixel_center = np.array(
        [[1.0 / size, 0.0, 0.5 / size], [0.0, 1.0 / size, 0.5 / size], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = (to_source @ pixel_center).astype(np.float32)
    return cv2.warpPerspective(
        image,
        transform,
        (size, size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _anchors() -> np.ndarray:
    options = dict(
        strides=(8, 16, 16, 16),
        min_scale=0.1484375,
        max_scale=0.75,
        offset=0.5,
        aspect=1.0,
        interpolated=1.0,
    )
    anchors = []
    layer = 0
    strides = options["strides"]
    while layer < len(strides):
        scales = []
        aspects = []
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            scale = options["min_scale"] + (options["max_scale"] - options["min_scale"]) * last / (len(strides) - 1)
            scales.append(scale)
            aspects.append(options["aspect"])
            if options["interpolated"] > 0:
                scale_next = 1.0 if last == len(strides) - 1 else (
                    options["min_scale"]
                    + (options["max_scale"] - options["min_scale"]) * (last + 1) / (len(strides) - 1)
                )
                scales.append(float(np.sqrt(scale * scale_next)))
                aspects.append(options["interpolated"])
            last += 1
        stride = strides[layer]
        cells = int(np.ceil(_DETECTOR_SIZE / stride))
        for y in range(cells):
            for x in range(cells):
                for _aspect in aspects:
                    anchors.append(((x + options["offset"]) / cells, (y + options["offset"]) / cells))
        layer = last
    return np.asarray(anchors, dtype=np.float64)


_ANCHORS = None


def _decode_blazeface(regressors: np.ndarray, scores: np.ndarray):
    global _ANCHORS
    if _ANCHORS is None:
        _ANCHORS = _anchors()
    probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))
    keep = probability >= _MIN_SCORE
    if not np.any(keep):
        return np.zeros((0, 16)), np.zeros(0)
    raw = regressors[keep]
    anchors = _ANCHORS[keep]
    probability = probability[keep]
    box = raw.copy()
    box[:, 0] = raw[:, 0] / _DETECTOR_SIZE + anchors[:, 0]
    box[:, 1] = raw[:, 1] / _DETECTOR_SIZE + anchors[:, 1]
    box[:, 2] = raw[:, 2] / _DETECTOR_SIZE
    box[:, 3] = raw[:, 3] / _DETECTOR_SIZE
    for keypoint in range(6):
        offset = 4 + keypoint * 2
        box[:, offset] = raw[:, offset] / _DETECTOR_SIZE + anchors[:, 0]
        box[:, offset + 1] = raw[:, offset + 1] / _DETECTOR_SIZE + anchors[:, 1]
    ymin = box[:, 1] - box[:, 3] / 2
    xmin = box[:, 0] - box[:, 2] / 2
    ymax = box[:, 1] + box[:, 3] / 2
    xmax = box[:, 0] + box[:, 2] / 2
    decoded = box.copy()
    decoded[:, 0] = ymin
    decoded[:, 1] = xmin
    decoded[:, 2] = ymax
    decoded[:, 3] = xmax
    return decoded, probability


def _project_detections(boxes: np.ndarray, scores: np.ndarray, matrix: np.ndarray):
    if len(scores) == 0:
        return boxes, scores

    def project(x, y):
        return (
            matrix[0] * x + matrix[1] * y + matrix[3],
            matrix[4] * x + matrix[5] * y + matrix[7],
        )

    projected = boxes.copy()
    for index in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[index, :4]
        corners = [project(xmin, ymin), project(xmax, ymin), project(xmax, ymax), project(xmin, ymax)]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        projected[index, 0] = min(ys)
        projected[index, 1] = min(xs)
        projected[index, 2] = max(ys)
        projected[index, 3] = max(xs)
        for keypoint in range(6):
            offset = 4 + keypoint * 2
            x, y = project(boxes[index, offset], boxes[index, offset + 1])
            projected[index, offset] = x
            projected[index, offset + 1] = y
    return projected, scores


def _best_face(detections):
    boxes, scores = detections
    if len(scores) == 0:
        return None
    order = np.argsort(-scores)
    used = np.zeros(len(scores), dtype=bool)
    for index in order:
        if used[index]:
            continue
        group = [index]
        for other in order:
            if other == index or used[other]:
                continue
            if _iou(boxes[index], boxes[other]) > _MIN_IOU:
                group.append(other)
        for item in group:
            used[item] = True
        weights = scores[group]
        total = float(weights.sum())
        merged = (boxes[group] * weights[:, None]).sum(axis=0) / total
        return merged, float(scores[index])
    return None


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[1]), float(second[1]))
    top = max(float(first[0]), float(second[0]))
    right = min(float(first[3]), float(second[3]))
    bottom = min(float(first[2]), float(second[2]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, float(first[3] - first[1])) * max(0.0, float(first[2] - first[0]))
    area_second = max(0.0, float(second[3] - second[1])) * max(0.0, float(second[2] - second[0]))
    union = area_first + area_second - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _expanded_rect(detection, width: int, height: int) -> _Roi:
    box, _score = detection
    xmin, ymin = float(box[1]), float(box[0])
    xmax, ymax = float(box[3]), float(box[2])
    x0 = float(box[4]) * width
    y0 = float(box[5]) * height
    x1 = float(box[6]) * width
    y1 = float(box[7]) * height
    rotation = _normalize_radians(-float(np.arctan2(-(y1 - y0), x1 - x0)))
    return _Roi(
        (xmin + xmax) / 2 * width,
        (ymin + ymax) / 2 * height,
        (xmax - xmin) * width * _RECT_SCALE,
        (ymax - ymin) * height * _RECT_SCALE,
        rotation,
    )


def _normalize_radians(angle: float) -> float:
    return float(angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi)))


def _decode_landmarks(raw: np.ndarray, context) -> np.ndarray | None:
    padding, rect, width, height = context
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    presence = float(values[1434]) if values.size > 1434 else 1.0
    # Граф MediaPipe прогоняет этот выход через сигмоиду. У текущей сети он уже
    # в диапазоне вероятности, повторная сигмоида не должна отбрасывать лицо.
    presence_score = presence if 0.0 <= presence <= 1.0 else float(1 / (1 + np.exp(-np.clip(presence, -100, 100))))
    if presence_score < _MIN_PRESENCE:
        return None
    landmarks = values[:1434].reshape(_NUM_LANDMARKS, 3)
    if np.nanmax(np.abs(landmarks[:, :2])) > 2:
        landmarks = landmarks.copy()
        landmarks[:, 0] /= _LANDMARK_SIZE
        landmarks[:, 1] /= _LANDMARK_SIZE
    left, top, right, bottom = padding
    span_x = 1.0 - left - right
    span_y = 1.0 - top - bottom
    x = (landmarks[:, 0] - left) / span_x
    y = (landmarks[:, 1] - top) / span_y
    matrix = _roi_matrix(rect, width, height)
    image_x = x * matrix[0] + y * matrix[1] + matrix[3]
    image_y = x * matrix[4] + y * matrix[5] + matrix[7]
    return np.column_stack((image_x * width, image_y * height))


def _landmark_output(interpreter) -> np.ndarray:
    outputs = [interpreter.get_tensor(item["index"]) for item in interpreter.get_output_details()]
    return _landmark_rows(outputs)


def _landmark_rows(outputs: list[np.ndarray]) -> np.ndarray:
    # Первый выход — 1434 координаты, второй — уверенность лица.
    flat = [np.asarray(value).reshape(np.asarray(value).shape[0], -1) for value in outputs]
    landmarks = flat[0]
    if landmarks.shape[-1] < 1434 and len(flat) > 1:
        landmarks, flat[1] = flat[1], landmarks
    presence = flat[1] if len(flat) > 1 else np.ones((landmarks.shape[0], 1), np.float32)
    return np.concatenate([landmarks[:, :1434], presence[:, :1]], axis=1)
