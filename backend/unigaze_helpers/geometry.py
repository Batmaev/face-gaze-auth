"""Геометрия нормализации лица из демо UniGaze.

Модель обучена на лице, приведённом к фиксированной камере. Здесь те же шаги:
фиктивная камера кропа, поза по шести точкам, варп и обратный поворот взгляда.
"""

from __future__ import annotations

import cv2
import numpy as np

# Строки 20, 23, 26, 29, 15, 19 из unigaze/data/face_model.txt.
# Порядок: внешний и внутренний угол правого глаза, внутренний и внешний
# левого, правое и левое крыло носа. «Правый» — глаз человека, на кадре он слева.
FACE_POINTS_3D = np.array(
    [
        [-4.65094985961914062e01, -3.83270950317382812e01, 3.64160041809082031e01],
        [-1.77607212066650391e01, -3.25851974487304688e01, 2.90761566162109375e01],
        [1.80439109802246094e01, -3.09568233489990234e01, 2.90629673004150391e01],
        [4.47375869750976562e01, -3.41078720092773438e01, 3.67324371337890625e01],
        [-1.06116647720336914e01, 1.29583492279052734e01, 2.16227645874023438e01],
        [1.12896299362182617e01, 1.38642444610595703e01, 2.18379001617431641e01],
    ],
    dtype=np.float64,
)

# Те же шесть точек в 68-точечной разметке face_alignment: уголки глаз и крылья носа.
FACE_POINT_INDICES = (36, 39, 42, 45, 31, 35)

FOCAL_NORM = 960.0
DISTANCE_NORM = 600.0
ROI_SIZE = (224, 224)
MAX_HEAD_RAD = 80.0 * np.pi / 180.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def dummy_camera(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Камера демо: фокус равен четырём ширинам кадра, центр — главная точка."""
    height, width = image.shape[:2]
    focal = float(width) * 4.0
    camera = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((1, 5), dtype=np.float64)
    return camera, distortion


def crop_face(image: np.ndarray, landmarks: np.ndarray, scale: float = 2.0):
    """Кроп вокруг всех данных точек, как в predict_gaze_video.py."""
    x_min = int(landmarks[:, 0].min())
    x_max = int(landmarks[:, 0].max())
    y_min = int(landmarks[:, 1].min())
    y_max = int(landmarks[:, 1].max())
    width = x_max - x_min
    height = y_max - y_min
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    x0 = max(0, center_x - int(width * scale // 2))
    x1 = min(image.shape[1], center_x + int(width * scale // 2))
    y0 = max(0, center_y - int(height * scale // 2))
    y1 = min(image.shape[0], center_y + int(height * scale // 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = image[y0:y1, x0:x1]
    shifted = landmarks - np.array([x0, y0], dtype=np.float64)
    return crop, shifted


def estimate_head_pose(landmarks: np.ndarray, camera: np.ndarray, distortion: np.ndarray):
    image_points = landmarks.reshape(6, 1, 2).astype(np.float64)
    object_points = FACE_POINTS_3D.reshape(6, 1, 3)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera,
        distortion,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera,
        distortion,
        rvec,
        tvec,
        True,
    )
    if not ok:
        return None
    return rvec, tvec


def face_center_camera(rotation_matrix: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Середина между центром глаз и центром крыльев носа, в СК камеры."""
    posed = rotation_matrix @ FACE_POINTS_3D.T + tvec.reshape(3, 1)
    eyes = posed[:, 0:4].mean(axis=1)
    nose = posed[:, 4:6].mean(axis=1)
    return ((eyes + nose) * 0.5).reshape(3, 1)


def normalize_face(image: np.ndarray, center: np.ndarray, rvec: np.ndarray, camera: np.ndarray):
    """Варп лица в камеру 224×224 и матрица R, которой этот варп сделан."""
    center = center.reshape(3, 1).astype(np.float64)
    distance = float(np.linalg.norm(center))
    if distance < 1e-6:
        return None
    z_scale = DISTANCE_NORM / distance
    camera_norm = np.array(
        [
            [FOCAL_NORM, 0.0, ROI_SIZE[0] / 2.0],
            [0.0, FOCAL_NORM, ROI_SIZE[1] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = np.diag([1.0, 1.0, z_scale])
    head_rotation = cv2.Rodrigues(rvec)[0]
    forward = (center[:, 0] / distance).reshape(3)
    down = np.cross(forward, head_rotation[:, 0])
    down_norm = np.linalg.norm(down)
    if down_norm < 1e-8:
        return None
    down /= down_norm
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, down, forward)).T
    warp = camera_norm @ scale @ rotation @ np.linalg.inv(camera)
    normalized = cv2.warpPerspective(image, warp, ROI_SIZE)
    head_norm = rotation @ head_rotation
    return normalized, rotation, head_norm


def eye_aspect_ratio(
    landmarks: np.ndarray,
    left: tuple[int, ...] = (36, 37, 38, 39, 40, 41),
    right: tuple[int, ...] = (42, 43, 44, 45, 46, 47),
) -> float:
    """Меньший из двух глаз. Ниже примерно 0.2 глаз закрыт. UniGaze это не считает.

    left и right — шесть индексов в порядке EAR: внешний угол, верх, верх,
    внутренний угол, низ, низ.
    """

    def one(index: tuple[int, ...]) -> float:
        eye = landmarks[list(index)]
        vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        if horizontal < 1e-6:
            return 0.0
        return float(vertical / (2.0 * horizontal))

    return min(one(left), one(right))


def head_pose_rejected(head_norm: np.ndarray) -> bool:
    pitch = np.arcsin(np.clip(head_norm[1, 2], -1.0, 1.0))
    yaw = np.arctan2(head_norm[0, 2], head_norm[2, 2])
    return float(np.hypot(pitch, yaw)) > MAX_HEAD_RAD


def pitchyaw_to_vector(pitchyaw: np.ndarray) -> np.ndarray:
    pitch = float(pitchyaw[0])
    yaw = float(pitchyaw[1])
    cos_pitch = np.cos(pitch)
    return np.array(
        [cos_pitch * np.sin(yaw), np.sin(pitch), cos_pitch * np.cos(yaw)],
        dtype=np.float64,
    )


def gaze_in_camera(pitchyaw: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Единичный вектор взгляда в СК камеры кропа. Это R^{-1} от выхода сети."""
    vector = pitchyaw_to_vector(pitchyaw).reshape(3, 1)
    vector = np.linalg.inv(rotation) @ vector
    return (vector / np.linalg.norm(vector)).reshape(3)


def camera_plane_hit(origin: np.ndarray, gaze: np.ndarray) -> np.ndarray | None:
    """Точка, где луч из центра лица пересекает плоскость камеры z = 0.

    gaze — вектор pitchyaw после R^{-1}. При взгляде в камеру он смотрит
    от цели к лицу, поэтому в формуле стоит минус.
    Масштаб — единицы фиктивной камеры, не пиксели и не доли экрана.
    """
    origin = origin.reshape(3)
    gaze = gaze.reshape(3)
    if abs(float(gaze[2])) < 1e-6:
        return None
    ratio_x = float(gaze[0] / gaze[2])
    ratio_y = float(gaze[1] / gaze[2])
    return np.array(
        [
            origin[0] - origin[2] * ratio_x,
            origin[1] - origin[2] * ratio_y,
        ],
        dtype=np.float64,
    )
