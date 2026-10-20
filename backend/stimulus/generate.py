"""Сплайновая траектория: кубический сплайн через контрольные точки.
"""

import numpy as np
import scipy.interpolate as si


def eval_spline(ctrl, u, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Сплайн в параметре u ∈ [0, 1], в той же нормировке, что и кадры.

    Нормировка в [0, 1]×[0, 1] считается по равномерной сетке из n_frames
    и применяется к любой другой выборке. Иначе при другом FPS минимум и максимум
    съехали бы, и траектория не совпала бы с показанной.
    """
    ctrl = np.asarray(ctrl, dtype=float)
    if ctrl.ndim != 2 or ctrl.shape[1] != 2 or len(ctrl) < 4:
        raise ValueError("ctrl должен быть массивом формы (n, 2), n ≥ 4")
    tck, _ = si.splprep(ctrl.T, s=0)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    u_ref = np.linspace(0, 1, n_frames)
    x_ref, y_ref = si.splev(u_ref, tck)
    if u.shape == u_ref.shape and np.array_equal(u, u_ref):
        x, y = x_ref, y_ref
    else:
        x, y = si.splev(u, tck)
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    x = (np.asarray(x, dtype=float) - x_ref.min()) / (x_ref.max() - x_ref.min())
    y = (np.asarray(y, dtype=float) - y_ref.min()) / (y_ref.max() - y_ref.min())
    return x, y


def sample_count(n_frames: int, stim_fps: float, gaze_fps: float) -> int:
    """Сколько отсчётов стимула приходится на частоту взгляда.

    Целый шаг stim[::step] даёт ровно это число: индексы 0, step, 2·step, … < n_frames.
    """
    step = stim_fps / gaze_fps
    return int(np.floor((n_frames - 1e-9) / step)) + 1


def sample_at_fps(ctrl, n_frames: int, stim_fps: float, gaze_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Кадры стимула в моменты взгляда: t = 0, 1/gaze_fps, 2/gaze_fps, …"""
    step = stim_fps / gaze_fps
    frame_index = np.arange(sample_count(n_frames, stim_fps, gaze_fps)) * step
    # Кадр i показан в момент i / stim_fps и лежит на параметре i / (n_frames - 1).
    u = frame_index / (n_frames - 1)
    return eval_spline(ctrl, u, n_frames)


def candidate_splines(n: int, n_frames: int, n_ctrl: int = 6) -> np.ndarray:
    """N случайных стимулов сразу на частоте взгляда, форма (n, n_frames, 2).

    Тот же кубический сплайн, что у показанного стимула, но сетка сразу из n_frames
    точек, без прореживания 60 FPS.
    """
    if n_frames < 2:
        raise ValueError("у стимула-кандидата должно быть хотя бы 2 отсчёта")
    u = np.linspace(0, 1, n_frames)
    out = np.empty((n, n_frames, 2), dtype=float)
    rng = np.random.default_rng()
    for i in range(n):
        ctrl = rng.uniform(0, 1, (n_ctrl, 2))
        out[i, :, 0], out[i, :, 1] = eval_spline(ctrl, u, n_frames)
    return out
