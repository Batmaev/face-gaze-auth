"""Операции над компактным описанием стимула (models.Stimulus).

Единственное место, где разбирается тип стимула. Новый тип — новый класс
в models.py и новая ветка в каждой функции здесь; роутер и сравнение
траекторий о типах не знают.
"""

import numpy as np

from models import SplineStimulus, Stimulus
from stimulus.generate import candidate_splines, eval_spline, sample_at_fps


N_CTRL_POINTS = 6


def random_stimulus(fps: float, duration_sec: float) -> Stimulus:
    ctrl = np.random.uniform(0, 1, (N_CTRL_POINTS, 2))
    return SplineStimulus(ctrl=ctrl.tolist(), fps=fps, duration_sec=duration_sec)


def n_frames(stim: Stimulus) -> int:
    return int(round(stim.fps * stim.duration_sec))


def frames(stim: Stimulus) -> tuple[np.ndarray, np.ndarray]:
    """Кадры в [0, 1]² на частоте stim.fps — то, что показывают пользователю."""
    match stim:
        case SplineStimulus():
            n = n_frames(stim)
            return eval_spline(stim.ctrl, np.linspace(0, 1, n), n)
    raise _unsupported(stim)


def sample(stim: Stimulus, gaze_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Стимул в моменты кадров взгляда t = 0, 1/gaze_fps, 2/gaze_fps, … < duration_sec."""
    match stim:
        case SplineStimulus():
            return sample_at_fps(stim.ctrl, n_frames(stim), stim.fps, gaze_fps)
    raise _unsupported(stim)


def candidates(stim: Stimulus, n: int, n_samples: int) -> np.ndarray:
    """n случайных стимулов того же семейства, форма (n, n_samples, 2).

    Нулевое распределение для конформного p-value: кандидаты должны
    генерироваться так же, как показанный стимул.
    """
    match stim:
        case SplineStimulus():
            return candidate_splines(n, n_samples, n_ctrl=len(stim.ctrl))
    raise _unsupported(stim)


def _unsupported(stim) -> TypeError:
    return TypeError(f"неизвестный тип стимула {type(stim).__name__}")
