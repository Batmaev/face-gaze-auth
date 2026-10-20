"""Решение о жизненности по конформному рангу стимула.

Выравнивания взгляда нет. Для показанного стимула и для N свежих случайных
стимулов той же длины считаются corr0, inliers_without_fitting и vel_corr —
те же три величины, на которых обучен corr_model.joblib. Скор модели
переводится в 1 − p, где p — конформный p-value нулевой гипотезы
«взгляд не связан с этим стимулом».
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import stimulus
from models import Stimulus


N_CANDIDATES = 499
SCORE_THRESHOLD = 0.98


INLIER_RADIUS = 0.15
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2
_FEATURES = ('corr0', 'inliers_without_fitting', 'vel_corr')

_model_path = Path(__file__).parent / 'corr_model.joblib'
corr_model = joblib.load(_model_path)
if tuple(corr_model.feature_names_in_) != _FEATURES:
    raise RuntimeError(
        f'corr_model.joblib ждёт {tuple(corr_model.feature_names_in_)}, в коде {_FEATURES}'
    )


@dataclass(frozen=True)
class LivenessDecision:
    score: float
    is_live: bool
    # Взгляд после min-max на сравниваемом окне. NaN, где лица не было.
    gaze_x: np.ndarray
    gaze_y: np.ndarray
    # Кадр близок к стимулу по обеим осям. Моргания и пропуски — False.
    inlier: np.ndarray
    # Секунды: candidates — генерация N стимулов, rank — признаки, модель и ранг.
    seconds: dict[str, float]


async def compare_trajectories(
    gaze_x,
    gaze_y,
    stim_x,
    stim_y,
    blink,
    stim: Stimulus,
) -> LivenessDecision:
    """Сравнивает взгляд с показанным стимулом и со свежими случайными.

    Ряды одной длины: stim_x, stim_y — показанный стимул уже в моменты кадров
    взгляда. stim — его описание, по нему генерируются случайные кандидаты
    того же семейства. blink — гарантированные пропуски: в corr0 и в доле
    близких точек их нет, в скорости окно, задевшее моргание, не учитывается.

    score = 1 − p ∈ [0, 1), is_live — score >= 0.96.
    Считается в рабочем потоке: внутри только CPU.
    """
    return await asyncio.to_thread(_compare_trajectories, gaze_x, gaze_y, stim_x, stim_y, blink, stim)


def _compare_trajectories(gaze_x, gaze_y, stim_x, stim_y, blink, stim: Stimulus) -> LivenessDecision:
    gaze_x = np.asarray(gaze_x, dtype=float).reshape(-1)
    gaze_y = np.asarray(gaze_y, dtype=float).reshape(-1)
    stim_x = np.asarray(stim_x, dtype=float).reshape(-1)
    stim_y = np.asarray(stim_y, dtype=float).reshape(-1)
    blink = np.asarray(blink, dtype=bool).reshape(-1)
    n = gaze_x.shape[0]
    if not (gaze_y.shape[0] == stim_x.shape[0] == stim_y.shape[0] == blink.shape[0] == n):
        raise ValueError('взгляд, стимул и моргания разной длины')

    gaze = np.column_stack((_minmax_1d(gaze_x), _minmax_1d(gaze_y)))
    if n < SAVGOL_WINDOW:
        return LivenessDecision(
            score=0.0,
            is_live=False,
            gaze_x=gaze[:, 0],
            gaze_y=gaze[:, 1],
            inlier=np.zeros(n, dtype=bool),
            seconds={'candidates': 0.0, 'rank': 0.0},
        )

    started = time.perf_counter()
    candidates = stimulus.candidates(stim, N_CANDIDATES, n)
    generated = time.perf_counter()

    stims = np.concatenate((np.column_stack((stim_x, stim_y))[None], candidates), axis=0)
    stims[:, :, 0] = _minmax_axis(stims[:, :, 0])
    stims[:, :, 1] = _minmax_axis(stims[:, :, 1])

    features = _features(gaze, stims, blink)
    model_scores = _model_scores(features)
    score = _conformal_score(model_scores[0], model_scores[1:])
    return LivenessDecision(
        score=score,
        is_live=score >= SCORE_THRESHOLD,
        gaze_x=gaze[:, 0],
        gaze_y=gaze[:, 1],
        inlier=_inlier_mask(gaze, stims[0], blink),
        seconds={'candidates': generated - started, 'rank': time.perf_counter() - generated},
    )


def _conformal_score(real: float, candidates: np.ndarray) -> float:
    """1 − p, p = (1 + #{кандидаты со скором ≥ реального}) / (N + 1).

    Нечисловой скор кандидата не считается победой реального стимула,
    знаменатель остаётся N + 1. Нечисловой скор реального — 0.
    """
    if not np.isfinite(real):
        return 0.0
    n_less = np.count_nonzero(np.isfinite(candidates) & (candidates < real))
    return float(n_less / (candidates.shape[0] + 1))


def _model_scores(features: np.ndarray) -> np.ndarray:
    finite = np.isfinite(features).all(axis=1)
    scores = np.full(features.shape[0], np.nan)
    if finite.any():
        frame = pd.DataFrame(features[finite], columns=list(_FEATURES))
        scores[finite] = corr_model.decision_function(frame)
    return scores


def _features(gaze: np.ndarray, stims: np.ndarray, blink: np.ndarray) -> np.ndarray:
    """gaze (T, 2), stims (M, T, 2), уже после min-max. Столбцы — _FEATURES.

    corr0 и доля близких точек — без кадров blink. vel_corr — по всем кадрам,
    но взгляд в моргании заменён на NaN до фильтра, как в lag0_velocity.
    """
    kept = gaze[~blink]
    kept_stims = stims[:, ~blink]
    if kept.shape[0] == 0:
        corr0 = np.full(stims.shape[0], np.nan)
        inliers = np.full(stims.shape[0], np.nan)
    else:
        corr0 = _mean_corr(kept, kept_stims, min_count=2)
        inliers = _inlier_share(kept, kept_stims)

    gaze_for_velocity = gaze.copy()
    gaze_for_velocity[blink] = np.nan
    velocity_gaze = _savgol_xy(gaze_for_velocity, SAVGOL_WINDOW, SAVGOL_POLYORDER)
    velocity_stim = _savgol_batch(stims, SAVGOL_WINDOW, SAVGOL_POLYORDER)
    vel_corr = _mean_corr(velocity_gaze, velocity_stim, min_count=5)
    return np.column_stack((corr0, inliers, vel_corr))


def _mean_corr(reference: np.ndarray, rows: np.ndarray, min_count: int) -> np.ndarray:
    """Среднее корреляций Пирсона по осям. reference (T, 2), rows (M, T, 2)."""
    return (
        _pearson(reference[:, 0], rows[:, :, 0], min_count)
        + _pearson(reference[:, 1], rows[:, :, 1], min_count)
    ) / 2


def _pearson(reference: np.ndarray, rows: np.ndarray, min_count: int) -> np.ndarray:
    """Пирсон reference (T,) с каждой строкой rows (M, T). Пары с NaN пропускаются."""
    paired = np.isfinite(reference)[None, :] & np.isfinite(rows)
    count = paired.sum(axis=1)
    mean_ref = np.sum(np.where(paired, reference[None, :], 0.0), axis=1) / np.maximum(count, 1)
    mean_row = np.sum(np.where(paired, rows, 0.0), axis=1) / np.maximum(count, 1)
    ref_c = np.where(paired, reference[None, :] - mean_ref[:, None], 0.0)
    row_c = np.where(paired, rows - mean_row[:, None], 0.0)
    cov = np.einsum('ij,ij->i', ref_c, row_c)
    var_ref = np.einsum('ij,ij->i', ref_c, ref_c)
    var_row = np.einsum('ij,ij->i', row_c, row_c)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / np.sqrt(var_ref * var_row)
    corr[(count < min_count) | ~(var_ref > 0) | ~(var_row > 0)] = np.nan
    return corr


def _inlier_share(gaze: np.ndarray, stims: np.ndarray) -> np.ndarray:
    """Доля кадров, где обе оси ближе INLIER_RADIUS. NaN в паре — не близко."""
    close = (
        (np.abs(gaze[None, :, 0] - stims[:, :, 0]) < INLIER_RADIUS)
        & (np.abs(gaze[None, :, 1] - stims[:, :, 1]) < INLIER_RADIUS)
    )
    return close.mean(axis=1)


def _inlier_mask(gaze: np.ndarray, stim: np.ndarray, blink: np.ndarray) -> np.ndarray:
    close = (
        (np.abs(gaze[:, 0] - stim[:, 0]) < INLIER_RADIUS)
        & (np.abs(gaze[:, 1] - stim[:, 1]) < INLIER_RADIUS)
        & np.isfinite(gaze).all(axis=1)
        & np.isfinite(stim).all(axis=1)
    )
    close[blink] = False
    return close


def _savgol_xy(xy: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Производная Савицкого — Голея, (T, 2).

    Пропуски на время фильтра заполняются интерполяцией, затем кадр,
    чьё окно задело пропуск, снова становится NaN.
    """
    missing = np.isnan(xy).any(axis=1)
    if missing.all() or int((~missing).sum()) <= window:
        return np.full_like(xy, np.nan)
    t = np.arange(xy.shape[0])
    known = ~missing
    filled = np.column_stack([np.interp(t, t[known], xy[known, i]) for i in range(xy.shape[1])])
    velocity = savgol_filter(filled, window, polyorder, deriv=1, axis=0)
    touched = np.convolve(missing.astype(float), np.ones(window), mode='same') > 0
    velocity[touched] = np.nan
    return velocity


def _savgol_batch(xy: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """_savgol_xy для каждой строки (M, T, 2). Без пропусков — один вызов фильтра."""
    if xy.shape[1] <= window:
        return np.full_like(xy, np.nan)
    missing = np.isnan(xy).any(axis=2)
    if missing.any():
        return np.stack([_savgol_xy(row, window, polyorder) for row in xy])
    return savgol_filter(xy, window, polyorder, deriv=1, axis=1)


def _minmax_axis(values: np.ndarray) -> np.ndarray:
    """Min-max по последней оси. NaN остаётся NaN, постоянный ряд становится NaN."""
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    lo = np.min(np.where(finite, values, np.inf), axis=-1)
    hi = np.max(np.where(finite, values, -np.inf), axis=-1)
    span = hi - lo
    good = (count >= 2) & np.isfinite(span) & (span > 0)
    out = np.full(values.shape, np.nan, dtype=float)
    if np.any(good):
        scaled = (values[good] - lo[good, None]) / span[good, None]
        out[good] = np.where(finite[good], scaled, np.nan)
    return out


def _minmax_1d(values: np.ndarray) -> np.ndarray:
    return _minmax_axis(np.asarray(values, dtype=float)[None])[0]
