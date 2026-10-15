import asyncio
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .fit import align_gaze

model_path = Path(__file__).parent / 'joint_model.joblib'
linear_model = joblib.load(model_path)

THRESHOLD = 0.5


async def align_and_compare_trajectories(df, fps: float) -> tuple[float, bool]:
    """Алгоритм сравнения траекторий, принимающий решение о жизненности.

    df должен иметь колонки:
    - gaze_x, gaze_y (float)
    - stim_x, stim_y (float)
    - blink (bool)

    Возвращает пару proba ∈ [0,1] и result (bool).
    Добавляет новые колонки в df.

    Считается в рабочем потоке: внутри только CPU, и вызов с событийного
    цикла иначе держал бы его до конца.
    """
    return await asyncio.to_thread(_align_and_compare_trajectories, df, fps)


def _align_and_compare_trajectories(df, fps: float) -> tuple[float, bool]:
    normalize_df(df)
    align_gaze(df)

    df = df[~df.blink].reset_index(drop=True)

    features = pd.DataFrame({
        'corr0': [get_corr0(df)],
        'inlier_shares': [df.inlier.mean()],
        'scale': [df.k_x[0] * df.k_y[0]],
        'lags2': [df.lag.iloc[0]**2 / fps**2],
        'fit_dists_mean': [get_fit_dists(df).mean()],
    })

    proba = linear_model.predict_proba(features)[0, 1]
    return proba, proba > THRESHOLD


def normalize_series(s: pd.Series):
    return (s - s.min()) / (s.max() - s.min())

def normalize_df(df: pd.DataFrame):
    df['stim_x'] = normalize_series(df.stim_x)
    df['stim_y'] = normalize_series(df.stim_y)
    df['gaze_x'] = normalize_series(df.gaze_x)
    df['gaze_y'] = normalize_series(df.gaze_y)
    return df


def get_fit_dists(df):
    return pd.Series(np.sqrt((df.gaze_fit_x - df.stim_shift_x)**2 + (df.gaze_fit_y - df.stim_shift_y)**2)).dropna()

def get_corr0(df, method='pearson'):
    corr_x = df.stim_x.corr(df.gaze_x, method=method)
    corr_y = df.stim_y.corr(df.gaze_y, method=method)
    return (corr_x + corr_y) / 2
