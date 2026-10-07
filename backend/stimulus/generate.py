from functools import wraps

import numpy as np
import scipy.interpolate as si


def normalized(func):
    """Нормализация в [0, 1]×[0, 1]"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        x, y = func(*args, **kwargs)
        x -= x.min()
        y -= y.min()
        x /= x.max()
        y /= y.max()
        return x, y
    return wrapper


@normalized
def random_spline(n_frames=600, n_ctrl=6) -> tuple[np.ndarray, np.ndarray]:
    """Генерирует траекторию стимула (случайный сплайн)"""
    ctrl = np.random.uniform(0, 1, (n_ctrl, 2))
    tck, u = si.splprep(ctrl.T, s=0)
    u_fine = np.linspace(0, 1, n_frames)
    x, y = si.splev(u_fine, tck)
    return x, y
