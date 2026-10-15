import numpy as np
from statsmodels.robust.norms import TukeyBiweight
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools import add_constant


def align_gaze(df, n_iterations=3, initial_lag_guess=0, min_lag=-12, max_lag=12):
    """
    Подбираем пространственное преобразование (калибровку) и сдвиг во времени, чтобы траектории взгляда и стимула максимально совпали.

    df должен иметь колонки:
    - gaze_x, gaze_y (float)
    - stim_x, stim_y (float)
    - blink (bool)

    Добавляем в df колонки:
    - gaze_fit_x, gaze_fit_y (float) — стимул, предсказанный из взгляда: k * gaze + b
    - k_x, k_y, b_x, b_y — коэффициенты этого предсказания
    - inlier, inlier_x, inlier_y (bool)
    - weight_x, weight_y (float) — веса бивеса Тьюки
    - lag (int, повторяющееся число, category)
    - max_corr (float, повторяющееся число, category)

    Возвращаем df.

    Алгоритм:

    1. Делаем догадку о лаге (6 кадров = 100 ms), затем находим выбросы устойчивой регрессией RLM с бивесом Тьюки
    2. Зная выбросы, перебираем лаги и находим тот, при котором корреляция максимальна
    3. Зная лаг, находим более правильный список выбросов
    4. Повторяем `n_iterations` раз

    Явного списка выбросов нет: бивес Тьюки обнуляет вес за своим порогом.
    Для лага берутся точки с ненулевым весом. Если все веса нулевые, берутся все точки.

    Моргания (из колонки `blink`) считаются "гарантированными выбросами"
    """

    inlier_mask = rlm_tukey(df, lag=initial_lag_guess)

    for _ in range(n_iterations):
        lag, max_corr = get_lag(df, inlier_mask, min_lag=min_lag, max_lag=max_lag)
        inlier_mask = rlm_tukey(df, lag)

    df['lag'] = lag
    df['lag'] = df['lag'].astype('category')
    df['max_corr'] = max_corr
    df['max_corr'] = df['max_corr'].astype('category')

    return df


def calculate_corrs(df, gaze_col='gaze_x', stim_col='stim_x', min_lag=0, max_lag=60):
    s1 = df[gaze_col]
    s2 = df[stim_col]

    lags = np.arange(min_lag, max_lag+1)

    corrs = [s1.corr(s2.shift(lag), method='spearman') for lag in lags]
    return lags, np.array(corrs)


def get_lag(df, valid_mask, min_lag=0, max_lag=60):
    df = df.copy()
    df.loc[~valid_mask, ['gaze_x', 'gaze_y']] = np.nan

    lags, corrs_x = calculate_corrs(df, gaze_col='gaze_x', stim_col='stim_x', min_lag=min_lag, max_lag=max_lag)
    lags, corrs_y = calculate_corrs(df, gaze_col='gaze_y', stim_col='stim_y', min_lag=min_lag, max_lag=max_lag)

    corrs = (corrs_x + corrs_y) / 2

    max_corr_idx = np.argmax(corrs)
    return lags[max_corr_idx], corrs[max_corr_idx]


def find_blinks(left_eye_blink, right_eye_blink, threshold=0.8, left_pad=5, right_pad=20):
    raw_blinks = (np.array(left_eye_blink) > threshold) | (np.array(right_eye_blink) > threshold)
    blinks = raw_blinks.copy()
    for i in range(len(raw_blinks)):
        if raw_blinks[i]:
            for j in range(max(0, i - left_pad), min(len(raw_blinks), i + right_pad)):
                blinks[j] = True
    return blinks


def rlm_tukey(df, lag):
    """RLM с бивесом Тьюки: stim_shift = k * gaze + b, отдельно по осям."""
    df[['stim_shift_x', 'stim_shift_y']] = df[['stim_x', 'stim_y']].shift(lag)
    gaze_x = df['gaze_x'].to_numpy(dtype=float)
    gaze_y = df['gaze_y'].to_numpy(dtype=float)
    stim_x = df['stim_shift_x'].to_numpy(dtype=float)
    stim_y = df['stim_shift_y'].to_numpy(dtype=float)

    valid = (
        np.isfinite(gaze_x) &
        np.isfinite(gaze_y) &
        np.isfinite(stim_x) &
        np.isfinite(stim_y) &
        (np.arange(len(df)) >= lag) &
        ~df['blink'].to_numpy(dtype=bool)
    )

    slope_x, intercept_x, weight_x = _fit_rlm_tukey_axis(gaze_x[valid], stim_x[valid])
    slope_y, intercept_y, weight_y = _fit_rlm_tukey_axis(gaze_y[valid], stim_y[valid])

    df['weight_x'] = np.nan
    df['weight_y'] = np.nan
    df.loc[valid, 'weight_x'] = weight_x
    df.loc[valid, 'weight_y'] = weight_y

    for name, value in (
        ('k_x', slope_x), ('k_y', slope_y),
        ('b_x', intercept_x), ('b_y', intercept_y),
    ):
        df[name] = value
        df[name] = df[name].astype('category')

    df['gaze_fit_x'] = slope_x * np.nan_to_num(gaze_x) + intercept_x
    df['gaze_fit_y'] = slope_y * np.nan_to_num(gaze_y) + intercept_y

    inlier_x = np.zeros(len(df), dtype=bool)
    inlier_y = np.zeros(len(df), dtype=bool)
    inlier_x[valid] = _positive_weights(weight_x)
    inlier_y[valid] = _positive_weights(weight_y)
    df['inlier_x'] = inlier_x
    df['inlier_y'] = inlier_y
    df['inlier'] = inlier_x & inlier_y
    return df['inlier']


def _fit_rlm_tukey_axis(x, y):
    exog = add_constant(np.asarray(x, dtype=float), has_constant='add')
    result = RLM(np.asarray(y, dtype=float), exog, M=TukeyBiweight()).fit()
    weights = np.asarray(result.weights, dtype=float).ravel()
    return float(result.params[1]), float(result.params[0]), weights


def _positive_weights(weights):
    inliers = np.asarray(weights) > 0
    if not np.any(inliers):
        return np.ones(len(weights), dtype=bool)
    return inliers
