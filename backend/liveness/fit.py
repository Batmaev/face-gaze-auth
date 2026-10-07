from sklearn.linear_model import RANSACRegressor
import numpy as np


def align_gaze(df, threshold=100.0, n_iterations=5, initial_lag_guess=6, min_lag=0, max_lag=60):
    """
    Подбираем пространственное преобразование (калибровку) и сдвиг во времени, чтобы траектории взгляда и стимула максимально совпали.

    df должен иметь колонки:
    - gaze_x, gaze_y (float)
    - stim_x, stim_y (float)
    - blink (bool)

    Добавляем в df колонки:
    - gaze_fit_x, gaze_fit_y (float)
    - inlier (bool)
    - lag (int, повторяющееся число, category)
    - max_corr (float, повторяющееся число, category)
    - k_x, k_y (коэффициенты калибровки, category)

    Возвращаем df.

    Алгоритм:

    1. Делаем догадку о лаге (6 кадров = 100 ms), затем находим выбросы с помощью RANSAC из sklearn
    2. Зная выбросы, перебираем лаги и находим тот, при котором корреляция максимальна
    3. Зная лаг, находим более правильный список выбросов
    3. Повторяем `n_iterations` раз

    `threshold` — порог в пикселях для RANSAC; выше — точка записывается в выбросы и не учитывается при подборе преобразования и сдвига во времени

    Моргания (из колонки `blink`) считаются "гарантированными выбросами"
    """

    # initial guess for lag => find outliers
    inlier_mask = ransac_independent(df, lag=initial_lag_guess, threshold=threshold)

    for _ in range(n_iterations):
        # get true lag based on inliers
        lag, max_corr = get_lag(df, inlier_mask, min_lag=min_lag, max_lag=max_lag)

        # get better scale fit
        inlier_mask = ransac_independent(df, lag, threshold=threshold)

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


def ransac_independent(df, lag, threshold=100.0):
    df[['stim_shift_x', 'stim_shift_y']] = df[['stim_x', 'stim_y']].shift(lag)
    Xx = df[['gaze_x']].values
    Xy = df[['gaze_y']].values
    yx = df[['stim_shift_x']].values
    yy = df[['stim_shift_y']].values

    valid_mask = (
        np.isfinite(Xx).ravel() &
        np.isfinite(Xy).ravel() &
        np.isfinite(yx).ravel() &
        np.isfinite(yy).ravel() &
        (np.arange(len(Xx)) >= lag) &
        ~df['blink'].to_numpy(dtype=bool)
    )

    Xx_valid = Xx[valid_mask].reshape(-1, 1)
    Xy_valid = Xy[valid_mask].reshape(-1, 1)
    yx = yx[valid_mask]
    yy = yy[valid_mask]

    rx = RANSACRegressor(residual_threshold=threshold, stop_probability=1, max_trials=10_000)
    ry = RANSACRegressor(residual_threshold=threshold, stop_probability=1, max_trials=10_000)

    rx.fit(Xx_valid, yx)
    ry.fit(Xy_valid, yy)

    df['k_x'] = rx.estimator_.coef_[0][0]
    df['k_y'] = ry.estimator_.coef_[0][0]
    df['k_x'] = df['k_x'].astype('category')
    df['k_y'] = df['k_y'].astype('category')

    df['gaze_fit_x'] = rx.predict(Xx)
    df['gaze_fit_y'] = ry.predict(Xy)

    df['inlier'] = False
    df.loc[valid_mask, 'inlier'] = rx.inlier_mask_ & ry.inlier_mask_

    return df['inlier']
