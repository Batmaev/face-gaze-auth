from typing import Literal
from sklearn.linear_model import RANSACRegressor
import numpy as np

def calculate_corrs(df, gaze_col='gaze_x', video_col='video_x', max_lag=100, min_lag=0):
    s1 = df[gaze_col]
    s2 = df[video_col]

    lags = np.arange(min_lag, max_lag+1)

    corrs = [s1.corr(s2.shift(lag)) for lag in lags]
    return lags, np.array(corrs)


def get_lag(df, valid_mask):
    df = df.copy()
    df.loc[~valid_mask, ['gaze_x', 'gaze_y']] = np.nan

    lags, corrs_x = calculate_corrs(df, gaze_col='gaze_x', video_col='stim_x')
    lags, corrs_y = calculate_corrs(df, gaze_col='gaze_y', video_col='stim_y')

    corrs = (corrs_x + corrs_y) / 2

    max_corr_idx = np.argmax(corrs)
    return lags[max_corr_idx], corrs[max_corr_idx]


def ransac_independent(df, lag, threshold=100.0, drop_blinks=False):
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
        (np.arange(len(Xx)) >= lag)
    )

    if drop_blinks:
        valid_mask = valid_mask & ~df['blink'].to_numpy(dtype=bool)

    Xx_valid = Xx[valid_mask].reshape(-1, 1)
    Xy_valid = Xy[valid_mask].reshape(-1, 1)
    yx = yx[valid_mask]
    yy = yy[valid_mask]

    rx = RANSACRegressor(residual_threshold=threshold, stop_probability=1, max_trials=10_000)
    ry = RANSACRegressor(residual_threshold=threshold, stop_probability=1, max_trials=10_000)

    rx.fit(Xx_valid, yx)
    ry.fit(Xy_valid, yy)

    df['gaze_fit_x'] = rx.predict(Xx)
    df['gaze_fit_y'] = ry.predict(Xy)

    df['inlier'] = False
    df.loc[valid_mask, 'inlier'] = rx.inlier_mask_ & ry.inlier_mask_

    return df['inlier']


def align_gaze(df, kind: Literal['independent'] = 'independent', threshold=100.0, n_iterations=1, drop_blinks=False):

    if kind == 'independent':
        ransac = ransac_independent
    else:
        raise ValueError(f"Invalid kind: {kind}")

    # initial guess for lag => find outliers
    inlier_mask = ransac(df, lag=24, threshold=threshold, drop_blinks=drop_blinks)

    for _ in range(n_iterations):
        # get true lag based on inliers
        lag, max_corr = get_lag(df, inlier_mask)

        # get better scale fit
        inlier_mask = ransac(df, lag, threshold=threshold, drop_blinks=drop_blinks)

    df['lag'] = lag
    df['lag'] = df['lag'].astype('category')
    df['max_corr'] = max_corr
    df['max_corr'] = df['max_corr'].astype('category')

    return df



def find_blinks(left_eye_blink, right_eye_blink, threshold=0.8, left_pad=5, right_pad=20):
    raw_blinks = (np.array(left_eye_blink) > threshold) | (np.array(right_eye_blink) > threshold)
    blinks = raw_blinks.copy()
    for i in range(len(raw_blinks)):
        if raw_blinks[i]:
            for j in range(max(0, i - left_pad), min(len(raw_blinks), i + right_pad)):
                blinks[j] = True
    return blinks