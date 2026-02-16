from pathlib import Path
import joblib
import numpy as np
import pandas as pd

model_path = Path(__file__).parent / 'linear_model.joblib'
linear_model = joblib.load(model_path)

THRESHOLD = 0.5

def get_fit_dists(df):
    return pd.Series(np.sqrt((df.gaze_fit_x - df.stim_shift_x)**2 + (df.gaze_fit_y - df.stim_shift_y)**2)).dropna()

def get_corr0(df, method='spearman'):
    corr_x = df.stim_x.corr(df.gaze_x, method=method)
    corr_y = df.stim_y.corr(df.gaze_y, method=method)
    return (corr_x + corr_y) / 2

def predict(df, drop_blinks):
    if drop_blinks:
        df = df[~df.blink].reset_index(drop=True)

    fit_dists = get_fit_dists(df)

    features = pd.DataFrame({
        'corr0_s': [get_corr0(df)],
        'inlier_shares': [df.inlier.mean()],
        'lags': [df.lag.iloc[0]],
        'fit_dists_mean': [fit_dists.mean()],
        # 'scale': [df.k_x[0] * df.k_y[0]]
    })

    proba = linear_model.predict_proba(features)[:, 1]
    return proba, proba > THRESHOLD
