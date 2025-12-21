from pathlib import Path
import joblib
import numpy as np
import pandas as pd

model_path = Path(__file__).parent / 'linear_model.joblib'
linear_model = joblib.load(model_path)

THRESHOLD = 0.58

x_cols = ['max_corrs', 'inlier_shares', 'lags', 'fit_dists_mean', 'fit_dists_std']

def get_fit_dists(df):
    return pd.Series(np.sqrt((df.gaze_fit_x - df.stim_shift_x)**2 + (df.gaze_fit_y - df.stim_shift_y)**2)).dropna()


def predict(df, drop_blinks):
    if drop_blinks:
        df = df[~df.blink].reset_index(drop=True)

    fit_dists = get_fit_dists(df)

    features = [
        df.max_corr.iloc[0],
        df.inlier.mean(),
        df.lag.iloc[0],
        fit_dists.mean(),
        fit_dists.std()
    ]

    proba = linear_model.predict_proba(np.array(features).reshape(1, -1))[:, 1]
    return proba, proba > THRESHOLD
