from pathlib import Path
import joblib
import numpy as np
import pandas as pd

model_path = Path(__file__).parent / 'linear_model.joblib'
linear_model = joblib.load(model_path)

THRESHOLD = 0.75

def get_fit_dists(df):
    return pd.Series(np.sqrt((df.gaze_fit_x - df.stim_shift_x)**2 + (df.gaze_fit_y - df.stim_shift_y)**2)).dropna()


def predict(df, drop_blinks):
    if drop_blinks:
        df = df[~df.blink].reset_index(drop=True)

    fit_dists = get_fit_dists(df)

    features = pd.DataFrame({
        'max_corrs': [df.max_corr.iloc[0]],
        'inlier_shares': [df.inlier.mean()],
        'inlier_x_shares': [df.inlier_x.mean()],
        'inlier_y_shares': [df.inlier_y.mean()],
        'lags': [df.lag.iloc[0]],
        'fit_dists_mean': [fit_dists.mean()],
        'score_x': [df.score_x.iloc[0]],
        'score_y': [df.score_y.iloc[0]],
    })

    proba = linear_model.predict_proba(features)[:, 1]
    return proba, proba > THRESHOLD
