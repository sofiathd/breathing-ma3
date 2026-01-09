import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(sig, fs, lo_hz=0.08, hi_hz=1.0, order=2):
    """Band-pass filter a 1D signal with a Butterworth filter."""
    nyq = 0.5 * fs
    hi_hz = min(hi_hz, 0.99 * nyq)
    lo_hz = max(lo_hz, 0.001)
    if hi_hz <= lo_hz:
        return sig  # fallback: no filtering
    b, a = butter(N=order, Wn=[lo_hz/nyq, hi_hz/nyq], btype="bandpass")
    return filtfilt(b, a, sig)

def interp_extrapolate(x_new, x, y):
    """Interpolate y(x) onto x_new with edge extrapolation."""
    x_new, x, y = np.asarray(x_new), np.asarray(x), np.asarray(y)
    out = np.interp(x_new, x, y)
    return out

def fill_nans_linear(x):
    """Utility for fill nans linear."""
    x = np.asarray(x, float).copy()
    good = np.isfinite(x)
    if good.all():
        return x
    if not np.any(good):
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x

def zscore(x, eps=1e-12):
    """Z-score normalize a 1D signal."""
    x = np.asarray(x, float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd < eps:
        return x * 0.0
    return (x - mu) / sd

def align_video_waveform_to_ref(df_run_time_s, t_video_s, sig_video, lag_s):
    """
    Returns video waveform sampled on df_run_time_s AFTER shifting time by lag_s.
    Interpretation: we sample video at (t + lag_s) to best align with reference at t.
    """
    df_run_time_s = np.asarray(df_run_time_s, float)
    t_video_s = np.asarray(t_video_s, float)
    sig_video = np.asarray(sig_video, float)

    # allow extrap with edge values to avoid NaNs at edges
    y = np.interp(df_run_time_s + lag_s, t_video_s, sig_video, left=sig_video[0], right=sig_video[-1])
    return y
