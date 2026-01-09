import numpy as np
from scipy.signal import coherence
from preprocess import *

def best_lag_seconds(x_ref_z, x_vid_z, fs, max_lag_s=5.0):
    """Utility for best lag seconds."""
    x_ref_z = np.asarray(x_ref_z, float)
    x_vid_z = np.asarray(x_vid_z, float)
    n = min(len(x_ref_z), len(x_vid_z))
    x_ref_z = x_ref_z[:n]
    x_vid_z = x_vid_z[:n]

    c = np.correlate(x_ref_z, x_vid_z, mode="full")
    lags = np.arange(-n + 1, n)

    max_lag = int(round(max_lag_s * fs))
    keep = (lags >= -max_lag) & (lags <= max_lag)
    c_keep = c[keep]
    l_keep = lags[keep]

    k = int(np.argmax(c_keep))
    return float(l_keep[k] / fs)

def coherence_band_stats(x_ref_z, x_vid_z, fs, lo_hz=0.07, hi_hz=1.0):
    """Utility for coherence band stats."""
    x_ref_z = np.asarray(x_ref_z, float)
    x_vid_z = np.asarray(x_vid_z, float)
    n = min(len(x_ref_z), len(x_vid_z))
    x_ref_z = x_ref_z[:n]
    x_vid_z = x_vid_z[:n]

    nyq = 0.5 * fs
    hi_hz = min(hi_hz, 0.99 * nyq)
    if hi_hz <= lo_hz or n < 64:
        return np.array([]), np.array([]), np.nan, np.nan

    # Key fix:
    # If nperseg == n (single segment), coherence tends to ~1 everywhere.
    # Force multiple segments.
    # Heuristic: aim for ~8 segments, clamp to [64, 512] and <= n//2.
    nperseg = int(np.clip(n // 8, 64, 512))
    nperseg = min(nperseg, max(64, n // 2))
    noverlap = nperseg // 2

    f, Cxy = coherence(x_ref_z, x_vid_z, fs=fs, nperseg=nperseg, noverlap=noverlap)

    band = (f >= lo_hz) & (f <= hi_hz)
    if not np.any(band):
        return f, Cxy, np.nan, np.nan

    c_band = Cxy[band]
    return f, Cxy, float(np.mean(c_band)), float(np.max(c_band))
    
def signal_quality_metrics(ref, vid, fs, lo_hz=0.07, hi_hz=1.0, max_lag_s=5.0, forced_lag_s=None):
    """Utility for signal quality metrics."""
    ref = fill_nans_linear(ref)
    vid = fill_nans_linear(vid)
    n = min(len(ref), len(vid))
    ref = ref[:n]
    vid = vid[:n]

    ref_z = zscore(ref)
    vid_z = zscore(vid)

    corr_z = np.nan
    if np.std(ref_z) > 1e-12 and np.std(vid_z) > 1e-12:
        corr_z = float(np.corrcoef(ref_z, vid_z)[0, 1])
    if forced_lag_s is None:
        if abs(corr_z) < 0.1: 
            lag_s = 0
        else:
            lag_s = best_lag_seconds(ref_z, vid_z, fs=fs, max_lag_s=max_lag_s)
    else:
        lag_s = float(forced_lag_s)
    f, Cxy, coh_mean, coh_peak = coherence_band_stats(ref_z, vid_z, fs=fs, lo_hz=lo_hz, hi_hz=hi_hz)

    return {
        "corr_z": corr_z,
        "best_lag_s": lag_s,
        "coh_mean_band": coh_mean,
        "coh_peak_band": coh_peak,
        "coh_f": f,
        "coh_Cxy": Cxy,
        "ref_z": ref_z,
        "vid_z": vid_z,
    }

def corrcoef_safe(a, b):
    """Utility for corrcoef safe."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])