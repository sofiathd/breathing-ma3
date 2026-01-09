import numpy as np
from scipy.signal import detrend, welch, find_peaks

def _parabolic_peak_interp(x0, y):
    """
    Parabolic interpolation around a discrete peak at index x0.
    y is 1D array. Returns (x_vertex, y_vertex) in index units.
    """
    if x0 <= 0 or x0 >= len(y) - 1:
        return float(x0), float(y[x0])
    y1, y2, y3 = float(y[x0-1]), float(y[x0]), float(y[x0+1])
    denom = (y1 - 2*y2 + y3)
    if abs(denom) < 1e-12:
        return float(x0), float(y2)
    delta = 0.5 * (y1 - y3) / denom   # vertex offset from x0
    xv = float(x0) + delta
    # vertex value (optional)
    yv = y2 - 0.25*(y1 - y3)*delta
    return xv, float(yv)

def pick_fundamental_harmonic_safe(f_hz, P, lo_hz, hi_hz):
    """
    Given PSD (f_hz, P), return a frequency with harmonic safety:
    if the top peak is likely a harmonic, prefer f/2 or f/3 if strong.
    """
    band = (f_hz >= lo_hz) & (f_hz <= hi_hz)
    f = f_hz[band]
    p = P[band]
    if len(f) < 5:
        return np.nan

    k = int(np.argmax(p))
    f1 = float(f[k])

    # candidates: f1, f1/2, f1/3 (clipped to band)
    cands = [f1]
    if f1/2 >= lo_hz: cands.append(f1/2)
    if f1/3 >= lo_hz: cands.append(f1/3)

    # score candidate by local power around it (Â± one bin)
    def local_power(freq):
        """Utility for local power."""
        j = int(np.argmin(np.abs(f - freq)))
        j0 = max(0, j-1); j1 = min(len(p), j+2)
        return float(np.sum(p[j0:j1]))

    scores = [(local_power(fc), fc) for fc in cands]
    scores.sort(reverse=True)

    # prefer lower freq if scores are close (harmonics often slightly higher)
    best_score, best_f = scores[0]
    for sc, fc in scores[1:]:
        if sc >= 0.85 * best_score:   # close -> choose lower freq
            best_f = min(best_f, fc)
    return float(best_f)


def estimate_rr_robust(sig, fs, lo_hz=0.07, hi_hz=1.0,
                       win_s=20.0, hop_s=2.0,
                       nperseg_max=512, min_valid=6):
    """Estimate rr robust."""
    sig = np.asarray(sig, float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < int(win_s * fs):
        return np.nan

    win = int(win_s * fs)
    hop = int(hop_s * fs)
    if hop < 1: hop = 1

    rr_list = []
    for start in range(0, len(sig) - win + 1, hop):
        x = detrend(sig[start:start+win])

        nperseg = min(nperseg_max, max(64, win // 4))
        noverlap = nperseg // 2
        nfft = int(max(16384, 2 ** int(np.ceil(np.log2(nperseg * 8)))))

        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        f0 = pick_fundamental_harmonic_safe(f, Pxx, lo_hz, hi_hz)
        if np.isfinite(f0):
            rr_list.append(60.0 * f0)

    if len(rr_list) < min_valid:
        return np.nan
    return float(np.median(rr_list))

def estimate_rr_single(sig, fs, lo_hz=0.07, hi_hz=1.0,
                       min_seconds=12.0, nfft_min=16384):
    """Estimate respiratory rate."""
    sig = np.asarray(sig, float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < max(64, int(min_seconds * fs)):
        return np.nan

    x = detrend(sig)
    n = len(x)

    # Force multiple segments (avoid nperseg == n)
    nperseg = int(np.clip(n // 8, 64, 512))
    nperseg = min(nperseg, max(64, n // 2))
    noverlap = nperseg // 2

    nfft = int(max(nfft_min, 2 ** int(np.ceil(np.log2(nperseg * 8)))))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    band = (f >= lo_hz) & (f <= hi_hz)
    if not np.any(band):
        return np.nan

    f_band = f[band]
    P_band = Pxx[band]
    k = int(np.argmax(P_band))

    y = np.log(np.maximum(P_band, 1e-30))
    kv, _ = _parabolic_peak_interp(k, y)

    df = float(f_band[1] - f_band[0]) if len(f_band) > 1 else np.nan
    f0 = float(f_band[0] + kv * df) if np.isfinite(df) else float(f_band[k])
    return 60.0 * f0

