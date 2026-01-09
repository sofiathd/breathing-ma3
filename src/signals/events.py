import numpy as np
from scipy.signal import find_peaks

def extract_breath_amplitudes(sig, fps, prominence_factor=0.3, hi_hz=1.0):
    """
    Returns the amplitude (depth) of breaths and their timestamps.
    """
    sig = sig - np.mean(sig)

    min_period_s = 1.0 / hi_hz  
    distance = int(0.6 * min_period_s * fps)
    distance = max(distance, 2)

    peaks, _ = find_peaks(sig, distance=distance, prominence=np.std(sig)*prominence_factor)
    
    valleys, _ = find_peaks(-sig, distance=distance, prominence=np.std(sig)*prominence_factor)
    
    if len(peaks) < 2 or len(valleys) < 2:
        return np.array([]), np.array([])

    t_amp = []
    amp_vals = []
    
    for p in peaks:
        past_valleys = valleys[valleys < p]
        if len(past_valleys) == 0:
            continue
            
        v = past_valleys[-1]
        
        height = sig[p] - sig[v]
        
        t_sec = (p + v) / 2.0 / fps
        
        t_amp.append(t_sec)
        amp_vals.append(height)
        
    return np.array(t_amp), np.array(amp_vals)

def breath_times_from_rf(time_s, rf_bpm):
    """
    Build breath event times from COSMED Rf(t) by integrating instantaneous rate.
    rf_bpm -> breaths/min. Returns breath timestamps (seconds).
    """
    t = np.asarray(time_s, float)
    rf = np.asarray(rf_bpm, float)
    m = np.isfinite(t) & np.isfinite(rf)
    t = t[m]; rf = rf[m]
    if len(t) < 5:
        return np.array([])

    r_hz = rf / 60.0

    cum = np.zeros_like(t)
    dt_ = np.diff(t)
    cum[1:] = np.cumsum(0.5 * (r_hz[1:] + r_hz[:-1]) * dt_)

    n0 = int(np.ceil(cum[0]))
    n1 = int(np.floor(cum[-1]))
    if n1 <= n0:
        return np.array([])

    targets = np.arange(n0, n1 + 1, dtype=float)

    bt = []
    j = 0
    for target in targets:
        while j < len(cum) - 1 and cum[j+1] < target:
            j += 1
        if j >= len(cum) - 1:
            break
        c0, c1 = cum[j], cum[j+1]
        if c1 <= c0:
            continue
        a = (target - c0) / (c1 - c0)
        t_cross = t[j] + a * (t[j+1] - t[j])
        bt.append(t_cross)

    return np.asarray(bt, float)

def match_events_nearest(t_ref, t_vid, y_vid, max_dt=1.0):
    """
    For each ref time, pick nearest video event within max_dt. Else NaN.
    Assumes t_vid sorted.
    """
    t_ref = np.asarray(t_ref, float)
    t_vid = np.asarray(t_vid, float)
    y_vid = np.asarray(y_vid, float)

    if len(t_ref) == 0 or len(t_vid) == 0:
        return np.full(len(t_ref), np.nan)

    out = np.full(len(t_ref), np.nan)
    idx = np.searchsorted(t_vid, t_ref)

    for i, t in enumerate(t_ref):
        cands = []
        if 0 <= idx[i] < len(t_vid):
            cands.append(idx[i])
        if 0 <= idx[i]-1 < len(t_vid):
            cands.append(idx[i]-1)

        best_j = None
        best_dt = None
        for j in cands:
            d = abs(t_vid[j] - t)
            if best_dt is None or d < best_dt:
                best_dt = d
                best_j = j

        if best_j is not None and best_dt <= max_dt:
            out[i] = y_vid[best_j]

    return out

def nearest_dt(t_ref, t_evt):
    t_ref = np.asarray(t_ref, float)
    t_evt = np.asarray(t_evt, float)
    if len(t_evt) == 0:
        return np.full(len(t_ref), np.nan)
    idx = np.searchsorted(t_evt, t_ref)
    out = np.full(len(t_ref), np.nan)
    for i, t in enumerate(t_ref):
        cands = []
        if 0 <= idx[i] < len(t_evt): cands.append(idx[i])
        if 0 <= idx[i]-1 < len(t_evt): cands.append(idx[i]-1)
        out[i] = np.min([abs(t_evt[j]-t) for j in cands]) if cands else np.nan
    return out