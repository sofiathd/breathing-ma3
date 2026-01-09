import pandas as pd
import numpy as np
import datetime as dt
import os

def _t_to_seconds(val):
    """Convert time to seconds."""
    if pd.isna(val): return np.nan
    if isinstance(val, dt.time): return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond * 1e-6
    if isinstance(val, pd.Timedelta): return val.total_seconds()
    if isinstance(val, dt.datetime): return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond * 1e-6
    td = pd.to_timedelta(str(val), errors="coerce")
    return td.total_seconds() if pd.notna(td) else np.nan

def load_cosmed_take(xlsx_path, take, sheet="Data", resample_dt_s=None):
    """Load cosmed take from excel file path and take identifier."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet)
    m = raw["Marker"].astype(str).fillna("").str.strip()
    idx = raw.index[m == take].to_numpy()
    if len(idx) == 0: raise ValueError(f"No rows found with Marker == '{take}'")
    start, end = int(idx[0]), int(idx[0])
    while end < len(raw) and str(raw.loc[end, "Marker"]).strip() == take: end += 1
    
    seg = raw.loc[start:end-1, ["t", "Rf", "VT", "VE"]].copy()
    seg["time_abs_s"] = seg["t"].apply(_t_to_seconds).astype(float)
    for c in ["Rf", "VT", "VE"]: seg[c] = pd.to_numeric(seg[c], errors="coerce")
    seg = seg.dropna(subset=["time_abs_s", "Rf", "VT", "VE"]).sort_values("time_abs_s")
    
    t0 = float(seg["time_abs_s"].iloc[0])
    seg["time_s"] = seg["time_abs_s"] - t0
    
    if resample_dt_s is None: return seg[["time_s", "Rf", "VT", "VE"]].reset_index(drop=True)
    grid = np.arange(0.0, float(seg["time_s"].iloc[-1]) + 1e-9, float(resample_dt_s))
    out = pd.DataFrame({"time_s": grid})
    for c in ["Rf", "VT", "VE"]:
        out[c] = np.interp(out["time_s"].to_numpy(), seg["time_s"].to_numpy(), seg[c].to_numpy())
    return out
