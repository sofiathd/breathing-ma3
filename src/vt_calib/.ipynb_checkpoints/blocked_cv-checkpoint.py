import numpy as np
from linear import regression_metrics, fit_linear_calibration

def blocked_kfold_indices(n, k=5):
    """
    Splits indices into k contiguous chunks.
    """
    k = int(max(2, k))
    idx = np.arange(n)
    return np.array_split(idx, k)

def cv_calibrate_linear_blocked(x, y, k=5, min_train=20, min_test=5):
    """
    Blocked K-fold CV for y=ax+b.
    Returns dict with mean metrics across folds.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < (min_train + min_test):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "k_used": 0, "n": n}

    folds = blocked_kfold_indices(n, k=k)

    maes, rmses, r2s = [], [], []
    k_used = 0

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if len(train_idx) < min_train or len(test_idx) < min_test:
            continue

        a, b = fit_linear_calibration(x[train_idx], y[train_idx])
        if not np.isfinite(a) or not np.isfinite(b):
            continue

        y_hat = a * x[test_idx] + b
        met = regression_metrics(y[test_idx], y_hat)
        maes.append(met["mae"]); rmses.append(met["rmse"]); r2s.append(met["r2"])
        k_used += 1

    if k_used == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "k_used": 0, "n": n}

    return {
        "mae": float(np.nanmean(maes)),
        "rmse": float(np.nanmean(rmses)),
        "r2": float(np.nanmean(r2s)),
        "k_used": int(k_used),
        "n": int(n),
    }

