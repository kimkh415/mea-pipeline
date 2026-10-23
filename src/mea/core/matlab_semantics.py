"""MATLAB numeric conventions (std, mean, unit conversion), kept to reproduce MATLAB numerics."""
from __future__ import annotations
import numpy as np


def std(x, ddof=1):
    """MATLAB std(): 0 for a single element, NaN for empty."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return 0.0
    return float(x.std(ddof=ddof))


def mean(x):
    """MATLAB mean(...,'omitnan'): NaN for empty."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else np.nan


def cv(x):
    """std/mean with MATLAB conventions (0 for a single value, not NaN)."""
    m = mean(x)
    if not np.isfinite(m) or m == 0:
        return np.nan
    return std(x) / m


# x / (1/1000) != x * 1000 in floating point; MATLAB divides, and ceil() can differ
K_MSEC_SEC = 1.0 / 1000.0


def sec_to_ms(x):
    """Seconds -> ms, as MATLAB (x / K_MSEC_SEC)."""
    return np.asarray(x, float) / K_MSEC_SEC if np.ndim(x) else float(x) / K_MSEC_SEC


def ms_to_sec(x):
    """ms -> seconds, as MATLAB (x * K_MSEC_SEC)."""
    return np.asarray(x, float) * K_MSEC_SEC if np.ndim(x) else float(x) * K_MSEC_SEC
