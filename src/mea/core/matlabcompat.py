"""Small MATLAB conventions that numpy does not share."""
from __future__ import annotations
import numpy as np


def prctile(x, p):
    """MATLAB prctile: point i sits at percentile 100*(i-0.5)/n, clamped."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    pos = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    return float(np.interp(p, pos, x, left=x[0], right=x[-1]))


def isoutlier_percentile(x, lower, upper):
    """MATLAB isoutlier(x,'percentiles',[lower upper])."""
    x = np.asarray(x, float)
    lo = prctile(x, lower) if lower > 0 else -np.inf
    hi = prctile(x, upper) if upper < 100 else np.inf
    return (x < lo) | (x > hi)
