"""MATLAB smoothdata, filloutliers, prctile and findpeaks, reproduced exactly so peak positions match MATLAB."""
from __future__ import annotations
import numpy as np


def prctile(x, p):
    """MATLAB prctile: point i at percentile 100*(i-0.5)/n, clamped; NaNs dropped.

    Differs from numpy's default i/(n-1) placement.
    """
    x = np.asarray(x, float)
    x = np.sort(x[~np.isnan(x)])
    n = x.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    q = (np.arange(1, n + 1) - 0.5) / n * 100.0
    return np.interp(np.asarray(p, float), q, x)


def gaussian_smooth(x, window):
    """MATLAB smoothdata(x, 'gaussian', w): sigma = w/5, kernel truncated and renormalised at edges."""
    x = np.asarray(x, float)
    n = x.size
    w = int(window)
    if w <= 1 or n == 0:
        return x.copy()
    half = (w - 1) / 2.0
    sigma = w / 5.0
    offs = np.arange(-np.floor(half), np.floor(half) + 1) if w % 2 else \
           np.arange(-w / 2 + 1, w / 2 + 1)
    k = np.exp(-0.5 * (offs / sigma) ** 2)

    # edge renormalisation: divide by the kernel mass inside the array
    num = np.convolve(x, k[::-1], mode="full")
    den = np.convolve(np.ones(n), k[::-1], mode="full")
    lo = int(-offs[0])
    return num[lo:lo + n] / den[lo:lo + n]


def filloutliers_clip_percentile(x, lo_pct, hi_pct):
    """MATLAB filloutliers(x, "clip", "percentiles", [lo hi])."""
    x = np.asarray(x, float).copy()
    lo = prctile(x, lo_pct)
    hi = prctile(x, hi_pct)
    return np.clip(x, lo, hi)


def _local_maxima(y):
    """Strict local maxima as findpeaks: endpoints ignored, first sample of a flat top."""
    n = y.size
    if n < 3:
        return np.array([], dtype=int)
    # collapse flat runs to their first sample, then find strict maxima
    keep = np.concatenate(([True], y[1:] != y[:-1]))
    idx = np.flatnonzero(keep)
    z = y[idx]
    if z.size < 3:
        return np.array([], dtype=int)
    hit = np.flatnonzero((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:])) + 1
    return idx[hit]


def _prominence(y, loc):
    """findpeaks prominence: peak minus the higher of the two minima before the signal exceeds the peak."""
    n = y.size
    p = y[loc]

    i = loc - 1
    left_min = p
    while i >= 0 and y[i] <= p:
        left_min = min(left_min, y[i])
        i -= 1
    if i < 0:
        left_min = min(left_min, y[:loc + 1].min())

    j = loc + 1
    right_min = p
    while j < n and y[j] <= p:
        right_min = min(right_min, y[j])
        j += 1
    if j >= n:
        right_min = min(right_min, y[loc:].min())

    return p - max(left_min, right_min)


def _prominences(y, locs):
    """Prominence of every peak."""
    return np.array([_prominence(y, int(i)) for i in locs])


def findpeaks(y, min_prominence=None):
    """MATLAB findpeaks(y, 'MinPeakProminence', p) -> (pks, locs, proms); locs 0-based, no widths."""
    y = np.asarray(y, float)
    locs = _local_maxima(y)
    if locs.size == 0:
        return np.array([]), np.array([], dtype=int), np.array([])
    proms = _prominences(y, locs)
    if min_prominence is not None:
        keep = proms >= min_prominence
        locs, proms = locs[keep], proms[keep]
    return y[locs], locs, proms
