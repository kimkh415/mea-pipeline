"""Windowed lookups on sorted spike trains; windows are open (lo < t < hi), as MATLAB."""
from __future__ import annotations
import numpy as np


def window_bounds(t, lo, hi):
    """Return (l, r) so t[l[k]:r[k]] holds spikes with lo[k] < t < hi[k]."""
    t = np.asarray(t)
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    return (np.searchsorted(t, lo, side="right"),
            np.searchsorted(t, hi, side="left"))


def counts_in_windows(t, lo, hi):
    """Number of spikes strictly inside each window."""
    l, r = window_bounds(t, lo, hi)
    return r - l


def any_in_windows(t, lo, hi):
    """True where a window holds at least one spike."""
    l, r = window_bounds(t, lo, hi)
    return r > l


def spikes_in_window(t, lo, hi):
    """Spikes strictly inside a single window (a view)."""
    l, r = window_bounds(t, lo, hi)
    return t[l[0]:r[0]]


def restrict(t, t0, t1):
    """Spikes with t0 < t < t1."""
    return t[np.searchsorted(t, t0, "right"):np.searchsorted(t, t1, "left")]
