"""Population binning: pooled mean-across-units histograms and dyadic bin ladders."""
from __future__ import annotations
import numpy as np


def bin_edges(bin_ms, t_range_s):
    """Edges t0:bin:t1 in ms, trailing partial bin dropped."""
    t0, t1 = t_range_s[0] * 1000.0, t_range_s[1] * 1000.0
    n = int(np.floor((t1 - t0) / bin_ms))
    n = max(n, 1)
    return t0 + np.arange(n + 1) * bin_ms


def histogram(t, bin_ms, t_range_s, weights=None):
    """Spike counts per bin; last right edge inclusive, as histcounts."""
    e = bin_edges(bin_ms, t_range_s)
    idx = np.searchsorted(e, t, side="right") - 1
    n = e.size - 1
    idx[(t == e[-1])] = n - 1                       # histcounts: last edge inclusive
    keep = (idx >= 0) & (idx < n)
    w = None if weights is None else np.asarray(weights)[keep]
    return np.bincount(idx[keep], weights=w, minlength=n)


def pooled_histogram(t_all, unit_of_spike, n_units, bin_ms, t_range_s, normalise=False):
    """Mean per-unit histogram over units with spikes; normalise divides each unit by its spike count."""
    counts_per_unit = np.bincount(unit_of_spike, minlength=n_units)
    active = counts_per_unit > 0
    n_active = int(active.sum())
    if n_active == 0:
        return np.zeros(bin_edges(bin_ms, t_range_s).size - 1)
    if normalise:
        w = 1.0 / counts_per_unit[unit_of_spike] / n_active
    else:
        w = np.full(t_all.size, 1.0 / n_active)
    return histogram(t_all, bin_ms, t_range_s, weights=w)


def dyadic_ladder(h, n_levels):
    """Histograms at bin, 2*bin, 4*bin ... by summing pairs of finer bins (odd tail dropped)."""
    out = [h]
    for _ in range(n_levels - 1):
        n = (h.size // 2) * 2
        if n == 0:
            break
        h = h[:n].reshape(-1, 2).sum(1)
        out.append(h)
    return out


def geometric_bins(bin_min_s, bin_max_s, scaling=2.0, max_it=100):
    """Geometric bin ladder from bin_min_s up to bin_max_s."""
    bins = [bin_min_s]
    n = 0
    while bins[-1] < bin_max_s and n < max_it:
        bins.append(bins[-1] * scaling)
        n += 1
    return np.array(bins)
