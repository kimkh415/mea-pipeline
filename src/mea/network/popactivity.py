"""Population-activity peak detection (port of PopulationActivityDetection.m, refinePeakDetection.m).

Trace = mean over units (caller mask, then a min-rate gate) of each unit's binned train.
"""
from __future__ import annotations
import numpy as np

from ..core.matlabsignal import (gaussian_smooth, filloutliers_clip_percentile,
                                 findpeaks, prctile)
from ..core.matlab_semantics import sec_to_ms, ms_to_sec

SMOOTH_FACTOR = (3, 5, 7)
START_LEVEL = 0.20
END_LEVEL = 0.20
MIN_POINTS = 3
OUTLIER_TOP = 99.5
PERCENTILE_TOP = 95.0
CHAIN = 3
FACTOR_MOUNTAIN = 2.0
BIN_REDUCT = 0.30
MIN_THRESHOLD = 3.0
ACTIVITY_THRESHOLD = 0.05
BIN_FACTOR = (0.25, 0.40, 0.70, 1.0, 1.20)
WINDOW_FACTOR = 0.5
MIN_SAMPLES = 3


def bin_edges(bin_s, t0_s, t1_s):
    """Bin edges t0:bin:t1 in ms; trailing partial bin dropped."""
    t0, t1 = sec_to_ms(t0_s), sec_to_ms(t1_s)
    bin_ms = sec_to_ms(bin_s)
    if (t1 - t0) > bin_ms:
        # match MATLAB colon-operator tolerance (39.999999 bins -> 40)
        span = (t1 - t0) / bin_ms
        tol = 4.0 * np.spacing(max(abs(t0), abs(t1))) / bin_ms
        n = int(np.floor(span + tol))
    else:
        n = 1
    return t0 + np.arange(n + 1) * bin_ms


def _hist(t, edges):
    """histcounts: half-open bins, last right edge inclusive."""
    n = edges.size - 1
    if t.size == 0:
        return np.zeros(n)
    i = np.searchsorted(edges, t, side="right") - 1
    i[t == edges[-1]] = n - 1
    ok = (i >= 0) & (i < n)
    return np.bincount(i[ok], minlength=n).astype(float)


def select_units(trains, mask, t0_s, t1_s, opt_secondary,
                 min_rate_hz=ACTIVITY_THRESHOLD):
    """Indices of masked units passing the min-rate gate (gate skipped for secondary peaks)."""
    keep = np.asarray(mask, bool).copy()
    if not opt_secondary:
        dur = t1_s - t0_s
        for i in np.flatnonzero(keep):
            if trains[i].size / dur < min_rate_hz:
                keep[i] = False
    return np.flatnonzero(keep)


def population_trace(trains, idx, bin_s, t0_s, t1_s, opt_norm):
    """Mean-across-units binned trace and bin midpoints (s); opt_norm divides each unit by its spike count."""
    edges = bin_edges(bin_s, t0_s, t1_s)
    mid = ms_to_sec(edges[:-1] + sec_to_ms(bin_s) / 2.0)
    if idx.size == 0:
        return np.zeros(edges.size - 1), mid

    if opt_norm:
        # silent units excluded from the mean (NaN + omitnan in MATLAB)
        use = [i for i in idx if trains[i].size > 0]
        if not use:
            return np.zeros(edges.size - 1), mid
        acc = np.zeros(edges.size - 1)
        for i in use:
            acc += _hist(trains[i], edges) / trains[i].size
        return acc / len(use), mid

    acc = np.zeros(edges.size - 1)
    for i in idx:
        acc += _hist(trains[i], edges)
    return acc / idx.size, mid


def _mountain_correction(signal, threshold, pct=PERCENTILE_TOP, chain=CHAIN,
                         factor=FACTOR_MOUNTAIN, reduct=BIN_REDUCT, scale=None):
    """Ease the threshold when > factor % of bins lie in runs of >= chain bins above the pct percentile.

    scale (exponent denominator) defaults to factor, as in MATLAB.
    """
    if scale is None:
        scale = factor
    p95 = prctile(signal, pct)
    is_high = signal >= p95
    d = np.diff(np.concatenate(([False], is_high, [False])).astype(int))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    runs = ends - starts + 1
    n_high = int(runs[runs >= chain].sum())
    mountain = 100.0 * n_high / signal.size
    if mountain > factor:
        return threshold * (1 - reduct * (1 - np.exp(-mountain / scale)))
    return threshold


def _edges_of_peaks(smoothed, locs, proms, start_level, end_level):
    """Start/end bins: nearest sub-threshold bin on each side, bounded by neighbouring peaks (else the minimum)."""
    n = locs.size
    start_idx = np.empty(n, int)
    end_idx = np.empty(n, int)
    for k in range(n):
        pk = smoothed[locs[k]]
        thr_ini = pk - (1 - start_level) * proms[k]
        thr_fin = pk - (1 - end_level) * proms[k]

        lo = 0 if k == 0 else locs[k - 1]
        seg = smoothed[lo:locs[k] + 1]
        below = np.flatnonzero(seg < thr_ini)
        start_idx[k] = (below[-1] if below.size else int(np.argmin(seg))) + lo

        hi = smoothed.size if k == n - 1 else locs[k + 1] + 1
        seg = smoothed[locs[k]:hi]
        below = np.flatnonzero(seg < thr_fin)
        end_idx[k] = (below[0] if below.size else int(np.argmin(seg))) + locs[k]
    return start_idx, end_idx


def _p(nb, key, default):
    """Config value from nb, or default."""
    v = (nb or {}).get(key)
    return default if v is None else v


def detect(trains, mask, t_range_s, bin_s, threshold, smooth_add=0,
           opt_norm=True, opt_secondary=False, nb=None):
    """Detect population peaks; returns trace, smoothed trace and one entry per peak.

    The smoothing width yielding the most peaks wins.
    """
    t0_s, t1_s = float(t_range_s[0]), float(t_range_s[1])
    min_rate = float(_p(nb, "min_rate_hz", ACTIVITY_THRESHOLD))
    idx = select_units(trains, mask, t0_s, t1_s, opt_secondary, min_rate)
    spikes, mid = population_trace(trains, idx, bin_s, t0_s, t1_s, opt_norm)

    out = {"spikesBin": spikes, "time": mid, "bin": bin_s, "n_units_used": idx.size}

    smooth = tuple(_p(nb, "smooth_bins", SMOOTH_FACTOR))
    widths = (smooth[0],) if opt_secondary else smooth
    if not opt_secondary:
        threshold = _mountain_correction(
            spikes, threshold,
            float(_p(nb, "mountain_pct", PERCENTILE_TOP)),
            int(_p(nb, "mountain_chain", CHAIN)),
            float(_p(nb, "mountain_factor", FACTOR_MOUNTAIN)),
            float(_p(nb, "bin_reduct", BIN_REDUCT)),
            _p(nb, "mountain_scale", None))
    threshold = max(threshold, float(_p(nb, "min_threshold", MIN_THRESHOLD)))
    outlier_top = float(_p(nb, "outlier_top_pct", OUTLIER_TOP))
    edge = float(_p(nb, "edge_level", START_LEVEL))

    best = None
    smoothed_all = []
    for w in widths:
        sm0 = gaussian_smooth(spikes, w + smooth_add)
        sm = filloutliers_clip_percentile(sm0, 0, outlier_top)
        smoothed_all.append(sm)
        if sm.size < MIN_POINTS:
            continue
        _, locs, proms = findpeaks(sm, min_prominence=float(np.std(sm, ddof=1)) * threshold)
        if locs.size == 0:
            continue
        s_idx, e_idx = _edges_of_peaks(sm, locs, proms, edge, edge)
        # drop an event that starts before the previous one ended
        keep = np.ones(locs.size, bool)
        for k in range(locs.size - 1, 0, -1):
            if s_idx[k] <= e_idx[k - 1]:
                keep[k] = False
        pk = locs[keep]
        st = mid[s_idx[keep]] + bin_s / 2.0
        en = mid[e_idx[keep]] - bin_s / 2.0
        if best is None or pk.size > best[0].size:
            best = (pk, st, en, sm)

    if best is None:
        out.update(nBurst=0, IndivBurst=[], smoothed_spikesBin=smoothed_all[0]
                   if smoothed_all else spikes)
        return out

    pk, st, en, sm = best
    out["smoothed_spikesBin"] = sm
    out["nBurst"] = int(pk.size)
    bursts = []
    for k in range(pk.size):
        r = refine(trains, mask, st[k], en[k], bin_s, opt_norm, nb)
        bursts.append({"peak": r["peak"], "peakAmp": r["peakAmp"],
                       "peakIdx": int(pk[k]), "start": r["start"], "last": r["end"]})
    out["IndivBurst"] = bursts
    return out


def refine(trains, mask, ini0, fin0, bin0, opt_norm, nb=None):
    """Re-measure one event at several bin sizes in a widened window; report medians.

    Uses the mask without the rate gate.
    """
    idx = np.flatnonzero(np.asarray(mask, bool))
    edge = float(_p(nb, "edge_level", START_LEVEL))
    width = fin0 - ini0
    ini = ini0 - width * WINDOW_FACTOR
    fin = fin0 + width * WINDOW_FACTOR

    peak_t, start_t, end_t = [], [], []
    peak_amp = np.nan
    for bf in BIN_FACTOR:
        bin_s = bin0 * bf
        edges = bin_edges(bin_s, ini, fin)
        if edges.size - 1 <= MIN_SAMPLES:
            continue
        spikes, mid = population_trace(trains, idx, bin_s, ini, fin, opt_norm)
        sm = gaussian_smooth(spikes, 3)

        _, locs, proms = findpeaks(sm)
        if locs.size:
            j = int(np.argmax(sm[locs]))
            i_peak, peak, amp = int(locs[j]), float(sm[locs[j]]), float(proms[j])
        else:
            i_peak = int(np.argmax(sm))
            peak = amp = float(sm[i_peak])

        peak_t.append(mid[i_peak])
        if bf == 1.0:
            peak_amp = amp

        seg = sm[:i_peak + 1]
        below = np.flatnonzero(seg < peak - (1 - edge) * amp)
        s = below[-1] if below.size else int(np.argmin(seg))

        seg = sm[i_peak:]
        below = np.flatnonzero(seg < peak - (1 - edge) * amp)
        e = (below[0] if below.size else int(np.argmin(seg))) + i_peak

        start_t.append(mid[s])
        end_t.append(mid[e])

    f = lambda v: float(np.median(v)) if v else np.nan
    return {"peak": f(peak_t), "peakAmp": peak_amp, "start": f(start_t), "end": f(end_t)}
