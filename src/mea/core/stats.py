"""Spike-train statistics: dispersion, information, STTC, rate aggregation."""
from __future__ import annotations
import numpy as np
from .windows import window_bounds


# --- dispersion ---
def cv(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return np.nan
    m = np.nanmean(x)
    return np.nanstd(x, ddof=1) / m if m else np.nan


def fano_factor(counts):
    """Fano factor: Var(N)/Mean(N) of spike counts."""
    c = np.asarray(counts, float)
    m = np.nanmean(c)
    return np.nanvar(c, ddof=1) / m if m else np.nan


def fano_factor_of_train(t_ms, window_ms, t_range_ms):
    """Fano factor of a train counted in fixed window_ms bins from t_range_ms[0] (partial bin dropped).

    Rate-dependent; window_ms must match across compared values.
    """
    t = np.asarray(t_ms, float)
    t0, t1 = float(t_range_ms[0]), float(t_range_ms[1])
    n = int(np.floor((t1 - t0) / window_ms))
    if n < 2:
        return np.nan
    edges = t0 + np.arange(n + 1) * window_ms
    counts = np.histogram(t, bins=edges)[0]
    return fano_factor(counts)


def fano_factor_operational(t_ms, mean_isi_ms, w_o, t_range_ms, min_windows=10):
    """Operational-time Fano factor, window w_o * mean ISI (Rajdl, Lansky & Kostal 2020).

    NaN when fewer than min_windows windows fit.
    """
    m = float(mean_isi_ms)
    if not np.isfinite(m) or m <= 0:
        return np.nan
    w = float(w_o) * m
    if (float(t_range_ms[1]) - float(t_range_ms[0])) / w < min_windows:
        return np.nan
    return fano_factor_of_train(t_ms, w, t_range_ms)


def cv_squared(x):
    """CV^2 of intervals; equals the Fano factor only asymptotically for renewal trains."""
    v = cv(x)
    return v * v if np.isfinite(v) else np.nan


def lvr(isi):
    """Lv: 3/(n-1) * sum (I_i - I_i+1)^2 / (I_i + I_i+1)^2 over n ISIs (Shinomoto, Shima & Tanji 2003).

    Takes intervals, not timestamps: timestamps drive Lv toward 0.
    """
    x = np.asarray(isi, float)
    if x.size < 3:
        return np.nan
    a, b = x[:-1], x[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = 3.0 * (a - b) ** 2 / (a + b) ** 2
    return float(np.nansum(terms) / (x.size - 1))


# --- information ---
def entropy_bits(counts):
    """Shannon entropy of the empirical distribution of per-bin spike counts."""
    c = np.asarray(counts).astype(np.int64)
    if c.size == 0:
        return np.nan
    p = np.bincount(c - c.min()) / c.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def code_counts(counts):
    """Index-code a binned train -> (n_values, inverse_index), done once outside the pair loop."""
    vals, inv = np.unique(np.asarray(counts), return_inverse=True)
    return vals.size, inv.astype(np.int64)


def mutual_information_bits(code_a, code_b):
    """MI in bits from two pre-coded trains: (n_values, inverse_index)."""
    nx, xi = code_a
    ny, yi = code_b
    if xi.size != yi.size:
        raise ValueError("trains must have the same number of bins")
    n = xi.size
    joint = np.bincount(xi * ny + yi, minlength=nx * ny).reshape(nx, ny) / n
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log2(joint / (px * py))
    return float(np.nansum(np.where(joint > 0, term, 0.0)))


# --- correlation ---
def _sttc_one_direction(ref, other, dt, t0, t1, T):
    """(T, P) for one direction: tiled fraction of [t0, t1] and fraction of ref spikes with a partner."""
    if ref.size == 0:
        return 0.0, 0.0
    if other.size == 0:
        p = 0.0
    else:
        l, r = window_bounds(other, ref - dt, ref + dt)
        p = float((r > l).sum()) / ref.size
    starts = np.maximum(ref - dt, t0)
    ends = np.minimum(ref + dt, t1)
    o = np.argsort(starts)
    starts, ends = starts[o], ends[o]
    run = np.maximum.accumulate(ends)
    new = np.r_[True, starts[1:] > run[:-1]]
    idx = np.flatnonzero(new)
    seg_start = starts[new]
    seg_end = np.maximum.reduceat(ends, idx)
    return float((seg_end - seg_start).sum() / T), p


def sttc(a, b, dt, t_range_ms):
    """Spike-time tiling coefficient (Cutts & Eglen 2014) over an explicit window.

    Explicit window so values compare across pairs; "pair_span" uses the pair's own span (MATLAB legacy).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if isinstance(t_range_ms, str):
        if t_range_ms != "pair_span":
            raise ValueError(f"unknown t_range_ms mode {t_range_ms!r}")
        if a.size == 0 and b.size == 0:
            return np.nan
        allsp = np.concatenate([a, b])
        t0, t1 = float(allsp.min()), float(allsp.max())
    else:
        t0, t1 = float(t_range_ms[0]), float(t_range_ms[1])
    T = t1 - t0
    if T < 0:
        raise ValueError("t_range end precedes start")
    if T == 0:
        return np.nan          # Tfrac = 0/0, as MATLAB
    a = a[(a >= t0) & (a <= t1)]
    b = b[(b >= t0) & (b <= t1)]
    if a.size == 0 or b.size == 0:
        return 0.0
    TA, PA = _sttc_one_direction(a, b, dt, t0, t1, T)
    TB, PB = _sttc_one_direction(b, a, dt, t0, t1, T)

    def term(p, t):
        """(p - t) / (1 - p*t); NaN when the denominator is 0 (undefined, and would raise on floats)."""
        den = 1.0 - p * t
        return np.nan if den == 0 else (p - t) / den

    return float(0.5 * (term(PA, TB) + term(PB, TA)))


# --- aggregation ---
ISI_SPAN, COUNTING = "isi_span", "counting"
_RATE_ALIASES = {"legacy_isi_span": ISI_SPAN}      # old name, still accepted


def rate_mode(compat, default=ISI_SPAN):
    """Firing-rate mode from compat['firing_rate']: isi_span (1/mean ISI, reported) or counting (n/T).

    QC gates use n/T so long-silent units fail on total output.
    """
    v = (compat or {}).get("firing_rate", default)
    return _RATE_ALIASES.get(v, v)


def mean_rate_hz(n_spikes, duration_s):
    """Counting rate n/T, T = analysis window (not spike span)."""
    return np.asarray(n_spikes, float) / float(duration_s)


def population_mean(values, how="arithmetic"):
    """Mean of per-unit values; 'harmonic' of rates equals 1/mean(interval)."""
    v = np.asarray(values, float)
    if how == "arithmetic":
        return float(np.nanmean(v))
    if how == "harmonic":
        return float(np.nansum(~np.isnan(v)) / np.nansum(1.0 / v))
    raise ValueError(how)
