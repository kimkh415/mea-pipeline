"""Surrogate spike trains (null models for stages 3 and 6).

Per-tick Bernoulli spikes are sampled as Geometric(p) gaps: same distribution, cost scales with spike count.
"""
from __future__ import annotations
import numpy as np


def merge_intervals(lo, hi):
    """Union of possibly-overlapping [lo, hi) intervals, sorted."""
    if len(lo) == 0:
        return np.empty(0, int), np.empty(0, int)
    lo = np.asarray(lo); hi = np.asarray(hi)
    o = np.argsort(lo); lo, hi = lo[o], hi[o]
    keep_lo, keep_hi = [lo[0]], [hi[0]]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= keep_hi[-1]:
            keep_hi[-1] = max(keep_hi[-1], b)
        else:
            keep_lo.append(a); keep_hi.append(b)
    return np.array(keep_lo), np.array(keep_hi)


def constant_rate_segments(n_bin, burst_lo, burst_hi, p_in, p_out):
    """(lo, hi, p) runs of constant per-tick probability; burst bounds are tick indices clipped to [0, n_bin]."""
    lo, hi = merge_intervals(np.clip(burst_lo, 0, n_bin),
                             np.clip(burst_hi, 0, n_bin))
    segs, cur = [], 0
    for a, b in zip(lo, hi):
        if b <= a:
            continue
        if a > cur:
            segs.append((cur, a, p_out))
        segs.append((a, b, p_in))
        cur = b
    if cur < n_bin:
        segs.append((cur, n_bin, p_out))
    return segs


def sample_segment(rng, L, p):
    """Spike tick indices in L Bernoulli(p) ticks, via Geometric(p) gaps."""
    if L <= 0 or p <= 0:
        return np.empty(0, dtype=np.int64)
    if p >= 1:
        return np.arange(L, dtype=np.int64)
    expected = L * p
    n = int(expected + 6.0 * np.sqrt(max(expected, 1.0)) + 16)
    pos = rng.geometric(p, n).cumsum() - 1          # 0-based tick of each spike
    while pos.size and pos[-1] < L:                 # ran short: extend
        more = rng.geometric(p, n).cumsum() + pos[-1]
        pos = np.concatenate([pos, more])
    return pos[pos < L].astype(np.int64)


def sample_surrogate(rng, segments):
    """One surrogate train, as sorted tick indices across all segments."""
    out = [lo + sample_segment(rng, hi - lo, p) for lo, hi, p in segments]
    out = [a for a in out if a.size]
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def surrogate_times_ms(rng, segments, t_step_ms, window_start_ms):
    """Surrogate spike times in ms on the real train's axis (origin window_start_ms)."""
    ticks = sample_surrogate(rng, segments)
    return window_start_ms + ticks * t_step_ms


# --- burst shuffle: relocate intact bursts, then place single spikes ---

def _lazy_permutation(rng, n, need):
    """First `need` entries of a uniform random permutation of range(n) (sampling without replacement)."""
    need = min(need, n)
    if need >= n // 2:                       # dense: cheaper to permute outright
        return rng.permutation(n)[:need]
    seen = set()
    out = np.empty(need, np.int64)
    k = 0
    while k < need:
        cand = rng.integers(0, n, size=max(need - k, 16))
        for c in cand:
            if c not in seen:
                seen.add(c)
                out[k] = c
                k += 1
                if k == need:
                    break
    return out


def shuffle_bursts(rng, n_bin, burst_bins, n_single, max_tries=None):
    """Place bursts (widths in bins) at random free positions, then single spikes; returns (bins, is_burst).

    The scan pointer only moves forward, as MATLAB, so running out of positions ends placement.
    """
    occupied = np.zeros(n_bin, bool)
    kind = np.zeros(n_bin, np.int8)          # 1 = single spike, 2 = burst
    total = int(np.sum(burst_bins)) + int(n_single)
    need = max_tries or min(n_bin, 4 * total + 64)
    order = _lazy_permutation(rng, n_bin, need)

    ptr = 0
    for w in burst_bins:
        while ptr < order.size:
            lo = int(order[ptr])
            hi = min(n_bin, lo + int(w))     # MATLAB's ini+bin_burst, inclusive
            if not occupied[lo:hi + 1].any():
                occupied[lo:hi + 1] = True
                kind[lo:hi + 1] = 2
                break
            ptr += 1

    placed = 0
    while placed < n_single and ptr < order.size:
        b = int(order[ptr])
        if not occupied[b]:
            occupied[b] = True
            kind[b] = 1
            placed += 1
        else:
            ptr += 1

    bins = np.flatnonzero(occupied)
    return bins, kind[bins] == 2


def shuffle_burst_times_ms(rng, window_start_ms, bin_ms, n_bin, burst_bins, n_single):
    """Burst-shuffled surrogate times in ms at bin centres, from window_start_ms."""
    bins, is_burst = shuffle_bursts(rng, n_bin, burst_bins, n_single)
    return window_start_ms + (bins + 0.5) * bin_ms, is_burst
