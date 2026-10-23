"""Network-burst participants (port of findChannel_in_Peak.m, NetworkBurstPlayers.m).

screen: in-event vs overall rate, picks units for the next round; statistics: surrogate test, final set.
"""
from __future__ import annotations
import numpy as np

from ..core.windows import counts_in_windows
from ..core.surrogate import shuffle_burst_times_ms
from ..core.matlab_semantics import sec_to_ms

SAFE_K = 0.40
ACTIVITY_THRESHOLD = 3.0
K_ONSET = 0.20
N_SHUFFLING = 500
STAT_THRESHOLD = 0.95
MIN_PARTICIPATION = 0.20
MIN_PARTICIPATION_ABS = 2.0
ACTIVITY_THRESHOLD_FINAL = 1.90
SHUFFLE_BIN_MAX_MS = 50.0


def _padded_windows(bursts, k):
    """Event windows in ms, widened by a fraction k of their own width."""
    ini = sec_to_ms(np.array([b["start"] for b in bursts], float))
    fin = sec_to_ms(np.array([b["last"] for b in bursts], float))
    pad = (fin - ini) * k
    return ini - pad, fin + pad


def screen(trains, burst_labels, bursts, t_range_s, safe_k=SAFE_K,
           activity_threshold=ACTIVITY_THRESHOLD):
    """Rate screen -> (burst_id, n_bursting, meaningful).

    burst_id[k, ch]: 0 absent, 1 firing, 2 bursting in event k; zeroed for units
    whose in-event rate is not > activity_threshold x overall rate.
    """
    n_ch = len(trains)
    n_pk = len(bursts)
    lo, hi = _padded_windows(bursts, safe_k)
    burst_id = np.zeros((n_pk, n_ch), np.int8)

    # total event duration uses unpadded windows
    win_in = float(sum(b["last"] - b["start"] for b in bursts))
    t0, t1 = sec_to_ms(t_range_s[0]), sec_to_ms(t_range_s[1])
    dur = (t1 - t0) / 1000.0

    meaningful = np.zeros(n_ch, np.int8)
    for ch in range(n_ch):
        t = trains[ch]
        a = np.searchsorted(t, lo, "right")
        b = np.searchsorted(t, hi, "left")
        n_in = b - a
        burst_id[n_in > 0, ch] = 1
        for k in np.flatnonzero(n_in > 0):
            if burst_labels[ch][a[k]:b[k]].sum() > 0:
                burst_id[k, ch] = 2

        spikes_in = int(n_in.sum())
        spikes_tot = int(np.searchsorted(t, t1, "left") - np.searchsorted(t, t0, "right"))
        freq_in = spikes_in / win_in if win_in > 0 else 0.0
        freq_tot = spikes_tot / dur if dur > 0 else 0.0
        if freq_in > freq_tot * activity_threshold:
            meaningful[ch] = 1
        else:
            burst_id[:, ch] = 0

    n_bursting = (burst_id == 2).sum(axis=1)
    return burst_id, n_bursting, meaningful


def robust_participants(burst_id, n_bursting, n_solid,
                        min_phasic=0.6, min_tonic=0.95):
    """Units that participate reliably: bursting units need > min_phasic x n_solid events, others > min_tonic x."""
    n_ch = burst_id.shape[1]
    out = np.zeros(n_ch, bool)
    for ch in range(n_ch):
        n1 = int((burst_id[:, ch] == 1).sum())
        n2 = int((burst_id[:, ch] == 2).sum())
        if n2 > n1:
            out[ch] = n2 > n_solid * min_phasic
        else:
            out[ch] = (n1 + n2) > n_solid * min_tonic
    return out


def _shuffle_plan(t, burst_labels, isi_burst_mean_ms, t_range_s):
    """Bin size, bin count, burst widths (bins) and single-spike count for the surrogate."""
    t0, t1 = sec_to_ms(t_range_s[0]), sec_to_ms(t_range_s[1])
    bin_ms = min(SHUFFLE_BIN_MAX_MS, isi_burst_mean_ms) if np.isfinite(isi_burst_mean_ms) \
        else SHUFFLE_BIN_MAX_MS
    bin_ms = max(bin_ms, 1e-6)
    n_bin = int(round((t1 - t0) / bin_ms))
    ids = burst_labels[burst_labels > 0]
    widths = []
    if ids.size:
        for b in np.unique(ids):
            sel = t[burst_labels == b]
            dur = sel.max() - sel.min() + bin_ms / 2.0
            widths.append(int(np.ceil(dur / bin_ms)))
    n_single = int((burst_labels == 0).sum())
    return bin_ms, n_bin, np.array(widths, int), n_single


def statistics(trains, burst_labels, isi_burst_mean_ms, bursts, t_range_s,
               streams, n_shuffling=N_SHUFFLING, safe_k=SAFE_K,
               activity_final=ACTIVITY_THRESHOLD_FINAL):
    """Surrogate participation test -> (participation, score).

    Participates if it beats STAT_THRESHOLD of burst-shuffled surrogates on event count and clears
    three floors. score[:, 0] = fraction of surrogates beaten, score[:, 1] = events joined.
    """
    n_ch = len(trains)
    n_pk = len(bursts)
    lo1, hi1 = _padded_windows(bursts, K_ONSET)
    lo2, hi2 = _padded_windows(bursts, safe_k)
    t0, t1 = sec_to_ms(t_range_s[0]), sec_to_ms(t_range_s[1])
    dur = (t1 - t0) / 1000.0
    win_in = float(np.sum((hi2 - lo2) / 1000.0))

    participation = np.zeros(n_ch, np.int8)
    score = np.zeros((n_ch, 2))

    for ch in range(n_ch):
        t = trains[ch]
        if t.size == 0:
            continue
        event = int((counts_in_windows(t, lo1, hi1) > 0).sum())
        spikes_in = int(counts_in_windows(t, lo2, hi2).sum())
        spikes_tot = int(np.searchsorted(t, t1, "left") - np.searchsorted(t, t0, "right"))

        bin_ms, n_bin, widths, n_single = _shuffle_plan(
            t, burst_labels[ch], isi_burst_mean_ms[ch], t_range_s)

        rng = streams.for_item("participation", ch)
        hits = np.empty(n_shuffling, int)
        for i in range(n_shuffling):
            s, _ = shuffle_burst_times_ms(rng, t0, bin_ms, n_bin, widths, n_single)
            s.sort()
            hits[i] = int((counts_in_windows(s, lo1, hi1) > 0).sum())

        score[ch, 1] = event
        score[ch, 0] = float((hits < event).sum()) / n_shuffling

        if score[ch, 0] >= STAT_THRESHOLD and n_pk:
            freq_in = spikes_in / win_in if win_in > 0 else 0.0
            freq_tot = spikes_tot / dur if dur > 0 else 0.0
            ratio = freq_in / freq_tot if freq_tot > 0 else np.inf
            if (event / n_pk > MIN_PARTICIPATION and event >= MIN_PARTICIPATION_ABS
                    and ratio > activity_final):
                participation[ch] = 1

    return participation, score


def event_spike_times(trains, bursts, k_pad=SAFE_K, asym=True):
    """Per-event, per-unit spike times in padded windows; returns (times, lo, hi).

    asym pads the right edge twice, as MATLAB NetworkBurstPlayers does.
    """
    ini = sec_to_ms(np.array([b["start"] for b in bursts], float))
    fin = sec_to_ms(np.array([b["last"] for b in bursts], float))
    pad = (fin - ini) * k_pad
    lo = ini - pad
    hi = fin + (pad * 2 if asym else pad)
    out = []
    for t in trains:
        a = np.searchsorted(t, lo, "right")
        b = np.searchsorted(t, hi, "left")
        out.append([t[a[k]:b[k]] for k in range(len(bursts))])
    return out, lo, hi
