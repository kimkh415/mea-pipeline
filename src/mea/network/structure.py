"""Network-burst temporal structure: per-unit onset, peak, tail and onset rank (port of NetworkBurstStructure.m).

Events are aligned on the 30 ms bin with most active units; unit profiles use 10 ms bins.
"""
from __future__ import annotations
import numpy as np

from ..core.matlab_semantics import sec_to_ms

BIN_SIZE_POP = 30.0
BIN_NEURON = 10.0
THRESHOLD_ACT = 0.4
THRESHOLD_ACT_TWIN = 0.25
THRESHOLD_TAIL = 0.25


def structure(bursts, event_times, participants, member_mask, positions_xy):
    """Per-family population and per-unit firing profiles plus onset/peak/tail table."""
    member = np.asarray(member_mask, bool)
    idx_event = np.flatnonzero(member)
    n_burst = idx_event.size
    n_part = len(participants)
    if n_burst == 0 or n_part == 0:
        return {"populationFiring": [], "neuronFiring": np.zeros((0, 0, 5)),
                "structure": np.zeros((0, 4)), "mapPosition": np.zeros((0, 2)),
                "correlation": np.nan}

    pop, peak_burst = [], np.full(n_burst, np.nan)
    left, right = 0.0, 0.0
    for k, idx in enumerate(idx_event):
        ini = sec_to_ms(bursts[idx]["start"])
        fin = sec_to_ms(bursts[idx]["last"])
        n_bins = int(np.ceil((fin - ini) / BIN_SIZE_POP))
        s = np.zeros((n_bins, 5))
        s[:, 1] = ini + np.arange(n_bins) * BIN_SIZE_POP
        s[:, 2] = s[:, 1] + BIN_SIZE_POP
        s[:, 3] = (s[:, 1] + s[:, 2]) / 2.0
        for ch in participants:
            t = np.asarray(event_times[ch][idx], float)
            if t.size == 0:
                continue
            t = np.sort(t)
            a = np.searchsorted(t, s[:, 1], "left")     # >= left edge
            b = np.searchsorted(t, s[:, 2], "left")     # <  right edge
            n = b - a
            s[:, 0] += n
            s[:, 4] += (n > 0)
        pop.append(s)
        peak_burst[k] = s[int(np.argmax(s[:, 4])), 3]
        left = min(left, ini - peak_burst[k])
        right = max(right, fin - peak_burst[k])

    nbl = int(np.ceil(abs(left) / BIN_NEURON))
    nbr = int(np.ceil(abs(right) / BIN_NEURON))
    n_tot = nbl + nbr
    sig = np.zeros((n_part, n_tot, 5))
    if n_tot == 0:
        return {"populationFiring": pop, "neuronFiring": sig,
                "structure": np.full((n_part, 4), np.nan),
                "mapPosition": np.array([positions_xy[c] for c in participants], float) + 1,
                "correlation": np.nan}

    # bin i = 1..n_tot starts at peak - (nbl - i + 0.5)*BIN
    offs = -(nbl - (np.arange(n_tot) + 1) + 0.5) * BIN_NEURON
    for k, idx in enumerate(idx_event):
        lo = peak_burst[k] + offs
        hi = lo + BIN_NEURON
        sig[:, :, 0] = (lo + hi) / 2.0 - peak_burst[k]
        for j, ch in enumerate(participants):
            t = np.asarray(event_times[ch][idx], float)
            if t.size == 0:
                continue
            t = np.sort(t)
            n = np.searchsorted(t, hi, "left") - np.searchsorted(t, lo, "left")
            sig[j, :, 1] += n
            sig[j, :, 2] += (n > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        tot = sig[:, :, 1].sum(axis=1, keepdims=True)
        hgt = sig[:, :, 1].max(axis=1, keepdims=True)
        sig[:, :, 3] = sig[:, :, 1] / tot
        sig[:, :, 4] = sig[:, :, 1] / hgt

    # mean |correlation| between unit profiles
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = np.corrcoef(sig[:, :, 1])
    iu = np.triu_indices(n_part, 1)
    corr = float(np.mean(np.abs(cm[iu]))) if n_part > 1 else np.nan

    # --- onset, peak, tail per unit ---
    thr = THRESHOLD_ACT * n_burst
    thr_twin = THRESHOLD_ACT_TWIN * n_burst
    onset = np.zeros(n_part, int)
    for j in range(n_part):
        act = sig[j, :, 2]
        for b in range(n_tot - 1):
            if act[b] >= thr:
                onset[j] = b + 1                      # 1-based, as MATLAB
                break
            if act[b] >= thr_twin and act[b + 1] >= thr_twin:
                onset[j] = b + 2
                break

    ranks = np.unique(onset[onset > 0])
    out = np.full((n_part, 4), np.nan)
    for j in range(n_part):
        if onset[j] > 0:
            out[j, 0] = int(np.flatnonzero(ranks == onset[j])[0]) + 1
            out[j, 1] = (onset[j] - nbl - 0.5) * BIN_NEURON
        prof = sig[j, :, 1]
        pk = int(np.argmax(prof))
        out[j, 2] = (pk + 1 - nbl - 0.5) * BIN_NEURON
        # last bin above THRESHOLD_TAIL x peak; minus one (0 if none) to match MATLAB
        umbral = prof[pk] * THRESHOLD_TAIL
        tail = n_tot
        while tail > 0 and not prof[tail - 1] > umbral:
            tail -= 1
        tail = tail - 1 if tail > 0 else 0
        out[j, 3] = (tail - nbl - 0.5) * BIN_NEURON

    return {"populationFiring": pop, "neuronFiring": sig, "structure": out,
            "mapPosition": np.array([positions_xy[c] for c in participants], float) + 1,
            "correlation": corr}
