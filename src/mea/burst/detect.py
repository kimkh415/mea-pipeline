"""Max-interval single-unit burst detection (port of 02_h01_BurstDetection.m).

strict_legacy=True keeps MATLAB's early-ISI truncation branch inert, as it never yields a burst there.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Bursts:
    start_idx: np.ndarray      # first spike index of each burst
    end_idx: np.ndarray        # last spike index
    t: np.ndarray              # spike train, ms
    label: np.ndarray          # burst id per spike, 0 = not in a burst
    peak: np.ndarray           # peak time per burst, ms

    @property
    def n(self): return self.start_idx.size

    def start_ms(self): return self.t[self.start_idx] if self.n else np.array([])
    def last_ms(self):  return self.t[self.end_idx] if self.n else np.array([])
    def durations_ms(self): return self.last_ms() - self.start_ms()


def detect(t_ms, cfg, strict_legacy=True):
    p = cfg["burst_detection"]
    t = np.asarray(t_ms, float).ravel()
    n = t.size
    empty = Bursts(np.empty(0, int), np.empty(0, int), t, np.zeros(n, int), np.empty(0))
    if n < p["min_spikes"]:
        return empty
    isi = np.diff(t)

    seeds = np.flatnonzero(isi < p["isi_start_ms"])          # 0-based spike index
    starts, ends = [], []
    for b0 in seeds:
        if ends and b0 <= ends[-1]:                          # inside a previous burst
            continue
        e = b0 + p["min_spikes"] - 1
        if e >= n:
            continue
        head = isi[b0:e]                                     # min_spikes-1 intervals
        over = np.flatnonzero(head > p["isi_end_ini_ms"])
        if over.size == 0:
            thr = p["isi_end_ini_ms"] * p["factor_maturation"]
            while e < n - 1 and isi[e] < thr:
                thr = min(thr * p["factor_maturation"], p["isi_end_max_ms"])
                e += 1
        else:
            if strict_legacy:
                continue                                     # inert in MATLAB
            e = b0 + int(over[0])
        body = isi[b0:e]
        if body.size + 1 < p["min_spikes"]:
            continue
        if (t[e] - t[b0]) <= p["min_duration_ms"]:
            continue
        if body.min() >= p["isi_peak_ms"]:
            continue
        starts.append(b0); ends.append(e)

    if not starts:
        return empty
    starts = np.array(starts); ends = np.array(ends)

    # --- merge close bursts (reverse order, as MATLAB) ---
    keep = np.ones(starts.size, bool)
    for i in range(starts.size - 2, -1, -1):
        nxt = i + 1
        while nxt < starts.size and not keep[nxt]:
            nxt += 1
        if nxt >= starts.size:
            continue
        ibi = t[starts[nxt]] - t[ends[i]]
        dur_pre = t[ends[i]] - t[starts[i]]
        dur_post = t[ends[nxt]] - t[starts[nxt]]
        if ibi < max(min(dur_pre, dur_post), p["min_ibi_ms"]):
            ends[i] = ends[nxt]
            keep[nxt] = False
    starts, ends = starts[keep], ends[keep]

    # --- labels and peaks ---
    label = np.zeros(n, int)
    peaks = np.empty(starts.size)
    for j, (b0, e) in enumerate(zip(starts, ends), start=1):
        label[b0:e + 1] = j
        nb = e - b0 + 1
        n_peak = max(p["min_spikes"], int(np.ceil(p["ratio_peak"] * nb)))
        shift = n_peak // 2
        m = nb - n_peak + 1
        if m <= 0 or b0 + n_peak - 1 >= n:
            peaks[j - 1] = t[b0 + shift]
            continue
        k = np.arange(b0, b0 + m)
        widths = t[k + n_peak - 1] - t[k]                    # sliding window of n_peak spikes
        peaks[j - 1] = t[b0 + int(np.argmin(widths)) + shift]
    return Bursts(starts, ends, t, label, peaks)
