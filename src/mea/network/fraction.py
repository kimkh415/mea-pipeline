"""Default network-burst detector: >= frac_min of units fire within window_s.

Fixed threshold, unlike Yang's std-of-trace bar which bursts raise themselves.
20% follows Pasquale et al. 2010; events < merge_s apart are merged.
"""
from __future__ import annotations
import numpy as np

FRAC_MIN = 0.20
WINDOW_S = 0.100
STEP_S = 0.010
MERGE_S = 1.000
BACKGROUND_MAX = 0.10


def fraction_trace(trains_ms, t0_s, t1_s, window_s=WINDOW_S, step_s=STEP_S):
    """Fraction of units with >= 1 spike in each sliding window; returns (centres s, fraction)."""
    n = len(trains_ms)
    starts = np.arange(t0_s, t1_s - window_s, step_s)
    active = np.zeros(starts.size)
    for t in trains_ms:
        t = np.asarray(t, float) / 1000.0
        if t.size == 0 or starts.size == 0:
            continue
        j = np.searchsorted(t, starts, side="left")
        nxt = np.where(j < t.size, t[np.minimum(j, t.size - 1)], np.inf)
        active += nxt < starts + window_s
    return starts + window_s / 2.0, active / max(n, 1)


def detect(trains_ms, t_range_s, frac_min=FRAC_MIN, window_s=WINDOW_S,
           step_s=STEP_S, merge_s=MERGE_S, background_max=BACKGROUND_MAX):
    """Events where >= frac_min of units fire within window_s.

    Same output fields as detect.run; peakAmp is the peak fraction, background the median fraction.
    """
    t0, t1 = float(t_range_s[0]), float(t_range_s[1])
    tm, f = fraction_trace(trains_ms, t0, t1, window_s, step_s)
    above = f >= frac_min
    d = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    runs = []
    for a, b in zip(starts, ends):
        if runs and tm[a] - tm[runs[-1][1]] < merge_s:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    events = []
    for a, b in runs:
        k = a + int(np.argmax(f[a:b + 1]))
        events.append({"peak": float(tm[k]), "peakAmp": float(f[k]), "peakIdx": int(k),
                       "start": float(tm[a] - window_s / 2.0),
                       "last": float(tm[b] + window_s / 2.0)})
    background = float(np.median(f)) if f.size else np.nan
    return {"IndivBurst": events, "nBurst": len(events), "bin": float(window_s),
            "spikesBin": f, "time": tm, "trace_label": "fraction of units active",
            "threshold": float(frac_min), "background": background,
            "background_flag": bool(np.isfinite(background) and background > background_max)}
