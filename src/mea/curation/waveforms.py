"""Waveform shape metrics and legacy reviewer labels (Classify3BrainWaveform.m G/W/N calls).

New runs use bombcell unitType; labels are kept only to reproduce archived results.
"""
from __future__ import annotations
import json
import numpy as np

GOOD, WEAK, NOISE, UNSET = 1, 2, -1, 0
LABEL_NAMES = {GOOD: "good", WEAK: "weak", NOISE: "noise", UNSET: "unset"}


def mean_waveform(traces, dac_factor, offset_value):
    """Mean and SD of one unit's waveforms, in uV."""
    t = np.atleast_2d(np.asarray(traces, float))
    if t.size == 0:
        return np.array([]), np.array([])
    y = offset_value + dac_factor * t.mean(axis=0)
    dy = (offset_value + dac_factor * t).std(axis=0, ddof=0)
    return y, dy


def time_axis(waveform_length, sampling_rate):
    """Sample times (s), starting at 1/fs."""
    return np.arange(1, int(waveform_length) + 1, dtype=float) / float(sampling_rate)


def shape_metrics(y, sampling_rate):
    """Amplitude, trough-to-peak time, peak/trough ratio and polarity of a mean waveform."""
    y = np.asarray(y, float)
    if y.size < 3:
        return {"trough_uv": np.nan, "peak_uv": np.nan, "amplitude_uv": np.nan,
                "trough_to_peak_ms": np.nan, "peak_trough_ratio": np.nan,
                "inverted": False}
    # largest deflection of either sign, then the opposite extremum after it (handles inverted units)
    i_main = int(np.argmax(np.abs(y)))
    inverted = bool(y[i_main] > 0)
    after = y[i_main:]
    i_opp = i_main + int(np.argmin(after) if inverted else np.argmax(after)) \
        if after.size > 1 else i_main
    trough, peak = (y[i_opp], y[i_main]) if inverted else (y[i_main], y[i_opp])
    return {"trough_uv": float(trough), "peak_uv": float(peak),
            "amplitude_uv": float(abs(y[i_main] - y[i_opp])),
            "trough_to_peak_ms": (i_opp - i_main) / float(sampling_rate) * 1000.0,
            "peak_trough_ratio": float(abs(y[i_opp]) / abs(y[i_main]))
            if y[i_main] else np.nan,
            "inverted": inverted}


class WaveformLabels:
    """Reviewer labels keyed by cluster id (not position), saved as JSON."""

    def __init__(self, labels=None, meta=None):
        self.labels = dict(labels or {})
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path):
        d = json.loads(open(path).read())
        return cls({int(k): int(v) for k, v in d.get("labels", {}).items()},
                   d.get("meta", {}))

    def save(self, path):
        open(path, "w").write(json.dumps(
            {"labels": {str(k): int(v) for k, v in sorted(self.labels.items())},
             "meta": self.meta}, indent=2))
        return path

    @classmethod
    def from_matlab(cls, unit_label):
        """From MATLAB unitLabel (col 0 cluster id, col 1 class)."""
        a = np.atleast_2d(np.asarray(unit_label))
        return cls({int(c): int(v) for c, v in zip(a[:, 0], a[:, 1])})

    def to_array(self, cluster_ids, default=UNSET):
        return np.array([self.labels.get(int(c), default) for c in cluster_ids], int)

    def counts(self):
        out = {}
        for v in self.labels.values():
            out[LABEL_NAMES.get(v, str(v))] = out.get(LABEL_NAMES.get(v, str(v)), 0) + 1
        return out
