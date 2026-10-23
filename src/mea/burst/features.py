"""Single-unit burst features (port of 02_h02_BurstFeatures.m).

Each spike's ISI is the one following it (NaN for the last), as in MATLAB; non-burst ISI stats depend on it.
"""
from __future__ import annotations
import numpy as np
from ..core import matlab_semantics as ml


def features(b, cfg=None):
    t, lab = b.t, b.label
    n_spike = t.size
    isi_after = np.r_[np.diff(t), np.nan] if n_spike else np.array([])

    out = {"n_burst": b.n, "n_spikes": n_spike}

    # --- spikes outside bursts ---
    out_mask = lab == 0
    isi_out = isi_after[out_mask]
    out["nonburst_n_norm"] = float(out_mask.sum()) / n_spike if n_spike else np.nan
    out["nonburst_isi_mean"] = ml.mean(isi_out) if isi_out.size else np.nan
    out["nonburst_isi_cv"] = ml.cv(isi_out) if isi_out.size else np.nan

    if b.n == 0:
        for k in ("isi_mean", "ibi_mean", "ibi_cv", "duration_mean",
                  "nspikes_mean", "nspikes_norm", "isi_cv_within"):
            out[k] = np.nan
        out["nspikes_norm"] = 0.0
        return out

    # --- per burst ---
    counts = np.bincount(lab, minlength=b.n + 1)[1:]
    isi_mean = np.array([np.nanmean(isi_after[s:e]) if e > s else np.nan
                         for s, e in zip(b.start_idx, b.end_idx)])
    dur_s = b.durations_ms() / 1000.0
    ibi_s = np.diff(b.start_ms()) / 1000.0                     # start-to-start

    out.update({
        "isi_mean": float(np.nanmean(isi_mean)),
        "nspikes_mean": float(np.nanmean(counts)),
        "nspikes_norm": float(counts.sum()) / n_spike,
        "duration_mean": float(np.nanmean(dur_s)),
        "ibi_mean": float(np.nanmean(ibi_s)) if ibi_s.size else np.nan,
        "ibi_cv": ml.cv(ibi_s) if ibi_s.size else np.nan,
        "counts": counts, "isi_mean_per_burst": isi_mean, "duration_s": dur_s,
    })
    # ISI CV per burst, then averaged
    per_burst_cv = np.array([ml.cv(isi_after[s:e]) if e > s else np.nan
                             for s, e in zip(b.start_idx, b.end_idx)])
    out["isi_cv_per_burst"] = per_burst_cv
    out["isi_cv_within"] = ml.mean(per_burst_cv)
    return out
