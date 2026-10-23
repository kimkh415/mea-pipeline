"""Stage 4 -- QC final.  *_cleaned + *_QC -> curated Recording.

Applies the activity gate, crosstalk discards and replayed curation.json decisions.
"""
from __future__ import annotations
import numpy as np
from ..core.stats import cv, lvr, rate_mode, ISI_SPAN, COUNTING
from ..curation.decisions import Curation

VIEWS = {0: "below-threshold activity", 1: "all rasters",
         2: "rate vs CV-of-ISI and rate vs LVR outliers", 3: "candidate discards",
         4: "waveform classification (run last)", 5: "hypersynchrony discards"}


def gate_rate(t_ms, duration_s, mode):
    """Rate for the activity gate: 1/median(ISI) in Hz (MATLAB), or n/duration if counting.

    NaN with < 2 spikes, as MATLAB median([]); see _silent().
    """
    t = np.asarray(t_ms, float)
    if mode == COUNTING:
        return t.size / duration_s if duration_s > 0 else np.nan
    if t.size < 2:
        return np.nan
    med = float(np.median(np.diff(t)))
    return 1000.0 / med if med > 0 else np.nan


def _slice(t_ms, t0_s, t1_s):
    """Spikes strictly inside a condition window (both ends exclusive, as MATLAB)."""
    lo = np.searchsorted(t_ms, t0_s * 1000.0, "right")
    hi = np.searchsorted(t_ms, t1_s * 1000.0, "left")
    return t_ms[lo:hi]


def _silent(freq_cond, gate_conditions, threshold):
    """Silent if below threshold in EVERY gated condition (per-condition rates, as MATLAB).

    A NaN rate (0-1 spikes) does not count as low, so such a condition saves the unit.
    """
    low = np.asarray(freq_cond, float) < threshold        # NaN -> False
    return bool(low[gate_conditions].sum() >= gate_conditions.size)


def unit_table(rec, cfg):
    """Per-unit metrics for the review views, over non-TTX conditions only."""
    compat = cfg.get("compat", {})
    mode = rate_mode(compat)
    thr = float(cfg.get("qc_manual", {}).get("min_activity_hz", 0.04))

    cond = rec.conditions
    t0 = cond["t0_s"].to_numpy(float)
    t1 = cond["t1_s"].to_numpy(float)
    is_ttx = cond["is_ttx"].to_numpy(bool) if "is_ttx" in cond else np.zeros(len(cond), bool)
    gated = np.flatnonzero(~is_ttx)
    regular = np.flatnonzero(~is_ttx)
    length_rec = float((t1[regular] - t0[regular]).sum())

    rows = []
    for i in range(rec.n_units):
        t = rec.train(i)
        per_cond = [_slice(t, t0[c], t1[c]) for c in range(len(cond))]
        freq_cond = np.array([gate_rate(p, t1[c] - t0[c], mode)
                              for c, p in enumerate(per_cond)])
        train = np.concatenate([per_cond[c] for c in regular]) if regular.size else np.array([])
        isi = np.diff(train) if train.size > 1 else np.array([])
        freq = gate_rate(train, length_rec, mode)
        rows.append({
            "unit_id": rec.units["unit_id"].iloc[i],
            "n_spikes": train.size,
            "rate_hz": train.size / length_rec if length_rec > 0 else np.nan,
            "gate_rate_hz": freq,
            # NaN freq -> NaN CV/LVR, as MATLAB `if freq(k) > 0`
            "cv_isi": cv(isi) if (freq > 0 and isi.size > 1) else np.nan,
            "lvr": lvr(isi) if (freq > 0 and isi.size > 2) else np.nan,
            "silent": _silent(freq_cond, gated, thr),
        })
    import pandas as pd
    return pd.DataFrame(rows)


def run(rec, cfg, crosstalk_scores=None, curation=None):
    tbl = unit_table(rec, cfg)

    # 1. below-threshold activity
    low = np.flatnonzero(tbl["silent"].to_numpy(bool))
    rec.mark("qc_manual", "below_min_activity", low)

    # 2. hypersynchrony discards from stage 3
    if crosstalk_scores is not None:
        bad = np.flatnonzero(np.asarray(crosstalk_scores, bool))
        rec.mark("qc_manual", "hypersynchrony", bad)

    # 3. replay curation decisions
    orphaned = []
    if curation is not None:
        _, orphaned = curation.apply(rec)

    return {"table": tbl, "orphaned_curation_ids": orphaned,
            "n_low_activity": int(low.size)}

# --- review ordering ---
N_OUTSTANDING = 20
N_PLOT = 30


def _rank(values, ascending, n):
    """Positions of the n most extreme finite values (NaN excluded), stable order."""
    v = np.asarray(values, float)
    finite = np.flatnonzero(np.isfinite(v))
    order = finite[np.argsort(v[finite], kind="stable")]
    if not ascending:
        order = finite[np.argsort(-v[finite], kind="stable")]
    return order[:n]


def review_order(rec, cfg, burst_counts=None):
    """Ranked shortlists of active units for review (MATLAB option 2).

    cv/lvr: 2*n_outstanding highest CV / lowest Lv; freq: n_outstanding highest rate.
    burst_hi/burst_lo (only with `burst_counts`): of those, most / fewest bursts.
    Positions index the full unit array.
    """
    q = cfg.get("qc_manual", {})
    n_out = int(q.get("n_outstanding", N_OUTSTANDING))
    tbl = unit_table(rec, cfg)

    active = ~tbl["silent"].to_numpy(bool)
    idx_active = np.flatnonzero(active)
    n_active = idx_active.size
    n1 = min(n_out, n_active)
    n2 = min(n_out * 2, n_active)

    cv_a = tbl["cv_isi"].to_numpy()[idx_active]
    lvr_a = tbl["lvr"].to_numpy()[idx_active]
    freq_a = tbl["rate_hz"].to_numpy()[idx_active]      # counting rate

    out = {
        "n_active": int(n_active),
        "cv":   idx_active[_rank(cv_a, False, n2)],
        "lvr":  idx_active[_rank(lvr_a, True, n2)],
        "freq": idx_active[_rank(freq_a, False, n1)],
    }

    if burst_counts is not None:
        nb = np.asarray(burst_counts, float)
        for name, src, asc in (("burst_hi", "cv", False), ("burst_lo", "lvr", True)):
            shortlist = out[src]
            counts = nb[shortlist]
            order = np.argsort(counts, kind="stable")
            if not asc:
                order = np.argsort(-counts, kind="stable")
            out[name] = shortlist[order][:n1]
    return out


def silent_units(rec, cfg, limit=N_PLOT):
    """Units the activity gate dropped, in index order (first `limit`)."""
    tbl = unit_table(rec, cfg)
    return np.flatnonzero(tbl["silent"].to_numpy(bool))[:limit]
