"""Stage 3 -- QC crosstalk.  *_cleaned.mat -> per-pair scores.

Per unit pair: measure coincidence, screen, and test survivors against a surrogate null.
"""
from __future__ import annotations
import numpy as np
from ..core.windows import window_bounds
from ..core.surrogate import constant_rate_segments, surrogate_times_ms


# --- measurement ---
def coincidence(t1, in_burst1, t2, in_burst2, threshold_ms):
    """Masks of spikes in each train with a partner in the other within +-threshold.

    searchsorted finds candidates (padded by a few ULPs); abs(t2 - t1) < thr decides, as MATLAB.
    """
    n1, n2 = t1.size, t2.size
    hit1 = np.zeros(n1, bool)
    hit2 = np.zeros(n2, bool)
    if n1 == 0 or n2 == 0:
        return {"hit1": hit1, "hit2": hit2}
    pad = 4.0 * np.maximum(np.spacing(np.abs(t1)), np.spacing(threshold_ms))
    lo = np.searchsorted(t2, t1 - threshold_ms - pad, side="left")
    hi = np.searchsorted(t2, t1 + threshold_ms + pad, side="right")
    for i in np.flatnonzero(hi > lo):
        cand = np.arange(lo[i], hi[i])
        ok = np.abs(t2[cand] - t1[i]) < threshold_ms     # on the difference, as MATLAB
        if ok.any():
            hit1[i] = True
            hit2[cand[ok]] = True
    return {"hit1": hit1, "hit2": hit2}


def pair_metrics(t1, b1, t2, b2, threshold_ms):
    """Coincident fraction of burst, non-burst and all spikes for one pair."""
    c = coincidence(t1, b1, t2, b2, threshold_ms)
    h1, h2 = c["hit1"], c["hit2"]
    n_total = t1.size + t2.size
    n_burst = int(b1.sum() + b2.sum())
    n_non = n_total - n_burst
    hs_burst = int((h1 & b1).sum() + (h2 & b2).sum())
    hs_non = int((h1 & ~b1).sum() + (h2 & ~b2).sum())
    hs_tot = int(h1.sum() + h2.sum())
    f = lambda a, b: (a / b) if b else np.nan
    return {"burst_norm": f(hs_burst, n_burst), "burst_total": n_burst,
            "nonburst_norm": f(hs_non, n_non), "nonburst_total": n_non,
            "total_norm": f(hs_tot, n_total), "total_total": n_total,
            "ch1_norm": f(int(h1.sum()), t1.size), "ch2_norm": f(int(h2.sum()), t2.size)}


# --- screening ---
def screen(m, q):
    """Return red_flag: 2 = low synchrony on all three measures (skip stats), else 0."""
    cond = 0
    for key, cut in (("burst", q["screen_burst"]), ("nonburst", q["screen_nonburst"]),
                     ("total", q["screen_total"])):
        norm, tot = m[f"{key}_norm"], m[f"{key}_total"]
        if not np.isfinite(norm) or norm * 100 < cut:
            cond += 1
        elif tot * norm < q["min_spikes"]:
            cond += 1
    return 2 if cond == 3 else 0


def deserves_statistics(m, q):
    """Whether a pair is worth the surrogate test, plus which category triggered it."""
    hs = {k: m[f"{k}_norm"] * 100 for k in ("total", "burst", "nonburst")}
    ns = {k: (m[f"{k}_total"] * m[f"{k}_norm"]) for k in ("total", "burst", "nonburst")}
    flags = {"burst": 0, "nonburst": 0, "total": 0}
    if hs["total"] > q["pct_total"] and ns["total"] > q["min_spikes"]:
        flags["total"] = 1
        return True, flags
    burst_minority = (not np.isfinite(m["burst_total"])) or m["burst_total"] < m["nonburst_total"]
    if burst_minority:
        if hs["nonburst"] > q["pct_nonburst"] and ns["nonburst"] > q["min_spikes"]:
            flags["nonburst"] = 1; return True, flags
        if hs["burst"] > q["pct_burst"] * q["factor_minority"] and ns["burst"] > q["min_spikes"]:
            flags["burst"] = 1; return True, flags
    else:
        if hs["burst"] > q["pct_burst"] and ns["burst"] > q["min_spikes"]:
            flags["burst"] = 1; return True, flags
        if hs["nonburst"] > q["pct_nonburst"] * q["factor_minority"] and ns["nonburst"] > q["min_spikes"]:
            flags["nonburst"] = 1; return True, flags
    return False, flags


# --- statistics ---
def percentiles(observed, surrogate_vals, n_valid_min=1):
    """Percent of valid (finite) surrogates below `observed`, and the valid count.

    MATLAB divides by all iterations, including NaN ones.
    """
    v = np.asarray(surrogate_vals, float)
    valid = np.isfinite(v)
    n = int(valid.sum())
    if n < n_valid_min or not np.isfinite(observed):
        return np.nan, n
    return float((v[valid] < observed).sum()) / n * 100.0, n


def surrogate_metrics(rng, t1, in_burst1, t2, burst2, threshold_ms, t_range_ms,
                      t_step_ms, n_surrogates, cfg):
    """Coincidence metrics of unit 1 against `n_surrogates` shuffles of unit 2.

    Surrogates keep unit 2's burst windows and in/out-of-burst rates.
    """
    from ..core.surrogate import (constant_rate_segments, surrogate_times_ms)

    t0, t1_end = float(t_range_ms[0]), float(t_range_ms[1])
    n_bin = int(round((t1_end - t0) / t_step_ms))
    lab = burst2.label
    in_b = lab > 0
    p_in = float(in_b.sum()) / max(n_bin, 1)
    p_out = float((~in_b).sum()) / max(n_bin, 1)
    lo, hi = [], []
    for b in np.unique(lab[lab > 0]):
        sel = t2[lab == b]
        lo.append(int((sel.min() - t0) / t_step_ms))
        hi.append(int((sel.max() - t0) / t_step_ms))
    segs = constant_rate_segments(n_bin, np.array(lo, int), np.array(hi, int),
                                  p_in, p_out)
    out = {k: np.full(n_surrogates, np.nan) for k in
           ("burst_norm", "nonburst_norm", "total_norm")}
    for i in range(n_surrogates):
        st = surrogate_times_ms(rng, segs, t_step_ms, t0)
        st.sort()
        sb = np.zeros(st.size, bool)          # surrogate burst membership
        for a, b in zip(lo, hi):
            sb |= (st >= t0 + a * t_step_ms) & (st <= t0 + b * t_step_ms)
        m = pair_metrics(t1, in_burst1, st, sb, threshold_ms)
        for k in out:
            out[k][i] = m[k]
    return out


def run_statistics(rec, cfg, ctx, screen_result, streams):
    """Surrogate test for the pairs that screening kept.

    Unvalidated: archived _QC.mat scores come from an older MATLAB version.
    """
    from ..burst.detect import detect

    q = cfg["qc_crosstalk"]
    trains = ctx.get("pre_echo_trains") or [rec.train(i) for i in range(rec.n_units)]
    t_step = _t_step_ms(rec)
    thr = screen_result["threshold_ms"]
    flag = screen_result["red_flag"]
    bursts = [detect(t, cfg) for t in trains]
    inb = [b.label > 0 for b in bursts]
    t_range = (float(rec.conditions["t0_s"].min()) * 1000.0,
               float(rec.conditions["t1_s"].max()) * 1000.0)
    n_sur = int(q["n_surrogates"])

    results = {}
    n = len(trains)
    for i in range(n):
        for j in range(i + 1, n):
            if flag[i, j] != 0:
                continue
            m = pair_metrics(trains[i], inb[i], trains[j], inb[j], thr)
            wanted, number = deserves_statistics(m, q)
            entry = {"number": number,
                     "percentile": {k: 0 for k in ("burst", "nonburst", "total")},
                     "n_valid": {}, "final": {}}
            if not wanted:
                # demote the pair, skip surrogates
                flag[i, j] = flag[j, i] = 1
                entry["final"] = {k: 0 for k in ("burst", "nonburst", "total")}
                entry["master"] = 0
                results[(i, j)] = entry
                continue
            rng = streams.for_item("crosstalk", i, j)
            sur = surrogate_metrics(rng, trains[i], inb[i], trains[j], bursts[j],
                                    thr, t_range, t_step, n_sur, cfg)
            for key, cut in (("burst", q["percentile_burst"]),
                             ("nonburst", q["percentile_nonburst"]),
                             ("total", q["percentile_total"])):
                pct, n_valid = percentiles(m[f"{key}_norm"], sur[f"{key}_norm"])
                entry["n_valid"][key] = n_valid
                entry["percentile"][key] = int(np.isfinite(pct) and pct > cut)
            entry["final"] = {k: number[k] * entry["percentile"][k]
                              for k in ("burst", "nonburst", "total")}
            entry["master"] = int(any(entry["final"].values()))
            results[(i, j)] = entry

    master = sum(v["master"] for v in results.values())
    return {"pairs": results, "red_flag": flag, "n_tested": len(results),
            "n_significant": master, "discard_mask": discard_mask(results, n),
            "validated": False}


def discard_mask(pairs, n_units):
    """Drop the higher-indexed unit of each significant pair, if the lower is still kept.

    Order-dependent, so pairs are scanned in sorted (i, j) order.
    """
    out = np.zeros(int(n_units), bool)
    for (i, j) in sorted(pairs):
        if pairs[(i, j)]["master"] and not out[i]:
            out[j] = True
    return out


def _t_step_ms(rec, default=0.05):
    """Sampling period in ms; `default` if missing, NaN or non-positive."""
    v = rec.meta.get("t_step_ms")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) and v > 0 else default


def screen_all(rec, cfg, ctx):
    """Screen every unit pair (deterministic pass)."""
    from ..burst.detect import detect
    q = cfg["qc_crosstalk"]
    trains = ctx.get("pre_echo_trains") or [rec.train(i) for i in range(rec.n_units)]
    t_step = _t_step_ms(rec)
    thr = min(q["jitter_max_ms"], q["jitter_ticks"] * t_step)
    bursts = [detect(t, cfg) for t in trains]
    inb = [b.label > 0 for b in bursts]
    n = len(trains)
    flag = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            m = pair_metrics(trains[i], inb[i], trains[j], inb[j], thr)
            f = screen(m, q)
            flag[i, j] = flag[j, i] = f
    iu = np.triu_indices(n, 1)
    return {"red_flag": flag, "threshold_ms": thr,
            "n_pairs": int(iu[0].size),
            "n_screened_out": int((flag[iu] == 2).sum()),
            "n_for_statistics": int((flag[iu] == 0).sum())}
