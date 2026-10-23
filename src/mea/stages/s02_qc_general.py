"""Stage 2 -- QC general.  *_raw.mat -> cleaned Recording.

Five tests (echo, silent, TTX, noisy, picture), each its own flag.
Units are flagged, never renumbered.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ..core.stats import cv, lvr
from ..core.matlabcompat import isoutlier_percentile
from ..burst.detect import detect
from ..burst.features import features


def refractory_pass(t_ms, refractory_ms):
    """Merge spike pairs closer than the refractory period into their midpoint.

    Merged in reverse order, so a run of three collapses as (t1 + (t2+t3)/2) / 2.
    """
    t = np.asarray(t_ms, float).copy()
    if t.size < 2:
        return t, 0
    viol = np.flatnonzero(np.diff(t) < refractory_ms)
    n = viol.size
    if n == 0:
        return t, 0
    keep = np.ones(t.size, bool)
    for k in viol[::-1]:
        t[k] = (t[k] + t[k + 1]) / 2.0
        keep[k + 1] = False
    return t[keep], n


def run(rec, cfg, per_condition_counts, per_condition_length_s):
    """Run the five QC tests and mark failures on `rec`.

    per_condition_counts / _length_s: (n_conditions, n_units).
    """
    g = cfg["qc_general"]
    n = rec.n_units

    # --- 1. refractory / echo ---
    echo = np.zeros(n, bool)
    removed_pct = np.zeros(n)
    new_trains = []
    for i in range(n):
        t, n_close = refractory_pass(rec.train(i), g["refractory_ms"])
        new_trains.append(t)
        n0 = rec.offsets[i + 1] - rec.offsets[i]
        if n_close:
            removed_pct[i] = n_close * 100.0 / n0
            if removed_pct[i] > g["refractory_max_pct"] and n_close > g["refractory_min_spikes"]:
                echo[i] = True

    # --- 2. activity level ---
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = per_condition_counts / per_condition_length_s        # (cond, unit)
    active = np.any(rate > g["min_rate_hz"], axis=0)
    silent = ~active

    # --- 3. TTX sensitivity ---
    # TTX must remove >= ttx_min_drop of the unit's PEAK rate across conditions, not baseline
    ttx_rows = np.flatnonzero(rec.conditions["is_ttx"].to_numpy())
    ttx_sensitive = np.zeros(n, bool)
    ttx_ratio = np.full(n, np.nan)
    ttx_expected = np.full(n, np.nan)
    ttx_drop = np.full(n, np.nan)
    if ttx_rows.size:
        # round: 1 - 0.9 is 0.0999...98 in floating point; units exactly on the bar flip
        keep_frac = round(1.0 - float(g["ttx_min_drop"]), 12)
        max_rate = np.nanmax(rate, axis=0)
        # nanmin over TTX blocks = any block below the bar
        ttx_rate = np.nanmin(rate[ttx_rows, :], axis=0)
        ttx_sensitive = max_rate * keep_frac > ttx_rate
        with np.errstate(divide="ignore", invalid="ignore"):
            ttx_ratio = np.divide(ttx_rate, max_rate,
                                  out=np.full(n, np.nan), where=max_rate > 0)
        ttx_drop = 1.0 - ttx_ratio
        # spikes the bar corresponds to in the TTX block; 1-3 means counting noise
        ttx_expected = max_rate * keep_frac * float(
            per_condition_length_s[ttx_rows[0]][0]
            if np.ndim(per_condition_length_s) > 1
            else per_condition_length_s[ttx_rows[0]])

    # --- 4. tonic / abnormal activity ---
    noisy = _noisy(rec, new_trains, silent, cfg)

    # --- 5. spatial (photo crop) ---
    in_picture = np.ones(n, bool)        # all units when there is no photo

    # Mark each reason on the Recording
    # TTX-INSENSITIVE units fail (not neurons)
    fails_ttx = ~ttx_sensitive if ttx_rows.size else np.zeros(n, bool)
    for flag, mask, value in (("echo", echo, removed_pct),
                              ("silent", silent, None),
                              ("not_ttx_sensitive", fails_ttx, None),
                              ("noisy", noisy, None),
                              ("outside_picture", ~in_picture, None)):
        pos = np.flatnonzero(mask)
        if pos.size:
            # discard decided below by discard_mask + qc_general.tests
            rec.mark("qc_general", flag, pos,
                     None if value is None else np.asarray(value)[pos],
                     drop=False)

    # Label each unit by the first test it failed
    label = np.array(["good"] * n, dtype=object)
    for name, mask in (("echo", echo), ("silent", silent),
                       ("ttx_insensitive", fails_ttx), ("noisy", noisy),
                       ("outside_picture", ~in_picture)):
        label[(label == "good") & mask] = name

    # Store on the Recording for figures and stage 5
    rec.units["unitType_ours"] = label
    rec.units["ttx_ratio"] = ttx_ratio
    rec.units["ttx_drop"] = ttx_drop
    rec.units["ttx_expected_spikes"] = ttx_expected

    flags = {"echo": echo, "silent": silent, "ttx_sensitive": ttx_sensitive,
             "noisy": noisy, "in_picture": in_picture}
    tests = tuple(g.get("tests", ALL_TESTS))
    discarded = discard_mask(flags, bool(ttx_rows.size), tests)
    rec.units.loc[rec.units.index[np.flatnonzero(discarded)], "kept"] = False

    return {**flags, "label": label, "ttx_ratio": ttx_ratio,
            "ttx_drop": ttx_drop,
            "ttx_expected_spikes": ttx_expected, "removed_pct": removed_pct,
            "trains": new_trains, "rate": rate, "tests": list(tests),
            "n_discarded": int(discarded.sum())}


def _noisy(rec, trains, silent, cfg):
    """Flag tonic, high-count units unless their burst/non-burst contrast marks a neuron."""
    g = cfg["qc_general"]
    n = len(trains)
    n_units = np.array([t.size for t in trains], float)
    cvu = np.zeros(n); lvru = np.zeros(n)
    for i, t in enumerate(trains):
        if t.size > 1:
            isi = np.diff(t)
            m = isi.mean()
            cvu[i] = isi.std(ddof=1) / m if m else 0.0
            lvru[i] = lvr(isi)

    uniform = np.flatnonzero((cvu < g["cv_threshold"]) &
                             (lvru < g["lvr_threshold"]) & ~silent)
    out1 = uniform[isoutlier_percentile(n_units[uniform], 0, g["percentile"])] \
        if uniform.size else np.empty(0, int)

    non_silent = np.flatnonzero(~silent)
    out2 = non_silent[isoutlier_percentile(n_units[non_silent], 0, g["percentile_top"])] \
        if non_silent.size else np.empty(0, int)

    if g.get("noisy_index_mode", "cluster_index") == "legacy_find_positions":
        # positions 1..len(out2) instead of the outlier units themselves
        cand = (np.union1d(out1 + 1, np.arange(1, out2.size + 1)) - 1).astype(int)
    else:
        cand = np.union1d(out1, out2).astype(int)

    # Rescue units with a clear burst/tonic contrast
    keep_as_noisy = []
    for i in cand:
        f = features(detect(trains[i], cfg))
        isi_b, isi_nb = f["isi_mean"], f["nonburst_isi_mean"]
        contrast = (np.isfinite(isi_b) and np.isfinite(isi_nb)
                    and isi_b * g["isi_ratio"] < isi_nb)
        mostly_bursting = f["nspikes_norm"] > g["n_spikes_burst"]
        if not (contrast or mostly_bursting):
            keep_as_noisy.append(i)
    noisy = np.zeros(n, bool)
    noisy[np.array(keep_as_noisy, int)] = True
    return noisy


# Tests that discard a unit.
# echo/silent/noisy overlap bombcell; `[ttx]` alone avoids double-counting.
# Selects discards only; the Recording's spike times are never replaced.
ALL_TESTS = ("echo", "silent", "noisy", "ttx", "picture")


def non_ttx_survivors(flags, selected, has_ttx=True, tests=ALL_TESTS):
    """Units passing `selected` (upstream selection mask) and every QC test except ttx."""
    rest = tuple(t for t in tests if t != "ttx")
    return np.asarray(selected, bool) & ~discard_mask(flags, has_ttx, rest)


def discard_mask(flags, has_ttx, tests=ALL_TESTS):
    """Discard mask: echo | silent | noisy | ~ttx | ~picture, limited to `tests`."""
    unknown = sorted(set(tests) - set(ALL_TESTS))
    if unknown:
        raise ValueError(f"qc_general.tests names unknown tests {unknown}; "
                         f"available: {list(ALL_TESTS)}")
    n = flags["echo"].shape
    d = np.zeros(n, bool)
    if "echo" in tests:
        d |= flags["echo"]
    if "silent" in tests:
        d |= flags["silent"]
    if "noisy" in tests:
        d |= flags.get("noisy", np.zeros(n, bool))
    if "ttx" in tests and has_ttx:
        d |= ~flags["ttx_sensitive"]
    if "picture" in tests:
        d |= ~flags["in_picture"]
    return d
