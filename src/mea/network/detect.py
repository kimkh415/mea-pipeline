"""Network-burst detection chain (port of NetworkBurstDetection.m and helpers).

Rounds: beta (all units, bin ladder), alpha (robust participants), final (blended bin).
"""
from __future__ import annotations
import numpy as np

from . import popactivity as pa
from .bimodality import drop_echo_peaks

SCALING = 2.0
SCALING_ADJ = 0.75
N_SHORTLIST = 3
CHANGE_THRESHOLD = 0.40
MAX_IT = 100
BIN_DEFAULT1 = 0.100
BIN_DEFAULT2 = 0.150
BIN_MIN_SECONDARY = 0.010
SMOOTH_ADD = 0
WINDOW_FRACTION = 0.25
THRESHOLD_BROAD = 1.80
THRESHOLD_WINDOWS = 0.40
PERCENTILE_2GAP = 0.20
SEARCH_VALLEY = 1.0
THRESHOLD_SECONDARY = 1.00
MOVING_STEP = 0.25
THRESHOLD_BIS = 10.0


def bin_ladder(bin_min, bin_max, bin_center=None, scaling=SCALING,
               scaling_adj=SCALING_ADJ):
    """Bin-size ladder: doubling from bin_min, or refined around bin_center."""
    if bin_center is None or not np.isfinite(bin_center):
        bins = [bin_min]
        while bins[-1] < bin_max and len(bins) <= MAX_IT:
            bins.append(bins[-1] * scaling)
    else:
        bins = [bin_center / scaling / scaling]
        stop = (bin_max + bin_center) / 2.0
        while bins[-1] < stop and len(bins) <= MAX_IT:
            bins.append(bins[-1] * scaling * scaling_adj)
    return np.array(bins, float)


def choose_bin(n_burst, cv_ibi, change, shortlist=N_SHORTLIST,
               change_threshold=CHANGE_THRESHOLD):
    """Pick the bin whose event count is stable (|change| < threshold) and IBI CV is low.

    Returns None when no bin qualifies.
    """
    order = np.argsort(np.abs(np.where(np.isnan(change), np.inf, change)), kind="stable")
    cv_order = np.argsort(np.where(np.isnan(cv_ibi), np.inf, cv_ibi), kind="stable")
    best = None
    for i in range(min(shortlist, n_burst.size) - 1, -1, -1):
        idx = order[i]
        if np.isfinite(change[idx]) and abs(change[idx]) < change_threshold:
            if idx in cv_order[:shortlist]:
                best = int(order[i])
    if n_burst.size == 1:
        best = 0
    return best


def dynamic(trains, mask, t_range_s, bin_min, bin_max, bin_center, threshold,
            opt_norm=True, opt_distance=0, opt_binopti=0, rng=None,
            burst_labels=None, nb=None):
    """Sweep the bin ladder, then detect at the best bin."""
    nb = nb or {}
    bins = bin_ladder(bin_min, bin_max, bin_center,
                      float(nb.get("ladder_scaling") or SCALING),
                      float(nb.get("ladder_scaling_adj") or SCALING_ADJ))
    n = bins.size
    n_burst = np.zeros(n)
    cv_ibi = np.full(n, np.nan)
    change = np.full(n, np.nan)

    for k, b in enumerate(bins):
        r = pa.detect(trains, mask, t_range_s, b, threshold, 0, opt_norm, False, nb)
        n_burst[k] = r["nBurst"]
        if r["nBurst"] > 1:
            pk = np.array([x["peak"] for x in r["IndivBurst"]], float)
            ibi = np.diff(pk)
            # MATLAB std of one element is 0, not NaN
            m = np.nanmean(ibi) if ibi.size else np.nan
            sd = np.nanstd(ibi, ddof=1) if ibi.size > 1 else 0.0
            cv_ibi[k] = sd / m if m else np.nan
        if k > 0 and n_burst[k - 1] > 0:
            change[k - 1] = (n_burst[k] - n_burst[k - 1]) / n_burst[k - 1]

    best = choose_bin(n_burst, cv_ibi, change,
                      int(nb.get("ladder_shortlist") or N_SHORTLIST),
                      float(nb.get("ladder_change_thresh") or CHANGE_THRESHOLD))
    if best is None:
        return {"nBurst": 0, "bin": np.nan, "IndivBurst": [], "Secondary": _empty_secondary()}
    out = integrate(trains, mask, t_range_s, float(bins[best]), threshold,
                    opt_norm, opt_distance, opt_binopti, rng, burst_labels, nb)
    out["bin"] = float(bins[best])
    return out


def _empty_secondary():
    return {"nBurst": 0, "IndivBurst": [], "nBurst_per_gap": [], "bin": 0.0}


def integrate(trains, mask, t_range_s, bin_s, threshold, opt_norm=True,
              opt_distance=0, opt_binopti=0, rng=None, burst_labels=None, nb=None):
    """Detect primary peaks, then secondary peaks in the gaps."""
    nb = nb or {}
    r = pa.detect(trains, mask, t_range_s, bin_s, threshold, 0, opt_norm, False, nb)
    if opt_distance and r["nBurst"] > 0:
        kept, _ = drop_echo_peaks(
            r["IndivBurst"], rng,
            int(nb.get("echo_min_bursts") or 8),
            float(nb.get("echo_min_distance") or 4.0),
            float(nb.get("bimodality_bc") or 0.555))
        r["IndivBurst"] = kept
        r["nBurst"] = len(kept)

    if r["nBurst"] <= 0:
        r["Secondary"] = _empty_secondary()
        return r

    # Gap after each primary peak
    b = r["IndivBurst"]
    n = len(b)
    edges = np.empty((n, 2))
    for k in range(n):
        if bin_s < BIN_DEFAULT1:
            edges[k, 0] = b[k]["last"] + (b[k]["last"] - b[k]["start"]) / 2
        elif bin_s < BIN_DEFAULT2:
            edges[k, 0] = b[k]["last"]
        else:
            edges[k, 0] = b[k]["peak"] + (b[k]["last"] - b[k]["peak"]) / 2
        edges[k, 1] = t_range_s[1] if k == n - 1 else b[k + 1]["start"]

    r["Secondary"] = secondary(trains, mask, t_range_s, edges, bin_s, threshold,
                               opt_norm, opt_binopti, nb)
    return r


def _activity_drop(activity, time_binned, activity_matrix, threshold, noise, search_start):
    """Time at which the post-burst moving average first drops below baseline."""
    bin_size = float(np.median(np.diff(time_binned))) if time_binned.size > 1 else 1.0
    window_bin = int(round(search_start / bin_size))
    step_bin = max(1, int(round(search_start * MOVING_STEP / bin_size)))
    limit = int(np.ceil((time_binned[-1] - search_start) / max(search_start * MOVING_STEP, 1e-12)))

    out_time = search_start
    fin = 0
    hit = False
    for i in range(limit + 1):
        ini = i * step_bin
        fin = min(ini + window_bin, activity.size)
        if ini >= fin:
            break
        if np.nanmean(activity[ini:fin]) < threshold:
            out_time = time_binned[min(fin, time_binned.size - 1)]
            hit = True
            break
    if not hit:
        return out_time

    # Extend past the drop for a strong late peak inside a gap
    for i in range(activity_matrix.shape[0]):
        nan_at = np.flatnonzero(np.isnan(activity_matrix[i]))
        if nan_at.size == 0:
            continue
        end_gap = nan_at[0] - 1
        half = int(round(end_gap / 2))
        if half <= fin:
            continue
        sig = activity_matrix[i, :half]
        above = np.flatnonzero(sig > threshold + noise * THRESHOLD_BIS)
        if above.size and above[-1] > fin:
            loc = above[-1] + window_bin
            t2 = time_binned[loc] if time_binned.size > loc else \
                time_binned[-1] - time_binned[max(end_gap - window_bin, 0)]
            out_time = max(t2, out_time)
    return out_time


def secondary(trains, mask, t_range_s, edges, bin1, threshold, opt_norm,
              opt_binopti, nb=None):
    """Secondary peaks in the gaps between primaries; the bin yielding most peaks wins.

    Skipped for a bin unless enough gaps show early elevated activity.
    """
    if opt_binopti == 1:
        factors = [0.40, 0.20, 1.00] if bin1 < BIN_DEFAULT1 else [0.30, 0.12, 0.75, 0.05]
    elif opt_binopti == 0:
        factors = [0.30]
    else:
        factors = [opt_binopti / bin1]

    nb = nb or {}
    bin_floor = float(nb.get("secondary_bin_min_s") or BIN_MIN_SECONDARY)
    win_frac = float(nb.get("secondary_window_frac") or WINDOW_FRACTION)
    broad = float(nb.get("secondary_broad") or THRESHOLD_BROAD)
    min_windows = float(nb.get("secondary_min_windows") or THRESHOLD_WINDOWS)
    n_window = edges.shape[0]
    idx_all = np.flatnonzero(np.asarray(mask, bool))
    best = None
    best_n = 0

    for f in factors:
        bin_s = max(bin1 * f, bin_floor)
        ratio = np.full((n_window, 3), np.nan)
        largest_gap = 0.0
        for k in range(n_window):
            gap = edges[k, 1] - edges[k, 0]
            largest_gap = max(largest_gap, gap)
            s1, _ = pa.population_trace(trains, idx_all, bin_s, edges[k, 0],
                                        edges[k, 0] + gap * win_frac, opt_norm)
            s2, _ = pa.population_trace(trains, idx_all, bin_s,
                                        edges[k, 1] - gap * win_frac * 2,
                                        edges[k, 1], opt_norm)
            m1 = float(np.mean(s1)) if s1.size else np.nan
            m2 = float(np.mean(s2)) if s2.size else np.nan
            ratio[k] = (m1, m2, m1 / m2 if m2 else np.nan)

        high = np.flatnonzero(ratio[:, 2] > broad)
        if high.size / n_window <= min_windows:
            continue

        # Common time axis so gaps can be averaged
        _, tb = pa.population_trace(trains, idx_all, bin_s, edges[0, 0],
                                    edges[0, 0] + largest_gap, opt_norm)
        tb = tb - edges[0, 0]
        amat = np.full((n_window, tb.size), np.nan)
        for k in range(n_window):
            endt = min(edges[k, 1], t_range_s[1])
            s, _ = pa.population_trace(trains, idx_all, bin_s, edges[k, 0], endt, opt_norm)
            amat[k, :min(s.size, tb.size)] = s[:tb.size]

        pop = np.nanmean(amat, axis=0)
        umbral = float(np.nanmean(ratio[:, 1]))
        ruido = float(np.nanstd(ratio[:, 1], ddof=1))

        gaps2 = []
        for k in range(n_window):
            if (edges[k, 1] - edges[k, 0]) > bin_s * 5:
                sp = pa.detect(trains, mask, edges[k], bin_s, 0, SMOOTH_ADD,
                               opt_norm, True, nb)
                if sp["nBurst"] > 2:
                    pk = [x["peak"] for x in sp["IndivBurst"]]
                    gaps2.extend(np.diff(pk).tolist())
        if gaps2:
            g = np.sort(gaps2)
            search_start = SEARCH_VALLEY * g[int(np.ceil(PERCENTILE_2GAP * len(g))) - 1]
        else:
            search_start = largest_gap / 10.0
        drop_t = _activity_drop(pop, tb, amat, umbral, ruido, max(search_start, 1e-9))

        bursts, per_gap, total = [], np.zeros(n_window, int), 0
        for k in range(n_window):
            gap = edges[k, 1] - edges[k, 0]
            if k < n_window - 1:
                endt = min(edges[k, 0] + drop_t, edges[k, 1] - gap * win_frac)
            else:
                endt = min(edges[k, 0] + drop_t, t_range_s[1])
            if endt <= edges[k, 0]:
                continue
            thr2 = threshold if ratio[k, 2] < 1 else threshold * THRESHOLD_SECONDARY
            sp = pa.detect(trains, mask, (edges[k, 0], endt), bin_s, thr2,
                           SMOOTH_ADD, opt_norm, True, nb)
            per_gap[k] = sp["nBurst"]
            total += sp["nBurst"]
            for x in sp["IndivBurst"]:
                y = dict(x); y["parent"] = k
                bursts.append(y)

        if total > best_n:
            best_n = total
            best = {"nBurst": total, "IndivBurst": bursts,
                    "nBurst_per_gap": per_gap, "bin": bin_s}

    return best if best is not None else _empty_secondary()


# --- rounds ---
MIN_CHAN = 6
MIN_CHAN_BURST = 5
MIN_CHAN_RATIO = 0.075
MIN_CHAN_BURST_RATIO = 0.05
THRESHOLD_BURST_WITH_UNITS = 0.50
MIN_ROBUST = 4
BIN_MIN = 0.01
BIN_MAX = 3.0
BIN_DEFAULT = 0.20
MIN_SOLID_BURST = 3
MIN_CHAN_BURST_RATIO_SOLID = 0.4


def _fallback(trains, mask, t_range_s, threshold, opt_norm, opt_distance, rng,
              bin_default=BIN_DEFAULT, nb=None):
    """Detect at the default bin when no optimal bin was found."""
    r = integrate(trains, mask, t_range_s, bin_default, threshold, opt_norm,
                  opt_distance, 0, rng, nb=nb)
    r["bin"] = bin_default
    return r


def run(trains, burst_labels, isi_burst_mean_ms, t_range_s, streams,
        threshold=5.0, opt_distance=0, n_shuffling=500, clustering=True, nb=None):
    """Run the full chain; return final detection plus participating units.

    nBurst NaN means QC failed (not network bursting); peaks are still reported.
    """
    from . import players as pl

    nb = nb or {}
    detector = str(nb.get("detector", "fraction_active"))
    if detector == "fraction_active":
        return _run_fraction(trains, burst_labels, isi_burst_mean_ms, t_range_s,
                             streams, n_shuffling, nb)
    if detector != "yang":
        raise ValueError(f"network_burst.detector {detector!r}: "
                         "expected 'fraction_active' or 'yang'")
    bin_min = float(nb.get("bin_min_s", BIN_MIN))
    bin_max = float(nb.get("bin_max_s", BIN_MAX))
    bin_default = float(nb.get("bin_default_s", BIN_DEFAULT))
    min_robust = int(nb.get("min_robust", MIN_ROBUST))
    min_chan = int(nb.get("min_channels", MIN_CHAN))
    min_chan_burst = int(nb.get("min_channels_burst", MIN_CHAN_BURST))
    min_chan_ratio = float(nb.get("min_channel_ratio", MIN_CHAN_RATIO))
    min_chan_burst_ratio = float(nb.get("min_channel_burst_ratio", MIN_CHAN_BURST_RATIO))
    thresh_with_units = float(nb.get("threshold_with_units", THRESHOLD_BURST_WITH_UNITS))

    n = len(trains)
    all_units = np.ones(n, bool)
    rng = streams.for_item("bimodality")

    # --- round 1: all units, normalised and raw; keep whichever finds more ---
    beta = dynamic(trains, all_units, t_range_s, bin_min, bin_max, None, threshold,
                   True, opt_distance, 1, rng, nb=nb)
    if not np.isfinite(beta.get("bin", np.nan)):
        beta = _fallback(trains, all_units, t_range_s, threshold, True, opt_distance, rng, bin_default, nb)
    delta = dynamic(trains, all_units, t_range_s, bin_min, bin_max, None, threshold,
                    False, opt_distance, 0, rng, nb=nb)
    if not np.isfinite(delta.get("bin", np.nan)):
        delta = _fallback(trains, all_units, t_range_s, threshold, False, opt_distance, rng, bin_default, nb)
    if delta["nBurst"] > beta["nBurst"]:
        beta = delta

    if beta["nBurst"] <= 0:
        return {"nBurst": 0, "IndivBurst": [], "bin": beta.get("bin", np.nan),
                "Channels": np.array([], int), "ChannelsIntegrated": np.array([], int),
                "Secondary": _empty_secondary(), "clustering": None, "lowUnits": np.nan}

    # --- screen robust participants in round-1 events ---
    bid, n_bursting, meaningful = pl.screen(
        trains, burst_labels, beta["IndivBurst"], t_range_s,
        float(nb.get("safe_k") or pl.SAFE_K),
        float(nb.get("participation_thresh") or pl.ACTIVITY_THRESHOLD))
    solid_thr = max(MIN_CHAN_BURST, np.median(n_bursting) * MIN_CHAN_BURST_RATIO_SOLID)
    n_solid = max(int((n_bursting >= solid_thr).sum()), MIN_SOLID_BURST)
    robust = pl.robust_participants(bid, n_bursting, n_solid)

    # --- rounds 2 + 3: robust units only, blended bin ---
    flag2 = 0
    if robust.sum() < min_robust:
        final = beta
        bin_final = beta["bin"]
        flag2 = 2
        units = all_units
    else:
        units = robust
        alpha = dynamic(trains, units, t_range_s, bin_min, bin_max, beta["bin"],
                        threshold, True, opt_distance, 1, rng, nb=nb)
        if not np.isfinite(alpha.get("bin", np.nan)):
            alpha = _fallback(trains, units, t_range_s, threshold, True, opt_distance, rng, bin_default, nb)
            flag2 = 1
            bin_final = (beta["bin"] + alpha["bin"]) / 2.0
        else:
            bin_final = (beta["bin"] + alpha["bin"] * 2) / 3.0
        final = dynamic(trains, units, t_range_s, bin_final, bin_final, None,
                        threshold, True, opt_distance, 1, rng, nb=nb)

    if final["nBurst"] <= 0:
        return {"nBurst": 0, "IndivBurst": [], "bin": bin_final,
                "Channels": np.array([], int), "ChannelsIntegrated": np.array([], int),
                "Secondary": _empty_secondary(), "clustering": None, "lowUnits": np.nan}

    # --- participation stats on final events, then one more detection pass ---
    participation, part_score = pl.statistics(
        trains, burst_labels, isi_burst_mean_ms, final["IndivBurst"], t_range_s,
        streams, n_shuffling,
        float(nb.get("safe_k") or pl.SAFE_K),
        float(nb.get("participation_final") or pl.ACTIVITY_THRESHOLD_FINAL))
    channels = np.flatnonzero(participation) + 1
    if participation.sum() > 0:
        final = integrate(trains, participation.astype(bool), t_range_s, bin_final,
                          threshold, True, opt_distance,
                          final["Secondary"].get("bin", 0) or 0, rng, nb=nb)
        final["bin"] = bin_final

    out = dict(final)
    out["Channels"] = channels
    out["ChannelsIntegrated"] = channels
    out["lowUnits"] = 0
    # Per-unit measures
    out["participation"] = participation.astype(bool)
    out["participation_p"] = part_score[:, 0]  # fraction of surrogates beaten
    out["participation_events"] = part_score[:, 1]  # number of events the unit is in
    # From the screening pass (round-1 events, not the final set)
    out["screen_meaningful"] = np.asarray(meaningful, bool)  # in-event rate > ACTIVITY_THRESHOLD x own rate
    out["screen_bursting_fraction"] = ((bid == 2).mean(axis=0)  # events where the unit itself was bursting
                                       if bid.size else np.zeros(n))

    # Store the detector's own trace (participants only, per-unit normalised), not a raw histogram
    mask_final = np.zeros(n, bool)
    if channels.size:
        mask_final[channels - 1] = True
    idx_final = pa.select_units(trains, mask_final, t_range_s[0], t_range_s[1], False)
    # True matches the opt_norm of the final integrate() above
    trace, mid = pa.population_trace(trains, idx_final, bin_final,
                                     t_range_s[0], t_range_s[1], True)
    out["spikesBin"] = trace
    out["time"] = mid

    # --- final QC on primary events ---
    # below ~80 units the absolute floor min_chan dominates; NaN may mean too few units
    n_part = channels.size
    if out["nBurst"] > 0:
        if n_part < max(min_chan, min_chan_ratio * n):
            out["nBurst"] = np.nan
        else:
            ts, _, _ = pl.event_spike_times(
                trains, out["IndivBurst"], float(nb.get("safe_k") or pl.SAFE_K))
            min_units_burst = max(min_chan_burst, min_chan_burst_ratio * n)
            low = 0
            for k in range(len(out["IndivBurst"])):
                active = sum(1 for ch in channels - 1 if ts[ch][k].size > 0)
                if active < min_units_burst:
                    low += 1
            out["lowUnits"] = low
            if low / len(out["IndivBurst"]) > (1 - thresh_with_units):
                out["nBurst"] = np.nan
    out["flag2"] = flag2
    return out


def _run_fraction(trains, burst_labels, isi_burst_mean_ms, t_range_s, streams,
                  n_shuffling, nb):
    """fraction_active detector: events from fraction.detect, participation via the surrogate test.

    No events gives nBurst NaN; Yang's participation floors are not applied.
    """
    from . import players as pl
    from . import fraction as fr

    det = fr.detect(trains, t_range_s,
                    frac_min=float(nb.get("frac_min", fr.FRAC_MIN)),
                    window_s=float(nb.get("frac_window_s", fr.WINDOW_S)),
                    step_s=float(nb.get("frac_step_s", fr.STEP_S)),
                    merge_s=float(nb.get("frac_merge_s", fr.MERGE_S)),
                    background_max=float(nb.get("frac_background_max", fr.BACKGROUND_MAX)))
    n = len(trains)
    out = dict(det)
    out.update({"Secondary": _empty_secondary(), "clustering": None, "flag2": None,
                "Channels": np.array([], int), "ChannelsIntegrated": np.array([], int),
                "lowUnits": np.nan, "detector": "fraction_active"})
    if not det["IndivBurst"]:
        out["nBurst"] = np.nan
        return out

    safe_k = float(nb.get("safe_k") or pl.SAFE_K)
    bid, n_bursting, meaningful = pl.screen(
        trains, burst_labels, det["IndivBurst"], t_range_s, safe_k,
        float(nb.get("participation_thresh") or pl.ACTIVITY_THRESHOLD))
    participation, part_score = pl.statistics(
        trains, burst_labels, isi_burst_mean_ms, det["IndivBurst"], t_range_s,
        streams, n_shuffling, safe_k,
        float(nb.get("participation_final") or pl.ACTIVITY_THRESHOLD_FINAL))
    channels = np.flatnonzero(participation) + 1
    out.update({"Channels": channels, "ChannelsIntegrated": channels, "lowUnits": 0,
                "participation": participation.astype(bool),
                "participation_p": part_score[:, 0],
                "participation_events": part_score[:, 1],
                "screen_meaningful": np.asarray(meaningful, bool),
                "screen_bursting_fraction": ((bid == 2).mean(axis=0) if bid.size else np.zeros(n))})
    return out
