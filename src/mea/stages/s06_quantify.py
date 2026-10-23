"""Stage 6 -- Quantification.  *_QCmanual.mat -> per-condition metrics.

Levels: A spiking, B single-cell bursting (bursting vs tonic), C network bursting.
Disputed definitions are chosen by `compat`; `legacy` options reproduce the MATLAB.
"""
from __future__ import annotations
import numpy as np
from ..core import matlab_semantics as ml
from ..core.stats import (lvr, entropy_bits, code_counts, mutual_information_bits,
                          sttc, fano_factor, fano_factor_operational,
                          population_mean)
from ..core.binning import histogram
from ..burst.detect import detect
from ..burst.features import features


# --- level A ---
def spike_features(trains, n_units_field, rec_length_s, t_range_ms, cfg):
    """Level-A spiking metrics per unit and population; `trains` already windowed."""
    sfg = cfg["spike_features"]
    compat = cfg.get("compat", {})
    n = len(trains)

    isi = [np.diff(t) if t.size > 1 else np.array([]) for t in trains]
    isi_mean = np.array([ml.mean(x) for x in isi])
    isi_cv = np.array([ml.cv(x) for x in isi])

    # --- Fano factor ---
    # Var(N)/Mean(N) over fano_window_ms; MATLAB's pow2(cv) = 2^CV not computed
    bw = float(sfg["fano_window_ms"])
    ff = np.array([fano_factor(histogram(t, bw, (t_range_ms[0] / 1000.0,
                                                 t_range_ms[1] / 1000.0)))
                   if t.size > 1 else np.nan for t in trains])

    # Operational time (Rajdl, Lansky & Kostal 2020): window = w_o mean ISIs per unit
    w_o = float(sfg.get("fano_operational_isi", 5))
    min_w = int(sfg.get("fano_operational_min_windows", 10))
    ff_op = np.array([fano_factor_operational(t, m, w_o, t_range_ms, min_w)
                      if t.size > 1 else np.nan for t, m in zip(trains, isi_mean)])

    # --- firing rate ---
    rate_isi = np.array([1000.0 / m if np.isfinite(m) and m else np.nan for m in isi_mean])
    rate_cnt = np.asarray(n_units_field, float) / np.asarray(rec_length_s, float)
    # reported rate per compat.firing_rate (core.stats.rate_mode)
    from ..core.stats import rate_mode, ISI_SPAN
    rate = rate_isi if rate_mode(compat) == ISI_SPAN else rate_cnt

    # --- local variation ---
    # Lv on ISIs, not raw timestamps
    lv = np.array([lvr(x) if x.size > 2 else np.nan for x in isi])

    # --- binned trains for entropy and mutual information ---
    bw = float(sfg["entropy_bin_ms"])
    binned = [histogram(t, bw, (t_range_ms[0] / 1000.0, t_range_ms[1] / 1000.0)).astype(np.int64)
              for t in trains]
    ent = np.array([entropy_bits(b) for b in binned])
    codes = [code_counts(b) for b in binned]

    # --- pairwise ---
    # STTC (Cutts & Eglen 2014) at every sttc_dt_report_ms; sttc_dt_ms is primary
    dt = float(sfg["sttc_dt_ms"])
    dts = [float(x) for x in sfg.get("sttc_dt_report_ms") or [dt]]
    if dt not in dts:
        dts.append(dt)
    # recording window for every pair, so values are comparable across pairs
    win = t_range_ms

    mi = np.full((n, n), np.nan)
    st_by_dt = {d: np.full((n, n), np.nan) for d in dts}
    for i in range(n):
        for j in range(i + 1, n):
            v = mutual_information_bits(codes[i], codes[j])
            mi[i, j] = mi[j, i] = v
            for d in dts:
                c = sttc(trains[i], trains[j], d, win)
                st_by_dt[d][i, j] = st_by_dt[d][j, i] = c
    st = st_by_dt[dt]

    agg = "harmonic" if compat.get("population_rate") == "legacy_harmonic" else "arithmetic"

    # CV^2 reported beside the count Fano factor; they measure different things
    cv_sq = np.asarray(isi_cv, float) ** 2

    return {
        "n_spikes": np.asarray(n_units_field, float),
        "isi_mean": isi_mean, "isi_cv": isi_cv,
        "fano": ff,                    # Var(N)/Mean(N) on counts
        "fano_op": ff_op,              # the same, window = w_o mean ISIs
        "fano_cv2": cv_sq,             # CV of ISI, squared
        "rate_isi": rate_isi, "rate_count": rate_cnt, "rate": rate,
        "lvr": lv,
        "entropy": ent, "mutual_info": mi,
        "sttc": st, "sttc_by_dt": st_by_dt,
        "population": {
            "fanoFactor": ml.mean(ff),            # Var(N)/Mean(N), window below
            "fanoFactorCV2": ml.mean(cv_sq),      # CV(ISI)^2
            "fanoWindow_ms": float(sfg["fano_window_ms"]),
            "fanoFactorOp": ml.mean(ff_op),       # operational time, window below
            "fanoOpWindow_isi": w_o,
            # units behind fanoFactorOp (slow units cannot fill min_windows)
            "fanoOpUnits": int(np.isfinite(ff_op).sum()),
            "sttcDt_ms": dt,                      # the dt behind `sttc`
            # signed STTC, plus |STTC| for old numbers; |STTC| cannot reach 0
            **{f"sttc_dt{int(d)}": ml.mean(ml_colmeans(m))
               for d, m in sorted(st_by_dt.items())},
            **{f"sttc_dt{int(d)}_abs": ml.mean(ml_colmeans(np.abs(m)))
               for d, m in sorted(st_by_dt.items())},
            "sttcNegativeFraction": float(np.nanmean(st[np.triu_indices(n, 1)] < 0))
                                    if n > 1 else np.nan,
            "firingRateISI": ml.mean(rate_isi),
            "firingRateCount": ml.mean(rate_cnt),
            "firingRate": population_mean(rate, agg),
            "entropy": ml.mean(ent),
            "lvr": ml.mean(lv),                   # on ISIs, the definition
            "mutualInfo": ml.mean(ml_colmeans(mi)),
            "sttc": ml.mean(ml_colmeans(st)),   # signed; the primary dt
        },
    }


def ml_colmeans(M):
    """NaN-omitting column means of M (MATLAB mean(M,'omitnan'))."""
    return np.array([ml.mean(M[:, j]) for j in range(M.shape[1])])


# --- level B ---
def split_populations(n_burst, cfg):
    """Split units into bursting vs tonic by burst count.

    Threshold: max(min_n, mean - 2*sd) over units with more than min_n bursts.
    """
    min_n = int(cfg["experiment"].get("min_bursts_per_neuron", 5))
    nb = np.asarray(n_burst, float)
    seed = nb[nb > min_n]
    thr = min_n
    if seed.size:
        thr = max(min_n, ml.mean(seed) - 2.0 * ml.std(seed))
    return np.flatnonzero(nb > thr), np.flatnonzero(nb <= thr), thr


def burst_populations(bursts, feats, cfg):
    """Population burst, non-burst and tonic features."""
    compat = cfg.get("compat", {})
    agg = "harmonic" if compat.get("population_rate") == "legacy_harmonic" else "arithmetic"
    n_clusters = len(feats)
    nb = np.array([b.n for b in bursts], float)
    bursting, tonic, thr = split_populations(nb, cfg)

    def rate_from_interval(intervals_ms, to_min=False):
        """Rate from intervals (ms): harmonic = 1/mean(interval); arithmetic = mean of per-unit rates."""
        k = 1.0 / 60000.0 if to_min else 1.0 / 1000.0
        x = np.asarray(intervals_ms, float)
        if agg == "harmonic":
            m = ml.mean(x)
            return 1.0 / (m * k) if np.isfinite(m) and m else np.nan
        return population_mean(1.0 / (x * k), "arithmetic")

    out = {"n_bursting": bursting.size, "n_tonic": tonic.size,
           "nburst_threshold": thr, "burst": {}, "nonburst": {}, "tonic": {}}

    if bursting.size:
        g = lambda k: np.array([feats[i][k] for i in bursting], float)
        amp, brate = g("isi_mean"), np.asarray([feats[i]["ibi_mean"] for i in bursting], float) * 1000.0
        nbr = g("nonburst_isi_mean")
        out["burst"] = {
            "neurons": bursting.size / n_clusters,
            "nSpikes_norm": ml.mean(g("nspikes_norm")),
            "duration_mean": ml.mean(g("duration_mean")),
            "duration_cv": ml.cv(g("duration_mean")),
            "amplitude_mean": rate_from_interval(amp),
            "amplitude_cv": ml.cv(amp),
            "amplitude_ff": ml.mean([ml.mean(_ff_of(feats[i]["isi_cv_per_burst"]))
                                      for i in bursting]),
            "rate_mean": rate_from_interval(brate, to_min=True),
            "rate_cv": ml.cv(brate),
            "rate_ff": ml.mean(_ff_of(g("ibi_cv"))),
        }
        out["nonburst"] = {
            "nSpikes_norm": ml.mean(g("nonburst_n_norm")),
            "spikeRate_mean": rate_from_interval(nbr),
            "spikeRate_cv": ml.cv(nbr),
            "spikeRate_ff": ml.mean(_ff_of(g("nonburst_isi_cv"))),
        }
    else:
        out["burst"] = {"neurons": 0.0}

    if tonic.size:
        t_isi = np.array([feats[i]["nonburst_isi_mean"] for i in tonic], float)
        out["tonic"] = {
            "neurons": tonic.size / n_clusters,
            "spikeRate_mean": rate_from_interval(t_isi),
            "spikeRate_cv": ml.cv(t_isi),
            "spikeRate_ff": ml.mean(_ff_of(
                np.array([feats[i]["nonburst_isi_cv"] for i in tonic], float))),
        }
    else:
        out["tonic"] = {"neurons": 0.0}
    return out


def _ff_of(cv_values, compat=None):
    """CV squared; burst `_ff` fields have no counting window for a true Fano factor."""
    return np.asarray(cv_values, float) ** 2


# --- level C and driver ---
def network_level(trains, bursts, feats, t_range_s, cfg, streams):
    """Level C: network bursts and burst families for one condition.

    With no events, or clustering disabled, families/features/structure are None.
    """
    import numpy as np
    from ..network import detect as ndet, players as npl, features as nfeat
    from ..network import structure as nstruct, families as nfam

    nb = cfg.get("network_burst", {})
    cl = cfg.get("clustering", {})
    labels = [b.label for b in bursts]
    isi_mean = np.array([f.get("isi_mean", np.nan) for f in feats], float)

    det = ndet.run(trains, labels, isi_mean, t_range_s, streams,
                   threshold=float(nb.get("threshold_sd", 5.0)),
                   opt_distance=int(bool(nb.get("remove_echo", True))),
                   n_shuffling=int(nb.get("n_surrogates", 500)), nb=nb)
    if not det["IndivBurst"]:
        return {"detected": det, "families": None, "features": None,
                "structure": None}

    ch = det["Channels"] - 1
    ev, _, _ = npl.event_spike_times(trains, det["IndivBurst"],
                                     float(nb.get("safe_k") or npl.SAFE_K))
    n_spk = np.array([t.size for t in trains], float)
    positions = cfg.get("_positions") or [np.zeros(2)] * len(trains)

    n_ev = len(det["IndivBurst"])
    scalar = np.full((n_ev, nfam.N_SCALAR), np.nan)
    base = nfeat.features(det["IndivBurst"], ev, ch, np.ones(n_ev, bool),
                          len(trains), n_spk, ch.size, det["bin"])
    for k, m in enumerate(base["indivMetric"]):
        scalar[k, :4] = [m["duration"], m["nSpikes"], m["nNeuronsAct"], m["isi"]]
        if k > 0:
            scalar[k, 4] = det["IndivBurst"][k]["peak"] - det["IndivBurst"][k - 1]["peak"]
    pop = np.zeros((n_ev, len(trains)), bool)
    for k in range(n_ev):
        for u in range(len(trains)):
            pop[k, u] = len(ev[u][k]) > 0

    compat = cfg.get("compat", {})
    kw = dict(nan_aware=compat.get("cluster_metrics", "nan_aware") == "nan_aware",
              scale_mode=compat.get("scalar_scale", "robust"),
              empty_pair=compat.get("jaccard_empty", "zero"),
              w_scalar=float(cl.get("w_scalar", 0.35)),
              w_population=float(cl.get("w_population", 0.65)))
    # clustering.enabled: false skips families (saves little; detection dominates)
    if not cl.get("enabled", True):
        return {"detected": det, "families": None, "features": None,
                "structure": None}

    fams = nfam.cluster_events(
        scalar, pop, streams.for_item("families"),
        plus_one=compat.get("permutation_p", "plus_one") == "plus_one",
        n_perm=int(cl.get("n_permutation", 1000)),
        n_boot=int(cl.get("n_bootstrap", 1000)),
        max_k=int(cl.get("max_k", 4)),
        linkage_method=str(cl.get("linkage", "ward")),
        alpha=float(cl.get("alpha", 0.05)),
        sil_threshold=float(cl.get("silhouette_threshold", 0.40)),
        has_secondary=det["Secondary"]["nBurst"] > 0, **kw)

    per_family, struct = [], []
    for fam_id in np.unique(fams["label"]):
        mask = fams["label"] == fam_id
        per_family.append(nfeat.features(det["IndivBurst"], ev, ch, mask,
                                         len(trains), n_spk, ch.size, det["bin"]))
        struct.append(nstruct.structure(det["IndivBurst"], ev, ch, mask, positions))
    return {"detected": det, "families": fams, "features": per_family,
            "structure": struct}


def run_recording(rec, cfg, streams=None):
    """Levels A, B and C for every condition of one recording."""
    from ..core.windows import restrict
    out = []
    window_s = float(cfg["experiment"]["window_s"])
    for _, row in rec.conditions.iterrows():
        t0, t1 = float(row["t0_s"]), float(row["t1_s"])
        if window_s > 0 and t0 < t1 - window_s:
            t0 = t1 - window_s
        rng_ms = (t0 * 1000.0, t1 * 1000.0)
        trains = [restrict(rec.train(i), rng_ms[0], rng_ms[1]) for i in range(rec.n_units)]
        n_field = np.array([t.size for t in trains], float)
        recL = np.full(rec.n_units, t1 - t0)
        sf = spike_features(trains, n_field, recL, rng_ms, cfg)
        bursts = [detect(t, cfg) for t in trains]
        feats = [features(b) for b in bursts]
        bp = burst_populations(bursts, feats, cfg)
        entry = {"condition": row["name"], "window_s": (t0, t1),
                 "spike": sf, "bursts": bursts, "burst_features": feats,
                 "burst_population": bp}
        if streams is not None and cfg.get("network_burst", {}).get("enabled", True):
            cfg = dict(cfg)
            cfg["_positions"] = [rec.units[["x", "y"]].to_numpy(float)[i]
                                 for i in range(rec.n_units)] \
                if {"x", "y"}.issubset(rec.units.columns) else None
            entry["network"] = network_level(trains, bursts, feats,
                                             (t0, t1), cfg, streams)
        out.append(entry)
    res = {"per_condition": out}
    res["metrics"] = metrics_table(res)      # saved as metrics.parquet

    # --- per-unit network-burst participation, (condition, unit) ---
    part = np.zeros((len(out), rec.n_units), bool)
    part_p = np.full((len(out), rec.n_units), np.nan)
    part_ev = np.full((len(out), rec.n_units), np.nan)
    for k, entry in enumerate(out):
        det = (entry.get("network") or {}).get("detected") or {}
        p = det.get("participation")
        if p is not None and len(p) == rec.n_units:
            part[k] = np.asarray(p, bool)
            part_p[k] = np.asarray(det.get("participation_p"), float)
            part_ev[k] = np.asarray(det.get("participation_events"), float)
        else:                       # fall back to Channels (1-based)
            chans = np.asarray(det.get("Channels", []), int) - 1
            chans = chans[(chans >= 0) & (chans < rec.n_units)]
            part[k, chans] = True
    res["burst_participant"] = part
    res["burst_participation_p"] = part_p
    res["burst_participation_events"] = part_ev

    for key, dest in (("screen_meaningful", "burst_screen_meaningful"),
                      ("screen_bursting_fraction", "burst_screen_fraction")):
        arr = np.full((len(out), rec.n_units), np.nan)
        for k, entry in enumerate(out):
            det = (entry.get("network") or {}).get("detected") or {}
            v = det.get(key)
            if v is not None and len(v) == rec.n_units:
                arr[k] = np.asarray(v, float)
        res[dest] = arr
    return res


# --- export ---
def _walk(prefix, obj, out, level):
    """Flatten nested metric dicts into (level, metric, value) rows; scalars only."""
    import numpy as _np
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("values", "coords", "indivMetric", "populationFiring",
                     "neuronFiring", "structure", "mapPosition", "label",
                     "silhouette", "bootstrap", "permutation", "IndivBurst",
                     "spikesBin", "smoothed_spikesBin", "time", "Channels",
                     "ChannelsIntegrated", "Secondary", "counts",
                     "isi_mean_per_burst", "isi_cv_per_burst", "duration_s"):
                continue
            _walk(f"{prefix}.{k}" if prefix else k, v, out, level)
        return
    if isinstance(obj, (int, float, _np.floating, _np.integer)) and not isinstance(obj, bool):
        out.append({"level": level, "metric": prefix, "value": float(obj)})


def metrics_table(result):
    """Long table of every scalar metric: condition x level x metric -> value."""
    import pandas as pd
    rows = []
    for entry in result.get("per_condition", []):
        cond = entry["condition"]
        base = len(rows)
        _walk("", entry.get("spike") or {}, rows, "A_unit")
        _walk("", entry.get("burst_population") or {}, rows, "B_burst")
        net = entry.get("network") or {}
        if net.get("detected"):
            d = dict(net["detected"])
            _walk("", {"nBurst": d.get("nBurst"), "bin_s": d.get("bin"),
                       "n_channels": len(d.get("Channels", [])),
                       "lowUnits": d.get("lowUnits"),
                       # fraction_active only: median share of units active; flag = chance overlap risk
                       "background_fraction": d.get("background"),
                       "background_flag": (float(d["background_flag"])
                                           if "background_flag" in d else None)},
                  rows, "C_network")
        if net.get("families"):
            f = net["families"]
            _walk("", {"n_cluster": f.get("n_cluster"),
                       "quality": f.get("clusteringQuality")}, rows, "C_families")
        for i, feats in enumerate(net.get("features") or [], start=1):
            _walk("", feats, rows, f"C_family{i}")
        # Window and unit count on every row (MATLAB T_start, T_end, Number_Good_Clusters)
        t0, t1 = entry.get("window_s", (float("nan"),) * 2)
        n_units = len((entry.get("spike") or {}).get("rate", []))
        for r in rows[base:]:
            r["condition"] = cond
            r["t_start_s"] = t0
            r["t_end_s"] = t1
            r["n_units"] = n_units
    cols = ["condition", "level", "metric", "value",
            "t_start_s", "t_end_s", "n_units"]
    df = pd.DataFrame(rows, columns=cols)
    return df.dropna(subset=["metric"]).reset_index(drop=True)
