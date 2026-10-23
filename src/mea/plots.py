"""Per-recording diagnostic figures (PDF by default; `plots.enabled`, `plots.figures`).

Matplotlib is imported lazily, so the pipeline runs without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# min dpi for the rasterized spike layer of raster/overview
RASTER_DPI = 300

FIGURES = ("raster", "activity_per_condition", "electrode_scatter", "electrode_map",
           "overview", "isi", "ttx", "waveforms", "unit_QC", "selection_drops",
           "metrics")
# old names still accepted in plots.figures
RENAMED = {"funnel": "unit_QC", "conditions": "metrics",
           "footprint": "electrode_scatter", "population": "activity_per_condition"}


def _mpl(cfg):
    """Matplotlib with a non-interactive backend, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")            # no display on a cluster node
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    return matplotlib, plt


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def _condition_bands(ax, rec, y=None):
    """Shade alternate conditions and label them."""
    for i, row in enumerate(rec.conditions.itertuples()):
        if i % 2:
            ax.axvspan(row.t0_s, row.t1_s, color="0.92", zorder=0, lw=0)
        ax.axvline(row.t1_s, color="0.7", lw=0.6, zorder=1)
    # labels inside the axes, to avoid colliding with the title
    lo, hi = ax.get_ylim()
    top = hi - 0.02 * (hi - lo) if y is None else y
    for row in rec.conditions.itertuples():
        label = str(row.name)
        # strip the recording-name prefix
        label = label.replace(rec.name, "").strip("_") or "baseline"
        ax.text((row.t0_s + row.t1_s) / 2, top, label[:22],
                ha="center", va="top", fontsize=7, color="0.30",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.7, pad=1.2))


# --- figures ---
def raster(rec, path, cfg=None, max_units=400, max_spikes=200_000):
    """Spike raster over the whole timeline, conditions marked.

    Subsampled to `max_units` units and `max_spikes` spikes; the title says so.
    """
    _, plt = _mpl(cfg)
    if plt is None:
        return None
    n = rec.n_units
    idx = np.arange(n)
    if n > max_units:
        idx = np.linspace(0, n - 1, max_units).astype(int)

    xs, ys = [], []
    for row, i in enumerate(idx):
        t = rec.train(i)
        if t.size:
            xs.append(t / 1000.0)
            ys.append(np.full(t.size, row))
    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    note = ""
    if x.size > max_spikes:                  # thin for rendering only
        keep = np.linspace(0, x.size - 1, max_spikes).astype(int)
        x, y = x[keep], y[keep]
        note = f", {max_spikes:,} of {sum(s.size for s in xs):,} spikes drawn"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(x, y, s=0.4, marker="|", color="#14181F", linewidths=0.4,
               rasterized=True)           # one image, not 200k vector marks
    ax.set_ylim(-1, len(idx) * 1.04)      # headroom for the condition labels
    ax.set_xlim(rec.conditions["t0_s"].min(), rec.conditions["t1_s"].max())
    _condition_bands(ax, rec)
    _style(ax, f"{rec.name} -- raster"
               + (f" ({len(idx)} of {n} units{note})" if len(idx) < n or note else
                  f" ({n} units)"),
           "time (s)", "unit")
    fig.tight_layout()
    fig.savefig(path, dpi=max((cfg or {}).get("dpi", 150), RASTER_DPI))
    plt.close(fig)
    return path


def population(rec, result, path, cfg=None):
    """Population rate per condition with detected network bursts marked."""
    _, plt = _mpl(cfg)
    if plt is None or not result:
        return None
    per = result.get("per_condition") or []
    if not per:
        return None

    # shared y axis across conditions
    fig, axes = plt.subplots(len(per), 1, figsize=(11, 1.9 * len(per)),
                             sharex=False, sharey=True, squeeze=False)
    for ax, entry in zip(axes[:, 0], per):
        t0, t1 = entry["window_s"]
        det = (entry.get("network") or {}).get("detected") or {}

        # Draw the detector's own trace (participating units, per-unit normalised)
        t = np.asarray(det.get("time", []), float)
        y = np.asarray(det.get("spikesBin", []), float)
        bin_s = float(det.get("bin") or 0.2)
        if t.size and y.size:
            n = min(t.size, y.size)
            t, y = t[:n], y[:n]
        else:                       # no detection: re-bin all spikes
            edges = np.arange(t0, t1 + bin_s, bin_s)
            y = np.zeros(edges.size - 1)
            for i in range(rec.n_units):
                ts = rec.train(i) / 1000.0
                ts = ts[(ts >= t0) & (ts < t1)]
                if ts.size:
                    y += np.histogram(ts, bins=edges)[0]
            t = edges[:-1]
        ax.fill_between(t, y, step="post", color="#0D5C6B", alpha=.35, lw=0)
        ax.plot(t, y, drawstyle="steps-post", color="#0D5C6B", lw=.7)

        peaks = [b.get("peak") for b in det.get("IndivBurst", [])]
        n_b = det.get("nBurst", 0)
        for pk in peaks:
            if pk is not None and np.isfinite(pk):
                ax.axvline(float(pk), color="#8C3B3B", lw=.8, alpha=.85)
        label = (f"{len(peaks)} events" if np.isfinite(n_b) and n_b
                 else f"{len(peaks)} events, not network bursting")
        _style(ax, f"{entry['condition']}  --  {label}   [bin {bin_s:.2f} s]",
               None, det.get("trace_label", "normalised rate"))
        if det.get("threshold") is not None:
            ax.axhline(float(det["threshold"]), color="#5C6672", ls=":", lw=.8)
        ax.set_xlim(t0, t1)
    axes[-1, 0].set_xlabel("time (s)", fontsize=9)
    fig.suptitle(f"{rec.name} -- population activity", fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path


def _electrode_xy(rec, grid=(64, 64)):
    """Unit positions as electrode indices; microns (from channel_positions.npy) are converted."""
    x = rec.units["x"].to_numpy(float)
    y = rec.units["y"].to_numpy(float)
    span = max(np.nanmax(x) if x.size else 0, np.nanmax(y) if y.size else 0)
    if span > max(grid):                       # microns, not indices
        pitch = span / (max(grid) - 1)
        x, y = x / pitch, y / pitch
    return x, y


def footprint(rec, path, cfg=None, grid=(64, 64)):
    """Surviving units on the electrode array, sized by rate."""
    _, plt = _mpl(cfg)
    if plt is None:
        return None
    if not {"x", "y"}.issubset(rec.units.columns):
        return None
    x, y = _electrode_xy(rec, grid)
    dur = rec.duration_s()
    rate = np.array([rec.train(i).size / dur if dur else 0
                     for i in range(rec.n_units)], float)

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    sizes = 6 + 90 * (rate / rate.max() if rate.max() else rate)
    sc = ax.scatter(x, y, s=sizes, c=rate, cmap="viridis", alpha=.85,
                    edgecolors="none")
    ax.set_xlim(-1, grid[0]); ax.set_ylim(-1, grid[1])
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.04)
    cb.set_label("firing rate (Hz)", fontsize=8); cb.ax.tick_params(labelsize=7)
    _style(ax, f"{rec.name} -- {rec.n_units} units on the array", "electrode x", "electrode y")
    fig.tight_layout()
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path


# QC steps in pipeline order: (bar label, stage, flag, qc_general test)
# a qc_general flag counts only if its test is in qc_general.tests
UNIT_QC_STEPS = (
    ("TTX-sensitive", "qc_general", "not_ttx_sensitive", "ttx"),
    ("not echo", "qc_general", "echo", "echo"),
    ("not silent", "qc_general", "silent", "silent"),
    ("not noisy", "qc_general", "noisy", "noisy"),
    ("inside photo", "qc_general", "outside_picture", "picture"),
    ("no crosstalk", "qc_manual", "hypersynchrony", None),
    ("active", "qc_manual", "below_min_activity", None),
)
FINAL_STEP = ("final non-noise units", "integrate", "unit_type")


def unit_qc_counts(rec, tests=None):
    """Units left after each QC step, in pipeline order, from `rec.qc`.

    Returns [(label, n_left, n_removed)], from all clusters to the final units;
    a unit failing several tests is counted at the first.
    """
    from .stages.s02_qc_general import ALL_TESTS
    tests = set(ALL_TESTS if tests is None else tests)
    qc = rec.qc
    marked = {} if qc is None or not len(qc) else {
        k: set(g["unit_id"]) for k, g in qc.groupby(["stage", "flag"])}
    kept_ids = set(rec.units["unit_id"])
    every = kept_ids.union(*marked.values()) if marked else set(kept_ids)

    steps = [(lab, st, fl) for lab, st, fl, test in UNIT_QC_STEPS
             if test is None or test in tests]
    known = {(st, fl) for _, st, fl, _ in UNIT_QC_STEPS} | {FINAL_STEP[1:]}
    steps += [(f"{st}: {fl}", st, fl) for st, fl in sorted(marked) if (st, fl) not in known]
    steps.append(FINAL_STEP)

    out = [("all clusters", len(every), 0)]
    left = set(every)
    for lab, st, fl in steps:
        gone = left & marked.get((st, fl), set())
        left -= gone
        out.append((lab, len(left), len(gone)))
    return out


# bombcell's Kilosort 4 defaults, used when the sort's params can't be found
BOMBCELL_BASELINE_DEFAULT = {"waveform_baseline_window_start": 0,
                             "waveform_baseline_window_stop": 10,
                             "maxWvBaselineFraction": 0.3}
# qc_general flag -> test that discards on it
QC_GENERAL_TEST = {"echo": "echo", "silent": "silent", "not_ttx_sensitive": "ttx",
                   "noisy": "noisy", "outside_picture": "picture"}


def _bombcell_baseline_params(rec):
    """bombcell's baseline window and limit as the sort ran them, and where from."""
    import json
    import tarfile
    src = Path(str(rec.meta.get("imported_from", "") or ""))
    texts = []
    if src.is_dir():
        texts += [f.read_text for f in sorted(src.glob("kilosort4_*/sorter_output/bombcell_param.json"))]
        # archived sorts: inside the tarball
        for tgz in sorted(src.glob("kilosort4_*.tar.gz")):
            def _from_tar(tgz=tgz):
                with tarfile.open(tgz) as tf:
                    m = next(m for m in tf if m.name.endswith("sorter_output/bombcell_param.json"))
                    return tf.extractfile(m).read().decode()
            texts.append(_from_tar)
    for read in texts:
        try:
            p = json.loads(read())
            return {k: p.get(k, v) for k, v in BOMBCELL_BASELINE_DEFAULT.items()}, "sort"
        except (OSError, ValueError, StopIteration, tarfile.TarError):
            continue
    return dict(BOMBCELL_BASELINE_DEFAULT), "bombcell default"


def selection_split(rec, tests=None):
    """Indices of QC-passing units, split by the selection step's verdict.

    Returns (kept, dropped); selection drops bombcell noise not called Kilosort "good".
    """
    from .stages.s02_qc_general import ALL_TESTS
    tests = set(ALL_TESTS if tests is None else tests)
    qc = rec.qc
    ids = rec.units["unit_id"].to_numpy()
    if qc is None or not len(qc):
        return np.arange(len(ids)), np.array([], int)
    applied = qc[(qc["stage"] != "qc_general") |
                 qc["flag"].map(lambda f: QC_GENERAL_TEST.get(f) in tests)]
    failed_qc = set(applied.loc[applied["stage"] != "integrate", "unit_id"])
    dropped_sel = set(applied.loc[applied["stage"] == "integrate", "unit_id"])
    passed = np.array([u not in failed_qc for u in ids])
    at_sel = np.array([u in dropped_sel for u in ids])
    return np.flatnonzero(passed & ~at_sel), np.flatnonzero(passed & at_sel)


def selection_drops(rec, path, cfg=None, max_per_group=40):
    """Waveforms of QC-passing units, kept vs dropped at selection, with bombcell's baseline window.

    Skipped when the baseline limit is >= 1 (the rule can drop nothing).
    """
    _, plt = _mpl(cfg)
    if plt is None or rec.waveforms is None or rec.n_units == 0:
        return None
    bp, source = _bombcell_baseline_params(rec)
    lim = float(bp["maxWvBaselineFraction"])
    if lim >= 1.0:
        return None
    W = np.asarray(rec.waveforms, float)
    if W.ndim != 2 or len(W) != rec.n_units:
        return None
    kept, dropped = selection_split(rec, (cfg or {}).get("_qc_tests"))
    if kept.size + dropped.size == 0:
        return None
    w0, w1 = int(bp["waveform_baseline_window_start"]), int(bp["waveform_baseline_window_stop"])
    fs = float(rec.meta.get("sampling_hz") or 0) or None
    trough = int(np.median(np.argmax(np.abs(W), axis=1))) if len(W) else 0
    x = (np.arange(W.shape[1]) - trough) / fs * 1000.0 if fs else np.arange(W.shape[1])
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, idx, col, name in ((axes[0], kept, "#2a6f97", "kept by selection"),
                               (axes[1], dropped, "#b5462e", "dropped at selection")):
        ax.axvspan(x[min(w0, len(x) - 1)], x[min(max(w1 - 1, w0), len(x) - 1)],
                   color="#e8e2c8", zorder=0)
        for yv in (lim, -lim):
            ax.axhline(yv, color="#5C6672", ls=":", lw=1)
        if idx.size:
            Y = W[idx] / np.maximum(np.max(np.abs(W[idx]), axis=1, keepdims=True), 1e-12)
            show = rng.choice(idx.size, min(max_per_group, idx.size), replace=False)
            for k in show:
                ax.plot(x, Y[k], color=col, lw=.6, alpha=.35)
            ax.plot(x, np.median(Y, axis=0), color=col, lw=2.6)
        ax.set_ylim(-1.1, 1.1)
        _style(ax, f"{name} (n = {idx.size})", "ms" if fs else "sample", None)
    axes[0].set_ylabel("waveform / its peak")
    fig.suptitle(f"{rec.name} -- units that passed QC.  Shaded: bombcell's baseline window "
                 f"(samples {w0}-{w1}); dotted: ±{lim:g}, the limit inside it ({source})",
                 fontsize=9, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path


def unit_qc(rec, path, cfg=None):
    """Bar chart of units left after each QC step, annotated with removals."""
    _, plt = _mpl(cfg)
    if plt is None:
        return None
    qc = rec.qc
    if qc is None or not len(qc):
        return None
    rows = unit_qc_counts(rec, (cfg or {}).get("_qc_tests"))
    labels = [r[0] for r in rows]
    n = [r[1] for r in rows]
    removed = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(max(7.5, 1.05 * len(rows) + 1.5), 4.2))
    colors = ["#8A949B"] + ["#6E9CB8"] * (len(rows) - 2) + ["#1F5A7D"]
    ax.bar(range(len(rows)), n, color=colors, width=.66)
    top = max(n) if n else 1
    for i, (v, r) in enumerate(zip(n, removed)):
        ax.text(i, v + top * .015, f"{v:,}", ha="center", va="bottom", fontsize=9,
                color="#1D2A35", fontweight="bold" if i in (0, len(rows) - 1) else "normal")
        if i:
            ax.text(i, v + top * .075, f"−{r:,}", ha="center", va="bottom",
                    fontsize=8, color="#8C3B3B")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, top * 1.18)
    title = f"{rec.name} -- unit QC: {n[0]:,} clusters → {n[-1]:,} final non-noise units (red: removed at each step)"
    # flag count mismatches in the title
    n_sorted = int(rec.meta.get("n_cluster_original", 0) or 0)
    if n_sorted and n_sorted != n[0]:
        title += f"\nthe sorter reported {n_sorted:,} clusters; {n_sorted - n[0]:,} are not in the QC record"
    if n[-1] != rec.n_units:
        title += f"\nrecording holds {rec.n_units:,} units, not {n[-1]:,}"
    _style(ax, title, None, "units left")
    fig.tight_layout()
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path


# (panel title, [(metric key, label), ...]); Fano factor drawn under both definitions
PANELS = [
    ("firing rate (Hz)", [("firingRate", "firingRate")]),
    ("Fano factor", [("fanoFactor", "Var(N)/Mean(N)"),
                     ("fanoFactorCV2", "CV(ISI)^2")]),
    ("Local variation (Lv)", [("lvr", "Lv")]),
    ("entropy (bits)", [("entropy", "entropy")]),
    ("synchrony (STTC)", [("sttc", "STTC")]),
    ("units bursting (fraction)", [("bursting_fraction", "bursting")]),
    ("network bursts (count)\ngap = not network bursting", [("network/detected/nBurst", "events")]),
]


def _series_value(entry, key):
    """One condition's value for a panel series; NaN if absent.

    Plain key: level-A population metric. Slashed key: path into the result
    (e.g. network/detected/nBurst). `bursting_fraction` = n_bursting / (n_bursting + n_tonic).
    """
    if key == "bursting_fraction":
        bp = entry.get("burst_population") or {}
        nb, nt = bp.get("n_bursting"), bp.get("n_tonic")
        try:
            return float(nb) / (float(nb) + float(nt)) if (nb + nt) else np.nan
        except (TypeError, ValueError):
            return np.nan
    if "/" in key:
        v = entry
        for part in key.split("/"):
            v = v.get(part) if isinstance(v, dict) else None
            if v is None:
                return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return (entry.get("spike") or {}).get("population", {}).get(key, np.nan)

SERIES_STYLE = {                       # one colour per definition
    "fanoFactor":       ("#0D5C6B", "o-"),
    "fanoFactorCV2":    ("#3E8E7E", "s--"),
    "lvr":              ("#0D5C6B", "o-"),
}


def conditions(rec, result, path, cfg=None, metrics=None, title=None):
    """Population metrics across conditions, one panel each.

    `metrics`: plain metric names, or `(title, [(name, label), ...])` panels.
    """
    _, plt = _mpl(cfg)
    if plt is None or not result:
        return None
    per = result.get("per_condition") or []
    if not per:
        return None
    panels = _panels(metrics)
    # rec may be None for a legacy _Analyzed.mat result
    rec_name = getattr(rec, "name", None) or result.get("name", "")
    names = [e["condition"] for e in per]
    short = [n.replace(rec_name, "").strip("_") or "baseline" for n in names]

    ncol = min(4, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (panel_title, series) in zip(axes.flat, panels):
        drawn = 0
        for key, label in series:
            vals = [_series_value(e, key) for e in per]
            if not np.any(np.isfinite(np.asarray(vals, float))):
                continue           # metric absent, e.g. a legacy file's CV^2
            color, style = SERIES_STYLE.get(key, ("#0D5C6B", "o-"))
            ax.plot(range(len(vals)), vals, style, color=color, ms=4, lw=1.2,
                    label=label)
            drawn += 1
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
        _style(ax, panel_title, None, None)
        if len(series) > 1:
            # log: series differ several-fold
            ax.set_yscale("log")
        if drawn > 1:
            ax.legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle(title or f"{rec_name} -- metrics by condition",
                 fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - .05 / nrow))
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path


def _panels(metrics):
    """Accept the old flat list of names as well as the new panel form."""
    if not metrics:
        return PANELS
    out = []
    for m in metrics:
        if isinstance(m, str):
            out.append((m, [(m, m)]))
        else:
            out.append((m[0], list(m[1])))
    return out


def electrode_map(rec, path, cfg=None, grid=(64, 64), min_hz=0.01, smooth=2.0):
    """Smoothed firing density with unit positions on the array, per condition.

    Somatic (o) and non-somatic (x) units are marked separately.
    """
    # no units: skipped, not failed
    if rec.n_units == 0:
        return None

    _, plt = _mpl(cfg)
    if plt is None or not {"x", "y"}.issubset(rec.units.columns):
        return None
    x, y = _electrode_xy(rec, grid)
    counts, lengths = rec.per_condition()
    rates = np.divide(counts, lengths, out=np.zeros_like(counts), where=lengths > 0)
    names = list(rec.conditions["name"])
    kept = (rec.units["kept"].to_numpy(bool)
            if "kept" in rec.units.columns else np.ones(x.size, bool))

    # somatic = not inverted waveform
    somatic = np.ones(x.size, bool)
    if rec.waveforms is not None:
        from .curation.waveforms import shape_metrics
        fs = float(rec.meta.get("sampling_hz") or 19753.8)
        W = np.asarray(rec.waveforms, float)
        if W.ndim == 2 and len(W) == x.size:
            somatic = ~np.array([shape_metrics(w, fs)["inverted"] for w in W])

    ix = np.clip(np.rint(x).astype(int), 0, grid[0] - 1)
    iy = np.clip(np.rint(y).astype(int), 0, grid[1] - 1)

    ncol = min(len(names), 3)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.7 * nrow),
                             squeeze=False, constrained_layout=True)
    # one density scale across panels
    dens = []
    for k in range(len(names)):
        g = np.zeros(grid)
        np.add.at(g, (ix, iy), np.where(kept, rates[k], 0.0))
        try:
            from scipy.ndimage import gaussian_filter
            g = gaussian_filter(g, smooth)
        except ImportError:
            pass
        dens.append(g)
    vmax = max((d.max() for d in dens), default=1.0) or 1.0

    for k, name in enumerate(names):
        ax = axes[k // ncol][k % ncol]
        ax.imshow(dens[k].T, origin="lower", cmap="magma_r", vmin=0, vmax=vmax,
                  extent=(-.5, grid[0] - .5, -.5, grid[1] - .5), interpolation="bilinear")
        live = (rates[k] >= min_hz) & kept
        for mask, marker, size, lab in ((live & somatic, "o", 13, "somatic"),
                                        (live & ~somatic, "x", 15, "non-somatic")):
            if mask.any():
                ax.scatter(x[mask], y[mask], s=size, marker=marker,
                           facecolors="none" if marker == "o" else None,
                           edgecolors="#1560a8" if marker == "o" else None,
                           c=None if marker == "o" else "#c0504d",
                           linewidths=.7, label=lab)
        ax.set_xlim(-.5, grid[0] - .5); ax.set_ylim(-.5, grid[1] - .5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name.rsplit('.', 1)[0].replace(rec.name, '').strip('_') or 'baseline'}\n"
                     f"{int((live & somatic).sum())} somatic, "
                     f"{int((live & ~somatic).sum())} non-somatic",
                     fontsize=9, pad=5)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=.85)
    for k in range(len(names), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # drawn on survivors, so `kept` alone can't tell whether QC ran
    ran = (cfg or {}).get("_qc_ran")
    if ran is None:
        ran = not kept.all()
    held = (f"QC-passing ({int(kept.sum())})" if ran
            else f"all {int(kept.sum())} sorted (NO QC)")
    fig.suptitle(f"{rec.name} — where {held} units fire. Shading is smoothed "
                 f"firing density; one scale across panels.", fontsize=10)
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return path


def recording_overview(rec, path, cfg=None, bin_s=1.0):
    """Whole concatenated recording on one wide PDF page, compounds labelled.

    Spikes embedded as a RASTER_DPI image; axes stay vector.
    """
    _, plt = _mpl(cfg)
    if plt is None:
        return None
    dur = rec.duration_s()
    if not dur or not rec.n_units:
        return None

    trains = [rec.train(i) / 1000.0 for i in range(rec.n_units)]
    order = np.argsort([t[0] if t.size else np.inf for t in trains])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(max(16, min(dur / 90, 60)), 9),
        height_ratios=[3, 1], sharex=True)

    for row, u in enumerate(order):
        t = trains[u]
        if t.size:
            ax.plot(t, np.full(t.size, row), "|", ms=1.6, mew=.35,
                    color="#2b2b35", rasterized=True)
    ax.set_ylim(-1, max(len(order), 1)); ax.set_ylabel(f"unit ({len(order)})")

    spikes = [t for t in trains if t.size]
    allspikes = np.concatenate(spikes) if spikes else np.array([])
    edges = np.arange(0, dur + bin_s, bin_s)
    ax2.fill_between(edges[:-1], 0, np.histogram(allspikes, edges)[0] / bin_s,
                     color="#4a4a8a", lw=0)
    ax2.set_ylabel("population (Hz)"); ax2.set_xlabel("seconds, concatenated")

    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, c in enumerate(rec.conditions.itertuples()):
        for a_ in (ax, ax2):
            a_.axvspan(c.t0_s, c.t1_s, color=cmap[i % 10], alpha=.10, lw=0, zorder=0)
            if i:
                a_.axvline(c.t0_s, color="#d62728", lw=.9, ls="--", alpha=.8, zorder=3)
        ax.text((c.t0_s + c.t1_s) / 2, len(order) * 1.015, c.name, ha="center",
                va="bottom", fontsize=9, weight="bold", color=cmap[i % 10] * .7)

    for a_ in (ax, ax2):
        a_.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, dur)
    fig.suptitle(f"{rec.name} -- {dur/60:.1f} min concatenated, "
                 f"{len(order)} units", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=max((cfg or {}).get("dpi", 150), RASTER_DPI))
    plt.close(fig)
    return path



def isi_violations(rec, path, cfg=None, refractory_ms=1.0, limit=0.05):
    """Refractory violation fraction per unit against firing rate, with Poisson chance curve.

    `limit`: exclusion line (fraction, 5% by default).
    """
    _, plt = _mpl(cfg)
    if plt is None or not rec.n_units:
        return None
    dur = rec.duration_s()
    rate, frac = np.zeros(rec.n_units), np.zeros(rec.n_units)
    for i in range(rec.n_units):
        t = rec.train(i)
        rate[i] = t.size / dur if dur else 0.0
        if t.size > 1:
            frac[i] = float((np.diff(t) < refractory_ms).mean())

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bad = frac > limit
    ax.scatter(rate[~bad], 100 * frac[~bad], s=14, c="#3b6ea5",
               edgecolors="none", alpha=.8, label=f"pass ({int((~bad).sum())})")
    ax.scatter(rate[bad], 100 * frac[bad], s=20, c="#d62728",
               edgecolors="none", alpha=.85, label=f"over {limit:.0%} ({int(bad.sum())})")
    # Poisson expectation: P(next ISI < tau) = 1 - exp(-rate*tau)
    r = np.linspace(max(rate.min(), 1e-3), max(rate.max(), 1e-2), 200)
    ax.plot(r, 100 * (1 - np.exp(-r * refractory_ms / 1000.0)), color="#888",
            lw=1, ls=":", label="Poisson at the same rate")
    ax.axhline(100 * limit, color="#d62728", lw=1, ls="--", alpha=.7)
    ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, f"{rec.name} -- refractory violations ({refractory_ms:g} ms)",
           "firing rate (Hz)", f"spikes within {refractory_ms:g} ms (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path


def ttx_control(rec, path, cfg=None):
    """Peak vs TTX rate per unit, for s02's rule: TTX removes more than ttx_min_drop of the peak rate.

    Draws passing and non-passing units as disjoint sets.
    """
    if rec.n_units == 0:
        return None

    _, plt = _mpl(cfg)
    if plt is None or "is_ttx" not in rec.conditions.columns:
        return None
    ttx = rec.conditions["is_ttx"].to_numpy(bool)
    if not ttx.any() or ttx.all():
        return None

    GREY, BLUE, RED = "#b6b6c0", "#3b6ea5", "#c0504d"
    counts, lengths = rec.per_condition()
    rates = np.divide(counts, lengths, out=np.zeros_like(counts), where=lengths > 0)
    peak = np.nanmax(rates, axis=0)
    ttx_rate = np.nanmin(rates[ttx], axis=0)
    # display floor for the log axis only; the verdict uses unfloored rates
    floor = 1e-3
    b_, a_ = np.maximum(peak, floor), np.maximum(ttx_rate, floor)

    # floor the ratio itself, not the two rates
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(ttx_rate, peak, out=np.full(peak.shape, np.nan),
                          where=peak > 0)
    _pos = ratio[np.isfinite(ratio) & (ratio > 0)]
    rfloor = min(1e-3, float(_pos.min()) / 3) if _pos.size else 1e-3
    n_zero = int(np.sum(ratio == 0))
    ratio_plot = np.where(ratio == 0, rfloor, ratio)
    drawable = np.isfinite(ratio)          # peak == 0: no ratio exists

    min_drop = float((cfg or {}).get("_ttx_min_drop", 0.90))
    # round as s02 does (1.0 - 0.9 != 0.1)
    keep_frac = round(1.0 - min_drop, 12)
    # as s02: peak * (1 - min_drop) > min TTX rate
    passing = peak * keep_frac > ttx_rate

    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.7), squeeze=False,
                            gridspec_kw={"width_ratios": [1.15, 1.35]})
    ax, ax2 = axs[0]

    # TTX-passers that also survive every other QC step
    non_ttx = (rec.units["kept_non_ttx"].to_numpy(bool)
               if "kept_non_ttx" in rec.units.columns else None)
    basis = non_ttx if non_ttx is not None else (
        rec.units["kept"].to_numpy(bool) if "kept" in rec.units.columns else None)
    survive = (f", {int((passing & basis).sum())} after full QC"
               if basis is not None else "")
    rng_j = np.random.default_rng(0)          # seeded jitter

    # --- peak vs TTX, disjoint sets ---
    for label, mask, colour, dx in (("not passing", ~passing, GREY, 0.0),
                                    ("passing", passing, BLUE, 0.42)):
        if mask.sum() < 2:
            continue
        for k, vals in ((0, b_[mask]), (1, a_[mask])):
            v = ax.violinplot(np.log10(vals), positions=[k + dx], widths=.34,
                              showextrema=False, showmedians=True)
            for body in v["bodies"]:
                body.set_facecolor(colour); body.set_alpha(.55)
            v["cmedians"].set_color("#222")
            # jittered points on the TTX column only
            if k == 1:
                y = np.log10(vals)
                x = k + dx + rng_j.uniform(-.11, .11, y.size)
                big = mask.sum() > 200
                ax.scatter(x, y, s=3 if big else 11, c=colour,
                           alpha=.16 if big else .8, edgecolors="none",
                           linewidths=0,
                           # big set under the violin, small set over it
                           zorder=1.5 if big else 3)
        ax.plot([dx, 1 + dx],
                [np.log10(np.median(b_[mask])), np.log10(np.median(a_[mask]))],
                "-o", color=colour, lw=2, ms=5,
                label=f"{label} (n={int(mask.sum())}): "
                      f"{np.median(b_[mask]):.3f} → {np.median(a_[mask]):.3f} Hz")
    ax.set_xticks([0.21, 1.21]); ax.set_xticklabels(["peak", "TTX"])
    ax.set_ylabel("firing rate (log10 Hz)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax, title=f"TTX rate < {keep_frac:.0%} of peak "
                     f"(drop > {min_drop:.0%})")

    # --- per-unit ratio ---
    no = drawable & ~passing
    yes = drawable & passing
    ax2.scatter(peak[no], ratio_plot[no], s=9, c=GREY, edgecolors="none",
                alpha=.55, label=f"not passing ({int((~passing).sum())})")
    ax2.scatter(peak[yes], ratio_plot[yes], s=26, c=BLUE, edgecolors="none",
                alpha=.9, label=f"passing ({int(passing.sum())})")
    ax2.axhline(1.0, color="#888", lw=1, ls="--", alpha=.6)
    ax2.axhline(keep_frac, color=RED, lw=1, ls=":", alpha=.75)
    if n_zero:
        # mark the zero-ratio row at the floor
        ax2.axhline(rfloor, color="#888", lw=.8, ls="-", alpha=.45)
        ax2.text(.99, .012, f"TTX rate = 0  ({n_zero})", transform=ax2.transAxes,
                 ha="right", va="bottom", fontsize=7.5, color="#666")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("peak rate (Hz)")
    ax2.set_ylabel("TTX rate ÷ PEAK rate")
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax2, title=f"per unit — {int(passing.sum())} pass{survive}")

    fig.suptitle(f"{rec.name} — TTX response, {int(rec.n_units)} total units",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return path


# (column, panel title, class names); bombcell's splitGoodAndMua_NonSomatic gives five types
LABEL_SOURCES = (
    ("unitType_bombcell", "bombcell",
     {0: "noise", 1: "somatic good", 2: "somatic MUA",
      3: "non-somatic good", 4: "non-somatic MUA"}),
    ("unitType_ours", "ours (QC)", None),          # strings already
)


def waveforms(rec, path, cfg=None, max_per_class=40):
    """Normalised mean waveforms by class, one column per labeller (bombcell, ours).

    Returns None when the sort has no waveforms.
    """
    # no units: skipped, not failed
    if rec.n_units == 0:
        return None

    _, plt = _mpl(cfg)
    if plt is None or rec.waveforms is None:
        return None
    W = np.asarray(rec.waveforms, float)
    if W.ndim != 2 or len(W) != rec.n_units:
        return None

    present = [(col, title, names) for col, title, names in LABEL_SOURCES
               if col in rec.units.columns]
    if not present:
        return None

    # each unit normalised to its own peak
    fs = float(rec.meta.get("sampling_hz") or 19753.8)
    peak = np.nanmax(np.abs(W), axis=1)
    peak[peak == 0] = 1.0
    Wn = W / peak[:, None]
    t = np.arange(W.shape[1])

    ncol = len(present)
    classes = [sorted(set(rec.units[c].dropna())) for c, _, _ in present]
    nrow = max(len(k) for k in classes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 1.9 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    rng = np.random.default_rng(0)

    for j, ((col, title, names), keys) in enumerate(zip(present, classes)):
        vals = rec.units[col].to_numpy()
        for i in range(nrow):
            ax = axes[i][j]
            if i >= len(keys):
                ax.axis("off"); continue
            k = keys[i]
            idx = np.flatnonzero(vals == k)
            shown = rng.choice(idx, max_per_class, replace=False) \
                if idx.size > max_per_class else idx
            for u in shown:
                ax.plot(t, Wn[u], lw=.5, alpha=.35, color="#3b6ea5")
            if idx.size:
                ax.plot(t, np.nanmedian(Wn[idx], axis=0), lw=1.6, color="#c0504d")
            label = (names or {}).get(k, k)
            # trough-to-peak of the class median (rig artefact is ~0.35 ms)
            ttp = ""
            if idx.size:
                from .curation.waveforms import shape_metrics
                m = shape_metrics(np.nanmedian(W[idx], axis=0), fs)
                if np.isfinite(m["trough_to_peak_ms"]):
                    ttp = f"\n{m['trough_to_peak_ms']:.2f} ms t-p"
            ax.set_ylabel(f"{label}\nn={idx.size}{ttp}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10)
    for j in range(ncol):
        axes[-1][j].set_xlabel("sample", fontsize=8)

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#3b6ea5", lw=1, alpha=.6,
                               label="one unit"),
                        Line2D([], [], color="#c0504d", lw=1.8,
                               label="median of the class")],
               loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{rec.name} — mean waveform by unit, by labeller\n"
                 f"up to {max_per_class} per class, red = median, "
                 "each normalised to its own peak", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return path


# --- driver ---
def write_all(rec, cfg, result=None, out_dir=None, candidates=None):
    """Write every enabled figure for one recording; per-figure errors are reported, not raised.

    `candidates`: all units the QC judged, with `kept` = final verdict. Diagnostic
    figures (isi, ttx, waveforms, selection_drops) use it; the rest use `rec`.
    Returns {"written", "dir", "skipped"?, "failed"?}.
    """
    g = (cfg or {}).get("plots", {}) or {}
    if not g.get("enabled", True):
        return {"written": [], "skipped": "plots.enabled is false"}

    import matplotlib
    out = Path(out_dir or rec.artifact_dir().parent) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ext = g.get("format", "pdf")
    want = [RENAMED.get(f, f) for f in (g.get("figures") or FIGURES)]

    seen = rec if candidates is None else candidates
    # pass QC settings through so figures match the stage's verdict
    qcg = (cfg or {}).get("qc_general", {})
    # QC filtered iff the caller's candidates differ from the survivors
    qc_ran = (candidates is not None
              and getattr(candidates, "n_units", None) != getattr(rec, "n_units", None))
    g = {**g, "_qc_ran": qc_ran,
         "_ttx_min_drop": qcg.get("ttx_min_drop", 0.90),
         "_qc_tests": qcg.get("tests")}
    jobs = {
        # results: survivors
        "raster":     lambda p: raster(rec, p, g),
        "activity_per_condition": lambda p: population(rec, result, p, g),
        "electrode_scatter":      lambda p: footprint(rec, p, g),
        "unit_QC":    lambda p: unit_qc(rec, p, g),
        "metrics":    lambda p: conditions(rec, result, p, g),
        "electrode_map": lambda p: electrode_map(rec, p, g),
        "overview":      lambda p: recording_overview(rec, p, g),
        # diagnostics: all judged units, marked by verdict
        "isi":           lambda p: isi_violations(seen, p, g),
        "ttx":           lambda p: ttx_control(seen, p, g),
        "waveforms":     lambda p: waveforms(seen, p, g),
        "selection_drops": lambda p: selection_drops(seen, p, g),
    }
    written, failed, skipped = [], {}, []
    for name in want:
        if name not in jobs:
            failed[name] = "unknown figure"
            continue
        path = out / f"{rec.name}.{name}.{'pdf' if name == 'overview' else ext}"
        try:
            r = jobs[name](path)
        except Exception as e:                                   # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"
            continue
        (written if r else skipped).append(str(r) if r else name)
    res = {"written": written, "dir": str(out)}
    if skipped:
        res["skipped"] = skipped        # nothing to draw, e.g. no stage-6 result
    if failed:
        res["failed"] = failed
    return res


# --- legacy MATLAB results ---
def _matlab_str(v):
    """Text of a MATLAB char array or str; "" for an opaque MCOS `string`."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    a = np.atleast_1d(np.asarray(v, dtype=object)
                      if not hasattr(v, "dtype") else v)
    if a.dtype.kind in "US":
        return "".join(str(x) for x in a.ravel())
    return ""                      # opaque MCOS: caller falls back on the index


def _legacy_population(pop):
    """MATLAB population metrics, with fanoFactor/lvr renamed *Legacy (different definitions)."""
    out = {k: float(getattr(pop, k)) for k in dir(pop)
           if not k.startswith("_") and np.isscalar(getattr(pop, k))}
    alias = {"fanoFactor": "fanoFactorLegacy", "lvr": "lvrLegacy"}
    out = {alias.get(k, k): v for k, v in out.items()}
    if "firingRate" not in out and "firingRateISI" in out:
        out["firingRate"] = out["firingRateISI"]      # the rate we report
    return out


def from_analyzed_mat(path, name=None):
    """Load a MATLAB `_Analyzed.mat` into the result shape `population`/`conditions` draw."""
    import scipy.io as sio

    d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    per = []
    for i, c in enumerate(np.atleast_1d(d["PerCondition"])):
        nbd = c.NetworkBurstDetected
        ib = np.atleast_1d(getattr(nbd, "IndivBurst", []))
        bursts = [{"peak": float(np.ravel(b.peak)[0])}
                  for b in ib if hasattr(b, "peak")]
        pop = getattr(getattr(c, "SpikeFeature", None), "Population", None)
        win = np.ravel(np.asarray(c.window, float))
        per.append({
            # `condition` is opaque MCOS; `file` holds the same name as chars
            "condition": (_matlab_str(c.condition) or _matlab_str(getattr(c, "file", None))
                          or f"condition {i + 1}"),
            "window_s": (float(win[0]), float(win[-1])),
            "spike": {"population": _legacy_population(pop)} if pop else {},
            "network": {"detected": {
                "nBurst": float(np.ravel(getattr(nbd, "nBurst", np.nan))[0]),
                "bin": float(np.ravel(getattr(nbd, "bin", np.nan))[0]),
                "IndivBurst": bursts,
                # MATLAB's own binned trace
                "time": np.ravel(np.asarray(getattr(nbd, "time", []), float)),
                "spikesBin": np.ravel(np.asarray(getattr(nbd, "spikesBin", []), float)),
            }},
        })
    return {"per_condition": per, "name": name or Path(path).stem}


def population_from_trace(result, path, cfg=None, title=None):
    """`population`, drawn from a stored (MATLAB) trace instead of re-binning spikes."""
    _, plt = _mpl(cfg)
    if plt is None or not result:
        return None
    per = result.get("per_condition") or []
    if not per:
        return None

    fig, axes = plt.subplots(len(per), 1, figsize=(11, 1.9 * len(per)),
                             sharey=True, squeeze=False)    # as `population`
    for ax, entry in zip(axes[:, 0], per):
        det = (entry.get("network") or {}).get("detected") or {}
        t = np.asarray(det.get("time", []), float)
        y = np.asarray(det.get("spikesBin", []), float)
        if t.size and y.size:
            n = min(t.size, y.size)
            ax.fill_between(t[:n], y[:n], step="post", color="#8A5A14", alpha=.30, lw=0)
            ax.plot(t[:n], y[:n], drawstyle="steps-post", color="#8A5A14", lw=.7)
        for b in det.get("IndivBurst", []):
            pk = b.get("peak")
            if pk is not None and np.isfinite(pk):
                ax.axvline(float(pk), color="#8C3B3B", lw=.8, alpha=.85)
        n_b = det.get("nBurst", np.nan)
        n_ev = len(det.get("IndivBurst", []))
        label = (f"{n_ev} events" if np.isfinite(n_b) and n_b
                 else f"{n_ev} events, not network bursting")
        _style(ax, f"{entry['condition']}  --  {label}   "
                   f"[bin {det.get('bin', float('nan')):.2f} s]",
               None, "normalised rate")   # same quantity as `population`
        if t.size:
            ax.set_xlim(float(t.min()), float(t.max()))
    axes[-1, 0].set_xlabel("time (s)", fontsize=9)
    fig.suptitle(title or f"{result.get('name','')} -- population activity",
                 fontsize=10, x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=(cfg or {}).get("dpi", 150))
    plt.close(fig)
    return path
