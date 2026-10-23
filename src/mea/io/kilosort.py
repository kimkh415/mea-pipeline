"""Stage 1, sorter arm: Kilosort + bombcell output folder -> Recording.

Unit-type selection is not done here; stage 5 applies it from config.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .recording import Recording

FILES = ("spike_times.npy", "spike_clusters.npy", "metadata.csv",
         "recording_data.csv")


def _column(df, *names):
    """First of `names` present; KeyError if none."""
    for n in names:
        if n in df.columns:
            return df[n]
    raise KeyError(f"none of {names} in {sorted(df.columns)}")


def read_conditions(path):
    """`recording_data.csv` -> (conditions frame in samples and seconds, fs)."""
    df = pd.read_csv(path)
    fs = np.asarray(_column(df, "sampling_freq", "sampling_frequency"), float)
    if not np.allclose(fs, fs[0]):
        raise ValueError(
            f"{path} has more than one sampling rate ({sorted(set(fs))}). Spike "
            "times are sample indices over the concatenated recording, so a "
            "single rate is needed to place them; this file cannot be shaped.")
    s0 = np.asarray(_column(df, "sample_start"), np.int64)
    s1 = np.asarray(_column(df, "sample_end"), np.int64)
    names = [str(x) for x in _column(df, "recording_name")]
    cond = pd.DataFrame({"name": names,
                         "t0_s": s0 / fs[0], "t1_s": s1 / fs[0],
                         "sample_start": s0, "sample_end": s1})
    cond["is_ttx"] = cond["name"].str.contains("ttx", case=False, regex=False)
    return cond, float(fs[0])


#: extra metadata.csv label columns to carry through (none currently)
OPTIONAL_LABELS = ()


def read_clusters(path):
    """`metadata.csv` -> one row per cluster: id, bombcell type, peak channel."""
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "cluster": np.asarray(_column(df, "new_cluster_id", "cluster_id"), int),
        "unitType_bombcell": np.asarray(_column(df, "unitType", "unit_type"), int),
        "channel": np.asarray(_column(df, "peak_channel"), int),
        "x": np.asarray(_column(df, "peak_chan_x"), float),
        "y": np.asarray(_column(df, "peak_chan_y"), float),
    })
    for col in OPTIONAL_LABELS:
        if col in df.columns:
            out[col] = np.asarray(df[col], int)
    # Kilosort good/mua; optional (older sorts lack it)
    if "KSLabel" in df.columns:
        out["ks_label"] = df["KSLabel"].astype(str).str.lower().to_numpy()
    return out


def import_kilosort(folder, name=None, source="Kilosort"):
    """Build a Recording from a sorter output folder; spikes outside every condition window are dropped."""
    folder = Path(folder)
    missing = [f for f in FILES if not (folder / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{folder} is not a shaped sorter output: missing {', '.join(missing)}. "
            "Run the Kilosort and bombcell compilation steps first.")

    cond, fs = read_conditions(folder / "recording_data.csv")
    clusters = read_clusters(folder / "metadata.csv")
    t_step_ms = 1000.0 / fs

    samples = np.load(folder / "spike_times.npy").ravel().astype(np.int64)
    labels = np.load(folder / "spike_clusters.npy").ravel().astype(np.int64)
    if samples.size != labels.size:
        raise ValueError(
            f"spike_times has {samples.size} entries but spike_clusters has "
            f"{labels.size}; they index the same spikes and must agree.")

    # Spikes past the csv end: csv is from a different concatenation (a longer csv can't be detected)
    end = int(cond["sample_end"].to_numpy().max())
    if samples.size and int(samples.max()) > end:
        raise ValueError(
            f"spike_times reaches sample {int(samples.max()):,} but "
            f"recording_data.csv ends at {end:,}. The two files come from "
            "different runs, or from sorter outputs built with different "
            "sample conventions. Spikes would be silently dropped and "
            "conditions mislabelled, so this refuses rather than guesses.")

    # both ends exclusive, as in MATLAB
    lo = cond["sample_start"].to_numpy()[:, None]
    hi = cond["sample_end"].to_numpy()[:, None]
    inside = ((samples[None, :] > lo) & (samples[None, :] < hi)).any(axis=0)

    lab = labels[inside]
    t_ms = samples[inside].astype(float) * t_step_ms

    ids = clusters["cluster"].to_numpy()
    unknown = sorted(set(lab.tolist()) - set(ids.tolist()))
    if unknown:
        raise ValueError(
            f"spike_clusters names {len(unknown)} clusters absent from "
            f"metadata.csv (first: {unknown[:5]}); the two files describe "
            "different sorter runs.")

    # keep spikeless clusters so positions match stored per-cluster decisions
    by_cluster = {int(c): np.sort(t_ms[lab == c]) for c in ids}
    trains = [by_cluster[int(c)] for c in ids]
    counts = np.array([t.size for t in trains], np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    ordered = np.concatenate(trains) if trains else np.array([], float)

    name = name or folder.name
    units = pd.DataFrame({
        "unit_id": [f"{name}:{source}:{int(c)}" for c in ids],
        "position": np.arange(ids.size),
        "cluster": ids,
        "channel": clusters["channel"].to_numpy(),
        "x": clusters["x"].to_numpy(), "y": clusters["y"].to_numpy(),
        "unitType_bombcell": clusters["unitType_bombcell"].to_numpy(),
        "source": source, "kept": True,
    })
    for col in OPTIONAL_LABELS:
        if col in clusters.columns:
            units[col] = clusters[col].to_numpy()
    if "ks_label" in clusters.columns:
        units["ks_label"] = clusters["ks_label"].to_numpy()

    # Optional waveforms.npy: legacy ragged object array or plain 2-D float
    waveforms = None
    wf = folder / "waveforms.npy"
    if wf.exists():
        try:
            w = np.load(wf, allow_pickle=True)
            if w.dtype == object:
                n = max(len(np.atleast_1d(x)) for x in w)
                w = np.array([np.pad(np.atleast_1d(x).astype(float),
                                     (0, n - len(np.atleast_1d(x))))
                              for x in w], dtype=float)
            waveforms = w if len(w) == len(units) else None
        except Exception:                                    # noqa: BLE001
            waveforms = None

    return Recording(
        name=name, t_ms=ordered, offsets=offsets, units=units,
        waveforms=waveforms,
        conditions=cond[["name", "t0_s", "t1_s", "is_ttx"]],
        meta={"imported_from": str(folder), "produced_by": "shape",
              "capabilities": sorted({"spikes", "conditions", "per_condition"}),
              "source": source, "t_step_ms": t_step_ms,
              "sampling_hz": fs, "n_cluster_original": int(ids.size),
              "n_spikes_outside_conditions": int((~inside).sum())},
    )
