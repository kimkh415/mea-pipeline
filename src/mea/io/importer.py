"""`mea import`: build a Recording from a legacy MATLAB stage output.

meta["capabilities"] lists what the file carries, so a stage can refuse instead of guessing.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from .matlab import load_mat, unpack_spike_timestamp, unpack_rec_conditions
from .recording import Recording


def _units_frame(s, name, source):
    n = s["n_units_struct"]
    return pd.DataFrame({
        "unit_id": [f"{name}:{source}:{int(c)}" for c in s["cluster"]],
        "position": np.arange(n),
        "cluster": s["cluster"].astype(int),
        "x": s["pos_x"], "y": s["pos_y"],
        "source": source, "kept": True,
    })


def import_mat(path, name=None, source="3Brain"):
    path = Path(path)
    from ..discovery import STAGE_OF, _split_suffix

    base, suffix, variant = _split_suffix(path.stem)
    if base is None:
        raise ValueError(
            f"{path.name} does not end in a recognised stage suffix "
            f"({', '.join(sorted(STAGE_OF))})")
    produced_by = STAGE_OF[suffix]
    if produced_by in ("quantify", "compile"):
        raise ValueError(
            f"{path.name} is a RESULTS file ({produced_by} output) and cannot be "
            "imported: it has no rec_conditions, so the spike trains cannot be "
            "placed on a timeline.  Import the file the stage before it wrote "
            "instead -- for quantify that is *_curated_QCmanual.mat.")
    name = name or base
    d = load_mat(path)

    s = unpack_spike_timestamp(d["spikeTimestamp"])
    rc = unpack_rec_conditions(d["rec_conditions"], d.get("Indiv_Cond"))
    cond = pd.DataFrame(rc, columns=["name", "t0_s", "t1_s"])
    cond["is_ttx"] = cond["name"].str.contains("ttx", case=False, regex=False)

    caps = {"spikes", "conditions"}
    extra = {}
    if "spike_plot" in d:                       # pre-echo-removal, all electrodes
        caps.add("pre_echo_spikes")
        extra["spike_plot"] = unpack_spike_timestamp(d["spike_plot"])
    if "Indiv_Cond" in d:
        caps.add("per_condition")
    for k in ("good_cluster", "unitLabel", "discarded_cluster", "n_cluster0",
              "cond_ttx", "n_discarded", "n_clusterOriginal"):
        if k in d:
            extra[k] = d[k]
    if "discarded_cluster" in d:
        caps.add("qc.general")
    if "good_cluster" in d:
        caps.add("qc.manual")

    rec = Recording(
        name=name, t_ms=s["t_ms"], offsets=s["offsets"],
        units=_units_frame(s, name, source), conditions=cond,
        meta={"imported_from": str(path), "produced_by": produced_by,
              "variant": variant,
              "capabilities": sorted(caps), "source": source,
              "t_step_ms": float(np.nanmedian(s["t_step_ms"])),
              "n_cluster_original": int(np.ravel(d["n_clusterOriginal"])[0])
                                    if "n_clusterOriginal" in d else s["n_units_struct"]},
    )
    # unitLabel: human reviewer's class per cluster (col 0 cluster, col 1 class); legacy only
    ul = extra.get("unitLabel")
    if ul is not None:
        ul = np.atleast_2d(np.asarray(ul))
        if ul.ndim == 2 and ul.shape[1] >= 2:
            by_cluster = dict(zip(ul[:, 0].astype(int), ul[:, 1].astype(int)))
            rec.units["unitType_reviewer"] = [by_cluster.get(int(c), 0)
                                       for c in rec.units["cluster"]]

    rec.legacy = extra          # raw MATLAB variables, for parity testing
    return rec


REQUIRES = {
    "qc_general":   {"spikes", "conditions"},
    "qc_crosstalk": {"pre_echo_spikes", "qc.general"},
    "qc_manual":    {"spikes", "qc.general"},
    "quantify":     {"spikes", "conditions"},
    "compile":      {"spikes", "conditions"},
}


def can_run(rec, stage):
    have = set(rec.meta.get("capabilities", []))
    missing = REQUIRES[stage] - have
    return (not missing), missing
