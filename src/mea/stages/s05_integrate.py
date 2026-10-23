"""Stage 5 -- unit selection.  *_QCmanual.mat -> *_curated_QCmanual.mat

Selects units by unit-type label (bombcell: 0 noise / 1 good somatic / 2 MUA somatic /
3 good non-somatic / 4 MUA non-somatic).  Merge helpers are kept for archived 3Brain data.
"""
from __future__ import annotations
import numpy as np

from ..core.windows import counts_in_windows


def select_by_label(labels, keep_labels):
    """Mask of units whose type label is in `keep_labels`."""
    lab = np.asarray(labels).ravel()
    return np.isin(lab, np.asarray(list(keep_labels)))


def spatial_overlap(pos_a, pos_b, radius):
    """Pairs of units from A and B sitting within `radius` electrodes."""
    a = np.asarray(pos_a, float)
    b = np.asarray(pos_b, float)
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 2), int)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    i, j = np.nonzero(d <= radius)
    return np.column_stack([i, j])


def train_similarity(t_a, t_b, jitter_ms):
    """Fraction of A's spikes that have a B spike within `jitter_ms`."""
    a = np.asarray(t_a, float)
    b = np.sort(np.asarray(t_b, float))
    if a.size == 0 or b.size == 0:
        return 0.0
    n = counts_in_windows(b, a - jitter_ms, a + jitter_ms)
    return float((n > 0).sum()) / a.size


def run(rec, cfg, ctx=None):
    """Apply the label selection to a Recording, marking (not deleting) the rest."""
    g = cfg.get("integrate", {})
    # Pick the label column; unitType_reviewer (human) only if asked for explicitly
    wanted = g.get("label_column")
    order = ([wanted] if wanted else
             ["unitType_bombcell", "unit_label", "unit_type"])
    labels, used = None, None
    for key in order:
        if key and key in rec.units.columns:
            labels, used = rec.units[key].to_numpy(), key
            break
    if labels is None and wanted:
        raise ValueError(
            f"integrate.label_column is {wanted!r} but the recording has no "
            f"such column. Available: {sorted(rec.units.columns)}")
    if labels is None:
        return {"n_selected": int(rec.kept_mask().sum()), "n_discarded": 0,
                "note": "no unit-type column; nothing to select on"}

    # Keep list by unit source: label 2 is MUA (kept) in bombcell, "weak" in 3Brain
    source = rec.units["source"].astype(str).str.lower() if "source" in rec.units else None
    keep = np.zeros(rec.n_units, bool)
    for name, opt in (("3brain", g.get("unit_select_3brain", [1])),
                      ("kilosort", g.get("unit_select_kilosort", [1, 2, 3, 4]))):
        sel = select_by_label(labels, opt)
        if source is None:
            keep |= sel
        else:
            keep |= sel & (source.to_numpy() == name)
    if source is None and not keep.any():
        keep = select_by_label(labels, g.get("unit_select_3brain", [1]))

    # Union: also keep Kilosort `good` clusters; no-op without ks_label
    rescued = np.zeros(rec.n_units, bool)
    if g.get("union_with_kilosort_good", True) and "ks_label" in rec.units:
        ks = rec.units["ks_label"].astype(str).str.lower().to_numpy()
        rescued = (ks == "good") & ~keep
        keep = keep | rescued

    bad = np.flatnonzero(~keep)
    rec.mark("integrate", "unit_type", bad, value=labels[bad])
    return {"n_selected": int(keep.sum()), "n_discarded": int(bad.size),
            "n_rescued_by_kilosort": int(rescued.sum()),
            "discarded_idx": bad}
