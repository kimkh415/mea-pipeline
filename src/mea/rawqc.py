"""Organoid-level QC on raw .brw files: count electrodes TTX silences, no sorting needed.

Counts are electrodes, not sorted units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import brw


@dataclass
class OrganoidQC:
    passed: bool
    n_active: int
    n_ttx_sensitive: int
    sensitive_fraction: float
    noise_floor: float
    threshold: int
    active_channels: np.ndarray
    sensitive_channels: np.ndarray
    baseline: str
    ttx: str

    def summary(self):
        return (f"{'PASS' if self.passed else 'FAIL'}: {self.n_ttx_sensitive} "
                f"TTX-sensitive electrodes (threshold {self.threshold}), "
                f"{self.n_active} active at baseline")


def ttx_gate(baseline_path, ttx_path, cfg=None, n_chunks=None):
    """Pass/fail a chip on how many active electrodes TTX silences.

    `n_chunks=None` scans the whole file; sampling is too noisy for the pass/fail call.
    """
    g = ((cfg or {}).get("raw_qc") or {})
    thr = int(g.get("min_ttx_sensitive", 30))
    floor_mult = float(g.get("active_floor_multiple", 2.0))
    drop = float(g.get("drop_fraction", 0.5))

    def occupancy(path):
        chunks = None if n_chunks is None else brw.sample_chunks(path, n_chunks)
        return brw.channel_activity(path, chunks=chunks)["occupancy"]

    ob, ot = occupancy(baseline_path), occupancy(ttx_path)
    n = min(ob.size, ot.size)
    ob, ot = ob[:n], ot[:n]

    floor = float(np.median(ob))                     # per-file noise floor
    active = ob > floor * floor_mult
    sensitive = active & (ot < drop * ob)
    n_act, n_sen = int(active.sum()), int(sensitive.sum())
    return OrganoidQC(
        passed=n_sen > thr, n_active=n_act, n_ttx_sensitive=n_sen,
        sensitive_fraction=n_sen / n_act if n_act else 0.0,
        noise_floor=floor, threshold=thr,
        active_channels=np.flatnonzero(active),
        sensitive_channels=np.flatnonzero(sensitive),
        baseline=str(baseline_path), ttx=str(ttx_path),
    )


def footprint(qc, grid=(64, 64)):
    """(x, y) of the TTX-sensitive electrodes, for plotting the organoid outline."""
    ncol = grid[1]
    ch = qc.sensitive_channels
    return ch // ncol, ch % ncol
