"""Per-stage provenance: what ran, with what config, on what inputs."""
from __future__ import annotations
import json, hashlib, platform, time
from pathlib import Path
import numpy as np

from . import __version__

STAGE_CONFIG = {
    "shape":        ("experiment",),
    "qc_general":   ("experiment", "qc_general", "burst_detection"),
    "qc_crosstalk": ("experiment", "qc_crosstalk", "burst_detection"),
    "qc_manual":    ("experiment", "qc_manual", "burst_detection"),
    "integrate":    ("experiment",),
    "quantify":     ("experiment", "spike_features", "burst_detection",
                     "network_burst", "clustering", "compat"),
    "compile":      ("experiment", "compile", "compat"),
}


def stage_hash(cfg, stage, input_hashes=()):
    from .config import subtree_hash
    h = subtree_hash(cfg, *STAGE_CONFIG[stage])
    blob = (h + "".join(sorted(input_hashes))).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def record(rec, stage, cfg, duration_s, extra=None):
    p = rec.meta.setdefault("provenance", {})
    p[stage] = {
        "config_hash": stage_hash(cfg, stage),
        "seed": cfg["experiment"]["seed"],
        "duration_s": round(float(duration_s), 3),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mea": __version__,
        "numpy": np.__version__, "python": platform.python_version(),
        **(extra or {}),
    }
    return rec
