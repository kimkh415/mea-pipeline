"""Manual curation saved as JSON, keyed by stable unit_id so it survives upstream re-runs."""
from __future__ import annotations
import json, time
from pathlib import Path


class Curation:
    def __init__(self, recording, discarded=None, notes=None, reviewer=None):
        self.recording = recording
        self.discarded = list(discarded or [])
        self.notes = dict(notes or {})
        self.reviewer = reviewer
        self.when = time.strftime("%Y-%m-%dT%H:%M:%S")

    # --- disk ---
    def save(self, path):
        Path(path).write_text(json.dumps({
            "recording": self.recording, "reviewer": self.reviewer, "when": self.when,
            "discarded": self.discarded, "notes": self.notes}, indent=2))
        return path

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        c = cls(d["recording"], d.get("discarded"), d.get("notes"), d.get("reviewer"))
        c.when = d.get("when", c.when)
        return c

    # --- replay ---
    def apply(self, rec, stage="qc_manual"):
        """Mark curated units discarded -> (applied, orphaned ids)."""
        present = set(rec.units["unit_id"].tolist())
        applied = [u for u in self.discarded if u in present]
        orphaned = [u for u in self.discarded if u not in present]
        pos = rec.units.index[rec.units["unit_id"].isin(applied)].to_numpy()
        rec.mark(stage, "manual_discard", pos)
        return applied, orphaned

    # --- import from MATLAB ---
    @classmethod
    def from_good_cluster(cls, rec, good_cluster, reviewer="imported from _QCmanual.mat"):
        """Curation from a MATLAB good_cluster mask (one entry per unit position)."""
        import numpy as np
        g = np.ravel(np.asarray(good_cluster)).astype(bool)
        ids = rec.units["unit_id"].to_numpy()
        if g.size != ids.size:
            raise ValueError(f"good_cluster has {g.size} entries, recording has {ids.size} units")
        return cls(rec.name, discarded=[str(u) for u in ids[~g]], reviewer=reviewer)
