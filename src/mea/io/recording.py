"""Recording: the store every stage reads and writes.

Spikes are one array grouped by unit (offsets mark boundaries); QC sets kept=False and never renumbers units.
"""
from __future__ import annotations
import numpy as np, pandas as pd, json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Recording:
    name: str
    t_ms: np.ndarray                    # all spikes, sorted within unit
    offsets: np.ndarray                 # n_units + 1
    units: pd.DataFrame                 # unit_id, channel, x, y, source, kept
    conditions: pd.DataFrame            # name, t0_s, t1_s, is_ttx
    qc: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["unit_id", "stage", "flag", "value", "kept"]))
    meta: dict = field(default_factory=dict)
    waveforms: np.ndarray | None = None

    # --- views ---
    @property
    def n_units(self) -> int:
        return len(self.units)

    def train(self, i: int) -> np.ndarray:
        """Spike times (ms) of the unit at position i, as a view."""
        return self.t_ms[self.offsets[i]:self.offsets[i + 1]]

    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def unit_of_spike(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_units, dtype=np.int32), self.counts())

    def kept_mask(self) -> np.ndarray:
        return self.units["kept"].to_numpy(bool)

    def duration_s(self) -> float:
        return float(self.conditions["t1_s"].max() - self.conditions["t0_s"].min())

    def per_condition(self):
        """(n_conditions, n_units) spike counts and window lengths (s)."""
        t0 = self.conditions["t0_s"].to_numpy(float) * 1000.0
        t1 = self.conditions["t1_s"].to_numpy(float) * 1000.0
        counts = np.empty((t0.size, self.n_units), float)
        for i in range(self.n_units):
            tr = self.train(i)
            counts[:, i] = (np.searchsorted(tr, t1, "right")
                            - np.searchsorted(tr, t0, "left"))
        lengths = np.repeat(((t1 - t0) / 1000.0)[:, None], self.n_units, axis=1)
        return counts, lengths

    # --- artifacts ---
    def artifact_dir(self):
        d = Path(self.meta.get("_dir", ".")) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def persist(self, stage: str, result: dict):
        """Save a stage's arrays/frames to artifacts/ and scalars to meta; grant its capabilities."""
        if not result:
            return
        d = self.artifact_dir()
        scalars = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                np.save(d / f"{stage}.{k}.npy", v)
            elif isinstance(v, pd.DataFrame):
                v.to_parquet(d / f"{stage}.{k}.parquet", index=False)
            elif isinstance(v, (int, float, bool, str)) or v is None:
                scalars[k] = v
        if scalars:
            self.meta.setdefault("results", {})[stage] = scalars
        # grant both the stage name and its capability (e.g. "qc_general" and "qc.general")
        from ..pipeline import BY_KEY
        granted = {stage}
        st = BY_KEY.get(stage)
        if st is not None and st.produces:
            granted.add(st.produces)
        if stage == "qc_general":
            granted.add("pre_echo_spikes")      # results carry the pre-echo trains
        self.meta.setdefault("capabilities", [])
        self.meta["capabilities"] = sorted(set(self.meta["capabilities"]) | granted)

    def load_artifacts(self):
        """Everything `persist` wrote, keyed `<stage>.<name>`."""
        d = Path(self.meta.get("_dir", ".")) / "artifacts"
        out = {}
        if not d.exists():
            return out
        for f in sorted(d.iterdir()):
            key = f.name.rsplit(".", 1)[0]
            if f.suffix == ".npy":
                out[key] = np.load(f, allow_pickle=False)
            elif f.suffix == ".parquet":
                out[key] = pd.read_parquet(f)
        return out

    def context(self):
        """Inter-stage context rebuilt from disk (stages run as separate processes)."""
        counts, lengths = self.per_condition()
        ctx = {"per_condition_counts": counts, "per_condition_length_s": lengths}
        ctx.update(self.load_artifacts())
        # alias artifacts to the names stages ask for
        if "qc_crosstalk.discard_mask" in ctx:
            ctx["crosstalk_scores"] = ctx["qc_crosstalk.discard_mask"]
        if "qc_general.trains" in ctx:
            ctx["pre_echo_trains"] = ctx["qc_general.trains"]
        return ctx

    # --- edits ---
    def mark(self, stage: str, flag: str, unit_pos, value=None, drop=True):
        """Record a QC flag; `drop=False` records it without setting kept=False."""
        unit_pos = np.atleast_1d(np.asarray(unit_pos))
        if unit_pos.size == 0:
            return
        ids = self.units["unit_id"].to_numpy()[unit_pos]
        vals = np.full(unit_pos.size, np.nan) if value is None else np.atleast_1d(value)
        add = pd.DataFrame({"unit_id": ids, "stage": stage, "flag": flag,
                            "value": vals, "kept": False})
        self.qc = pd.concat([self.qc, add], ignore_index=True)
        if drop:
            self.units.loc[self.units.index[unit_pos], "kept"] = False

    def subset(self, mask) -> "Recording":
        """New Recording with only `mask` units; ids preserved."""
        mask = np.asarray(mask, bool)
        keep = np.flatnonzero(mask)
        trains = [self.train(i) for i in keep]
        sizes = np.array([t.size for t in trains], dtype=np.int64)
        off = np.zeros(keep.size + 1, dtype=np.int64); off[1:] = np.cumsum(sizes)
        return Recording(
            name=self.name,
            t_ms=np.concatenate(trains) if keep.size else np.array([]),
            offsets=off,
            units=self.units.iloc[keep].reset_index(drop=True),
            conditions=self.conditions.copy(),
            qc=self.qc.copy(),
            meta=dict(self.meta),
            waveforms=None if self.waveforms is None else self.waveforms[keep],
        )

    # --- disk ---
    def save(self, out_dir):
        d = Path(out_dir) / f"{self.name}.mea"; d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"unit_idx": self.unit_of_spike(), "t_ms": self.t_ms}
                     ).to_parquet(d / "spikes.parquet", index=False)
        self.units.to_parquet(d / "units.parquet", index=False)
        self.conditions.to_parquet(d / "conditions.parquet", index=False)
        self.qc.to_parquet(d / "qc.parquet", index=False)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        if self.waveforms is not None:
            np.save(d / "waveforms.npy", self.waveforms)
        return d

    @classmethod
    def load(cls, d):
        d = Path(d)
        sp = pd.read_parquet(d / "spikes.parquet")
        units = pd.read_parquet(d / "units.parquet")
        counts = np.bincount(sp["unit_idx"].to_numpy(), minlength=len(units))
        off = np.zeros(len(units) + 1, dtype=np.int64); off[1:] = np.cumsum(counts)
        wf = d / "waveforms.npy"
        return cls(name=d.stem, t_ms=sp["t_ms"].to_numpy(), offsets=off, units=units,
                   conditions=pd.read_parquet(d / "conditions.parquet"),
                   qc=pd.read_parquet(d / "qc.parquet"),
                   meta={**json.loads((d / "meta.json").read_text()), "_dir": str(d)},
                   waveforms=np.load(wf) if wf.exists() else None)

    def __post_init__(self):
        self.meta.setdefault("capabilities", [])
