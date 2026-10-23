"""Stage registry: what each stage needs, produces, and whether it is implemented."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Stage:
    key: str
    title: str
    requires: set          # capability names -- see io/importer.REQUIRES
    produces: str          # capability granted on success
    run: Optional[Callable] = None
    note: str = ""
    state: bool = False    # output .mat carries spike trains + conditions (valid entry point)

    @property
    def implemented(self):
        return self.run is not None


def _s02(rec, cfg, ctx):
    from .stages import s02_qc_general as m
    return m.run(rec, cfg, ctx["per_condition_counts"], ctx["per_condition_length_s"])


def _s03(rec, cfg, ctx):
    """Crosstalk screening, then the surrogate test on pairs that pass screening."""
    from .stages import s03_qc_crosstalk as m
    from .rng import Streams
    screen = m.screen_all(rec, cfg, ctx)
    if not cfg.get("qc_crosstalk", {}).get("run_statistics", True):
        return screen
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    stats = m.run_statistics(rec, cfg, ctx, dict(screen, red_flag=screen["red_flag"].copy()),
                             Streams(seed, label=rec.name))
    return {**screen, "red_flag": stats["red_flag"],
            "discard_mask": stats["discard_mask"],
            "n_tested": stats["n_tested"], "n_significant": stats["n_significant"]}


def _s04(rec, cfg, ctx):
    from .stages import s04_qc_manual as m
    return m.run(rec, cfg, ctx.get("crosstalk_scores"), ctx.get("curation"))


def _s05(rec, cfg, ctx):
    from .stages import s05_integrate as m
    return m.run(rec, cfg, ctx)


def _s07(rec, cfg, ctx):
    """Placeholder: compile runs across recordings via `mea compile`, not `mea run`."""
    raise RuntimeError(
        "compile runs across recordings, not on one -- use `mea compile "
        "--input RESULTS_DIR --out DIR`")


def _s06(rec, cfg, ctx):
    from .stages import s06_quantify as m
    from .rng import Streams
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    return m.run_recording(rec, cfg, Streams(seed, label=rec.name))


STAGES = [
    Stage("shape", "Spike detection & shaping", set(), "spikes",
          note="enter from a Kilosort output folder; 3Brain's own detector is out of scope",
          state=True),
    Stage("qc_general", "QC - general", {"spikes", "conditions"}, "qc.general", _s02,
          state=True),
    Stage("qc_crosstalk", "QC - crosstalk", {"pre_echo_spikes", "qc.general"},
          "qc.crosstalk", _s03, note="coincidence screen, then a surrogate test on flagged pairs"),
    Stage("qc_manual", "QC - final / curation", {"spikes", "qc.general"}, "qc.manual", _s04,
          state=True),
    Stage("integrate", "Sorter integration", {"spikes"}, "integrated", _s05,
          note="unit-type selection; the merge runs only when both sorters were used",
          state=True),
    Stage("quantify", "Quantification", {"spikes", "conditions"}, "analyzed", _s06,
          note="levels A, B and C"),
    Stage("compile", "Compilation", {"analyzed"}, "compiled", _s07,
          note="joins every recording's metrics into one table; run it over the "
               "results root, not one recording"),
]
BY_KEY = {s.key: s for s in STAGES}
ORDER = [s.key for s in STAGES]
STATE_STAGES = [s.key for s in STAGES if s.state]


def import_entry(src):
    """Import a stage-1 input: Kilosort folder or legacy .mat (shared by CLI and Snakemake)."""
    from pathlib import Path as _P
    from .io.importer import import_mat
    from .io.kilosort import import_kilosort
    return import_kilosort(src) if _P(src).is_dir() else import_mat(src)


def import_source(rec, before=None):
    """Return (stage, path) of the latest output carrying spike trains + conditions.

    `before`: only consider stages strictly upstream of this one (for re-running a stage).
    """
    keys = ORDER[:ORDER.index(before)] if before else ORDER
    for k in reversed(keys):
        if k in STATE_STAGES and k in rec.outputs:
            return k, rec.outputs[k]
    # fall back to a sorter output as stage-1 state
    if getattr(rec, "sorter_output", None) is not None and "shape" in keys:
        return "shape", rec.sorter_output
    return None, None
