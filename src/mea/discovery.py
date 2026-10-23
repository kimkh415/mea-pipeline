"""Find recordings on disk and the stage each can enter at.

Stage outputs are classified by the MATLAB filename suffixes in STAGE_OF.
"""
from __future__ import annotations
import re
import warnings
from pathlib import Path
from dataclasses import dataclass, field

# suffix -> the stage that file is the OUTPUT of
STAGE_OF = {
    "_raw":       "shape",
    "_cleaned":   "qc_general",
    "_QC":        "qc_crosstalk",
    "_QCmanual":  "qc_manual",
    "_curated_QCmanual": "integrate",
    "_Analyzed":  "quantify",
    "_Compiled":  "compile",
}
ORDER = ["shape", "qc_general", "qc_crosstalk", "qc_manual",
         "integrate", "quantify", "compile"]

# raw inputs: .brw voltage (sorted by Kilosort); .bxr is unsupported
RAW_INPUT = {".brw": "shape_kilosort"}

UNSUPPORTED_RAW = {
    ".bxr": "3Brain's own spike detection. This pipeline sorts the raw .brw with "
            "Kilosort instead, so a .bxr is not an entry point.",
}

# a folder with all of these is a finished sorter run (stage-1 entry point)
SORTER_OUTPUT = ("spike_times.npy", "spike_clusters.npy",
                 "metadata.csv", "recording_data.csv")


@dataclass
class Found:
    name: str
    directory: Path
    outputs: dict = field(default_factory=dict)   # stage -> path
    raw: dict = field(default_factory=dict)       # kind -> [paths]
    variants: set = field(default_factory=set)    # e.g. {"curated"}
    sorter_output: Path | None = None             # a finished Kilosort run

    @property
    def entry_stage(self):
        """Stage after the furthest completed one (not the first gap)."""
        f = self.furthest
        if f is None:
            return ORDER[0]
        i = ORDER.index(f)
        return ORDER[i + 1] if i + 1 < len(ORDER) else None

    @property
    def furthest(self):
        have = [s for s in ORDER if s in self.outputs]
        if have:
            return have[-1]
        return "shape" if self.sorter_output is not None else None

    @property
    def has_state(self):
        """True if any stage output or sorter run exists."""
        return bool(self.outputs) or self.sorter_output is not None


VARIANTS = ("curated", "integrated")


def _split_suffix(stem):
    """stem -> (base, suffix, variant); longest suffix wins, _curated/_integrated stripped as variant."""
    for suf in sorted(STAGE_OF, key=len, reverse=True):
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            variant = None
            for v in VARIANTS:
                if base.endswith("_" + v):
                    base, variant = base[: -(len(v) + 1)], v
            return base, suf, variant
    return None, None, None


def scan(paths, cfg=None):
    """Group files or directories under `paths` into recordings."""
    found: dict[str, Found] = {}
    raw_files: list[Path] = []
    files = []
    for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
        p = Path(p)
        if p.is_dir():
            files += sorted(x for x in p.rglob("*") if x.is_file())
        elif p.is_file():
            files.append(p)

    # Find sorter output folders
    sorter_dirs = set()
    for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
        p = Path(p)
        if not p.is_dir():
            continue
        for d in [p, *(x for x in p.rglob("*") if x.is_dir())]:
            if all((d / f).exists() for f in SORTER_OUTPUT):
                sorter_dirs.add(d)
    for d in sorted(sorter_dirs):
        rec = found.setdefault(d.name, Found(d.name, d.parent))
        rec.sorter_output = d

    unsupported = sorted({f.suffix.lower() for f in files
                          if f.suffix.lower() in UNSUPPORTED_RAW})
    if unsupported:
        # warn rather than skip silently
        warnings.warn("ignoring " + "; ".join(
            f"{n} {ext} file(s) -- {UNSUPPORTED_RAW[ext]}"
            for ext in unsupported
            for n in [sum(f.suffix.lower() == ext for f in files)]))

    for f in files:
        if f.suffix.lower() in RAW_INPUT:
            raw_files.append(f)
            continue
        if f.suffix.lower() != ".mat":
            continue
        base, suf, variant = _split_suffix(f.stem)
        if base is None:
            continue
        rec = found.setdefault(base, Found(base, f.parent))
        rec.outputs[STAGE_OF[suf]] = f
        if variant:
            rec.variants.add(variant)

    # Group raw files into recordings (a folder may hold several)
    group_by = ((cfg or {}).get("dataset", {}).get("discovery", {})
                .get("group_by", "basename"))
    for d in {f.parent for f in raw_files}:
        here = sorted(f for f in raw_files if f.parent == d)
        groups = ({_common_stem([f.stem for f in here]): here} if group_by == "directory"
                  else group_by_stem(here))
        for base, members in groups.items():
            rec = found.setdefault(base, Found(base, d))
            for f in members:
                rec.raw.setdefault(RAW_INPUT[f.suffix.lower()], []).append(f)
    return dict(sorted(found.items()))


def _common_stem(stems):
    """Longest shared prefix, cut back to the last `_` field boundary."""
    import os
    if len(stems) == 1:
        return stems[0]
    pre = os.path.commonprefix(list(stems))
    return pre[:pre.rfind("_")].rstrip("_") if "_" in pre else ""


def group_by_stem(files):
    """Group raw files by shared field-aligned stem; with none, split on the first field and recurse."""
    files = list(files)
    if not files:
        return {}
    base = _common_stem([f.stem for f in files])
    if base:
        return {base: files}
    parts: dict[str, list] = {}
    for f in files:
        parts.setdefault(f.stem.split("_", 1)[0], []).append(f)
    if len(parts) == 1:                     # no field boundary to split on
        return {files[0].stem: files}
    out = {}
    for group in parts.values():
        out.update(group_by_stem(group))
    return out


def group_raw_by_recording(rec, cfg):
    """Per-condition raw files of a recording, in dataset.discovery.condition_order."""
    from .conditions import order_condition_files

    out = {}
    for kind, paths in rec.raw.items():
        out[kind] = order_condition_files(paths, cfg)[0] if paths else []
    return out
