"""Order a recording's files into conditions (dataset.discovery.condition_order).

A wrong order doesn't fail; it silently relabels the drugs.
"""
from __future__ import annotations

import datetime
import re
import os
from pathlib import Path

# .NET DateTime: ticks (100 ns since 0001-01-01) in bits 0-61, Kind in 62-63
_TICK_MASK = 0x3FFFFFFFFFFFFFFF
_EPOCH = datetime.datetime(1, 1, 1)


_FRACTION = re.compile(r"\.\d+")


def acquisition_time(path):
    """Acquisition time inside a 3Brain file, or None.

    Old layout: root attr ExperimentDateTimeUtc (.NET ticks); BrainWave 6: ExperimentInfo JSON.
    """
    try:
        import h5py
        with h5py.File(str(path), "r") as h:
            raw = h.attrs.get("ExperimentDateTimeUtc")
            if raw is not None:
                ticks = int(raw) & _TICK_MASK
                return _EPOCH + datetime.timedelta(microseconds=ticks // 10)
            return _from_experiment_info(h)
    except Exception:
        return None


def _from_experiment_info(h):
    """Time from BrainWave 6's ExperimentInfo JSON, or None."""
    import json

    if "ExperimentInfo" not in h:
        return None
    raw = h["ExperimentInfo"][()]
    blob = raw[0] if getattr(raw, "shape", None) else raw
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "replace")
    stamp = json.loads(blob).get("BasicData", {}).get("ExperimentDateTimeUtc")
    if not stamp:
        return None
    # trim 7-digit fractions to microseconds for older fromisoformat
    stamp = str(stamp).replace("Z", "+00:00")
    stamp = _FRACTION.sub(lambda m: m.group(0)[:7], stamp)
    return datetime.datetime.fromisoformat(stamp).replace(tzinfo=None)


def modified_time(path):
    """Filesystem mtime as naive UTC."""
    return datetime.datetime.fromtimestamp(os.path.getmtime(str(path)),
                                           datetime.timezone.utc).replace(tzinfo=None)


def base_order(paths, mode="file_time"):
    """Sort files by name, mtime or file_time (mtime if no timestamp) -> (paths, key per path)."""
    paths = [Path(p) for p in paths]
    if mode == "name":
        keyed = [(p.name, p) for p in paths]
        return [p for _, p in sorted(keyed)], {p: p.name for p in paths}

    if mode == "mtime":
        keys = {p: modified_time(p) for p in paths}
    else:                                    # file_time, name tiebreak
        keys = {}
        for p in paths:
            t = acquisition_time(p)
            keys[p] = t if t is not None else modified_time(p)
    return sorted(paths, key=lambda p: (keys[p], p.name)), keys


def apply_explicit(paths, order, baseline="Baseline"):
    """Reorder `paths` by 1-based indices or case-insensitive filename substrings.

    Raises unless the spec accounts for every file.
    """
    paths = [Path(p) for p in paths]
    if not order:
        raise ValueError("condition_order is 'explicit' but dataset.discovery.order is empty")

    if len(order) != len(paths):
        raise ValueError(
            f"dataset.discovery.order lists {len(order)} conditions but "
            f"{len(paths)} files were found:\n  order: {list(order)}\n  files: "
            + ", ".join(p.name for p in paths))

    if all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in order):
        idx = [int(o) for o in order]
        if sorted(idx) != list(range(1, len(paths) + 1)):
            raise ValueError(
                f"dataset.discovery.order must be a permutation of 1..{len(paths)}, "
                f"got {idx}.  It is 1-based and every position must appear once.")
        return [paths[i - 1] for i in idx]

    # Match drug names first; only the baseline label may take the one leftover file
    # (baseline files carry no drug word; other misses must stay errors)
    out = [None] * len(order)
    remaining = list(paths)
    deferred = []
    for i, want in enumerate(order):
        w = str(want).lower()
        hit = [p for p in remaining if w in p.name.lower()]
        if len(hit) > 1:
            raise ValueError(
                f"dataset.discovery.order entry {want!r} is ambiguous, it matches: "
                + ", ".join(p.name for p in hit))
        if not hit:
            deferred.append((i, want))
            continue
        out[i] = hit[0]
        remaining.remove(hit[0])

    if (len(deferred) == 1 and len(remaining) == 1
            and str(deferred[0][1]).lower() == str(baseline).lower()):
        out[deferred[0][0]] = remaining.pop()
        deferred = []
    if deferred:
        raise ValueError(
            "dataset.discovery.order "
            + ("entry " if len(deferred) == 1 else "entries ")
            + ", ".join(repr(w) for _, w in deferred)
            + " matches none of the remaining files: "
            + ", ".join(p.name for p in remaining)
            + f".  Only {baseline!r} may be left unmatched, since the untreated "
              "file is named for the recording rather than for a drug.")
    return out


def order_condition_files(paths, cfg):
    """Ordered condition files, plus a one-line account of how they were ordered."""
    disc = (cfg.get("dataset", {}) or {}).get("discovery", {}) or {}
    mode = disc.get("condition_order", "file_time")

    if mode == "explicit":
        base, _ = base_order(paths, "name")      # stable, reproducible base
        roles = ((cfg.get("dataset", {}) or {}).get("conditions", {}) or {}).get("roles", {})
        return apply_explicit(base, disc.get("order"),
                              baseline=roles.get("baseline", "Baseline")), \
            "explicit (dataset.discovery.order)"

    if mode not in ("file_time", "mtime", "name"):
        raise ValueError(
            f"dataset.discovery.condition_order={mode!r} is not one of "
            "file_time | mtime | name | explicit")

    ordered, _ = base_order(paths, mode)
    return ordered, mode


def describe(paths, cfg):
    """Rows for `mea conditions`: index, name, acquired and modified times."""
    disc = (cfg.get("dataset", {}) or {}).get("discovery", {}) or {}
    mode = disc.get("condition_order", "file_time")
    base, _ = base_order(paths, "name" if mode == "explicit" else mode)
    rows = []
    for i, p in enumerate(base, start=1):
        acq = acquisition_time(p)
        rows.append({"index": i, "name": p.name, "path": p,
                     "acquired": acq.isoformat(sep=" ", timespec="seconds") if acq else "-",
                     "modified": modified_time(p).isoformat(sep=" ", timespec="seconds")})
    return rows, mode
