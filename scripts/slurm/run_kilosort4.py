#!/usr/bin/env python3
"""Sort one recording with Kilosort 4 and write the files stage 1 reads.

All conditions are concatenated and sorted together so a cluster id is the same
unit across conditions; each condition's sample range goes to recording_data.csv.
Condition order comes from the pipeline's ordering (`--config`), not file mtime.
Unit types are recorded, not applied.

Usage:
    python run_kilosort4.py --data DIR --stem 268_wt_100Hz --out DIR --scratch DIR
"""
from __future__ import annotations

import argparse
import os as _os

# bombcell calls plt.show() otherwise, which blocks on a machine with a display
_os.environ["MPLBACKEND"] = "Agg"

# bombcell's raw-waveform joblib uses n_jobs=-1 (one mmap per core); cap it
_os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# OpenBLAS threads deadlock in Kilosort's SVD; set before numpy loads
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Use the pipeline's src for the shared condition ordering
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CLUSTER_ID_STRIDE = 1000        # must exceed the clusters any one block can find


# --------------------------------------------------------------- conditions
def ordered_conditions(data_dir, stem, cfg):
    """Every .brw condition of `stem`, in recorded order (via mea.conditions)."""
    from mea.conditions import order_condition_files, acquisition_time

    files = sorted(Path(data_dir).glob(f"{stem}*.brw"))
    if not files:
        raise SystemExit(f"no .brw files matching {stem}* in {data_dir}")

    mode = (cfg.get("dataset", {}).get("discovery", {}) or {}).get("condition_order")
    if mode == "file_time":
        missing = [f.name for f in files if acquisition_time(f) is None]
        if missing:
            raise SystemExit(
                "these files carry no acquisition timestamp, so the condition "
                "order cannot be read from the data:\n  " + "\n  ".join(missing) +
                "\n\nSay the order explicitly instead of falling back on the "
                "file system clock, which belongs to the copy and not the "
                "experiment.  In a config file passed with --config:\n"
                "\n  dataset:\n    discovery:\n"
                "      condition_order: explicit\n"
                "      order: [Baseline, Gabazine, CGP, D-AP5, DNQX, TTX]\n"
                "\n`order` accepts names (a case-insensitive substring of each "
                "filename) or 1-based positions in the alphabetical listing.\n"
                "`mea conditions --input DIR` prints that listing.")
    ordered, how = order_condition_files(files, cfg)
    print(f"    condition order: {how}")
    return list(ordered)


# ------------------------------------------------------------------- record
def write_recording_data(paths, recordings, out_dir, total=None, gaps=None):
    """Write each condition's sample range in the concatenated stream.

    `total` clips ranges to a `--tmax` head (later conditions become zero-length rows);
    `gaps` is the pad inserted before each condition under --restore-gaps.
    """
    rows, start = [], 0
    for i, (p, rec) in enumerate(zip(paths, recordings)):
        if gaps is not None:
            start += int(gaps[i])
        n_full = int(rec.get_num_samples())
        lo = start if total is None else min(start, total)
        hi = start + n_full if total is None else min(start + n_full, total)
        n = max(hi - lo, 0)
        rows.append({"recording_name": p.name,
                     "duration": n / float(rec.get_sampling_frequency()),
                     "num_samples": n,
                     "sampling_freq": float(rec.get_sampling_frequency()),
                     "sample_start": lo,
                     "sample_end": lo + n})
        start += n_full
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "recording_data.csv", index=False)
    return df


# Accura electrode pitch (um); checked against the geometry at sort time
PITCH_UM = 60.0


def force_utf8():
    """Force UTF-8 stdout/stderr; bombcell's emoji crash a non-UTF-8 console."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() \
                not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_blocks(spec, n_blocks):
    """`None` -> every block; `"0"` -> [0]; `"1-7"` -> [1..7]; `"0,3"` -> [0, 3]."""
    if not spec:
        return list(range(n_blocks))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out += list(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    bad = [b for b in out if not 0 <= b < n_blocks]
    if bad:
        raise SystemExit(f"blocks {bad} are outside 0-{n_blocks - 1}")
    return sorted(dict.fromkeys(out))


# --------------------------------------------------------------------- sort
def sort_block(recording, block, block_no, out_dir, params):
    """Kilosort 4 on one slice of channels, via `run_sorter`."""
    from spikeinterface.sorters import run_sorter

    ids = recording.channel_ids[block]
    sub = recording.select_channels(ids)

    # Check pitch: probeinterface derives geometry from the chip model name
    loc = sub.get_channel_locations()
    steps = np.diff(np.unique(np.round(loc[:, 1], 3)))
    if steps.size:
        pitch = float(np.median(steps))
        if abs(pitch - PITCH_UM) > 1.0:
            raise SystemExit(
                f"electrode pitch reads {pitch:.1f} um, expected {PITCH_UM:g}. "
                "The spacing settings are in microns, so they mean something "
                "different at this geometry. Confirm the chip model against "
                "the datasheet before sorting.")

    # Rough GPU memory floor (~8.2 GB per 512 channels); fail before writing the binary
    need = 8.2 * sub.get_num_channels() / 512
    try:
        import torch
        have = torch.cuda.mem_get_info()[1] / 1e9
        print(f"    GPU {have:.0f} GB; {sub.get_num_channels()} channels need "
              f"roughly {need:.0f} GB", flush=True)
        if have < need:
            raise SystemExit(
                f"this GPU has {have:.0f} GB and {sub.get_num_channels()} "
                f"channels are expected to need about {need:.0f} GB.\n"
                "Either request a larger card, or split the array with "
                "--n-blocks 8 -- which costs little on recordings made with "
                "NeighborsScope NoNeighbors.")
    except ImportError:
        pass

    folder = out_dir / f"kilosort4_{block_no}"
    t0 = time.time()
    run_sorter(sorter_name="kilosort4", recording=sub, folder=str(folder),
               verbose=True, **params)
    return folder / "sorter_output", time.time() - t0


def _close_figures():
    """Close bombcell's figures so they don't accumulate across blocks."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:                            # noqa: BLE001
        pass


def _bombcell_unit_type(out):
    """Unit types from run_bombcell's return (dict, or tuple with types at index 2)."""
    if isinstance(out, dict):
        return out["unit_type"]
    if isinstance(out, (tuple, list)):
        for cand in (out[2],) if len(out) > 2 else ():
            arr = np.asarray(cand)
            if arr.ndim == 1 and arr.dtype.kind in "iufb":
                return arr
        raise SystemExit(
            f"run_bombcell returned {len(out)} values and position 2 is not a "
            f"1-D numeric array of unit types (got {type(out[2]).__name__} at 2). "
            "Its signature has changed; check bombcell's docstring and fix "
            "_bombcell_unit_type.")
    raise SystemExit(f"run_bombcell returned an unexpected {type(out).__name__}")


# bombcell overrides. No .brw reader, so it runs from templates only.
# TODO: spatial-decay threshold is bombcell's Neuropixels default; unvalidated for a 60 um grid.
BOMBCELL_PARAM = {
    # Separate quality from somatic/non-somatic:
    #   1 good somatic   2 MUA somatic   3 good non-somatic   4 MUA non-somatic
    "splitGoodAndMua_NonSomatic": True,
    # Disable the baseline-flatness noise rule (ratio never exceeds 1); TTX test covers it
    "maxWvBaselineFraction": 1.0,
    "extract_raw_waveforms": False,
    # Gives figures a destination, so bombcell doesn't call plt.show()
    "savePlots": True,
    "plotGlobal": True,
}



def unit_types(results_dir, n_clusters, fs, allow_fallback=False):
    """Per-cluster unit types from bombcell, or Kilosort's KSLabel if allowed.

    Returns (types, source): 0 noise / 1 good somatic / 2 MUA somatic /
    3 good non-somatic / 4 MUA non-somatic.
    """
    try:
        import bombcell as bc
        param = bc.get_default_parameters(str(results_dir), kilosort_version=4)
        param.update(BOMBCELL_PARAM)
        param["ephys_sample_rate"] = float(fs)
        out = bc.run_bombcell(str(results_dir), str(results_dir / "bombcell"),
                              param=param)
        # Save the params used
        (results_dir / "bombcell_param.json").write_text(json.dumps(
            {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
             for k, v in sorted(param.items())}, indent=2))
        types = np.asarray(_bombcell_unit_type(out))
        # NaN (template with no spikes) -> 0, not INT64_MIN
        types = np.where(np.isfinite(types), types, 0).astype(int)
        if types.size != n_clusters:
            raise SystemExit(
                f"bombcell returned {types.size} unit types for {n_clusters} "
                "clusters; the two must line up or the labels attach to the "
                "wrong units.")
        # 0-4 with splitGoodAndMua_NonSomatic on; 0-3 with it off.
        bad = sorted(set(np.unique(types)) - {0, 1, 2, 3, 4})
        if bad:
            raise SystemExit(f"bombcell returned unit types outside 0-4: {bad}")
        _close_figures()
        return types, "bombcell"
    except SystemExit:
        raise
    except Exception as e:                       # noqa: BLE001
        import traceback
        msg = (f"bombcell failed ({type(e).__name__}: {e})\n"
               + traceback.format_exc())
        if not allow_fallback:
            raise SystemExit(
                msg + "\n\nThe pipeline's unit selection was set against "
                "bombcell's four-way call, so falling back on Kilosort's "
                "good/mua silently changes which units survive.  Fix bombcell, "
                "or pass --allow-kslabel-fallback to accept the weaker labels "
                "knowingly; they are marked in metadata.csv either way.\n"
                "On Windows, a UnicodeEncodeError here is bombcell's emoji "
                "meeting a non-UTF-8 console: set PYTHONUTF8=1.")
        print(f"    {msg} -- falling back on Kilosort labels", flush=True)

    labels = np.ones(n_clusters, int)
    tsv = results_dir / "cluster_KSLabel.tsv"
    if tsv.exists():
        df = pd.read_csv(tsv, sep="\t")
        col = "KSLabel" if "KSLabel" in df.columns else df.columns[-1]
        m = {"good": 1, "mua": 2}
        for cid, lab in zip(df.iloc[:, 0], df[col]):
            if 0 <= int(cid) < n_clusters:
                labels[int(cid)] = m.get(str(lab).lower(), 2)
    return labels, "kilosort_kslabel"


def collect(block_dirs, block_nos, per_block, out_dir, source, fs,
            allow_fallback=False):
    """Pool the blocks into the files stage 1 reads.

    Ids and channel offsets come from `block_nos` (blocks actually sorted), not list position.
    """
    if len(block_dirs) != len(block_nos):
        raise SystemExit("block_dirs and block_nos must line up")
    times, clusters, meta, waves = [], [], [], []
    for d, i in zip(block_dirs, block_nos):
        st = np.load(d / "spike_times.npy").ravel().astype(np.int64)
        cl = np.load(d / "spike_clusters.npy").ravel().astype(np.int64)
        n_clusters = int(cl.max()) + 1 if cl.size else 0
        # unitType from the bombcell package, not spikeinterface's reimplementation
        types, how = unit_types(d, n_clusters, fs, allow_fallback)

        # Offset ids so they stay unique across blocks
        offset = i * CLUSTER_ID_STRIDE
        times.append(st)
        clusters.append(cl + offset)

        pos = np.load(d / "channel_positions.npy")
        templates = np.load(d / "templates.npy")
        peak = np.abs(templates).max(axis=1).argmax(axis=1) if templates.size else \
            np.zeros(n_clusters, int)
        # Template on each unit's peak channel (2-D float)
        if templates.size:
            waves.append(templates[np.arange(n_clusters), :, peak[:n_clusters]])
        chan = peak + i * per_block
        # Kilosort's good/mua, kept beside bombcell's (stage 5 uses both)
        ks_label = np.array(["unknown"] * n_clusters, dtype=object)
        ks_tsv = d / "cluster_KSLabel.tsv"
        if ks_tsv.exists():
            t = pd.read_csv(ks_tsv, sep="\t")
            t.columns = ["cluster_id", "KSLabel"]
            m = dict(zip(t["cluster_id"].astype(int), t["KSLabel"].astype(str)))
            for cid in range(n_clusters):
                ks_label[cid] = m.get(cid, "unknown")

        meta.append(pd.DataFrame({
            "cluster_id": np.arange(n_clusters),
            "new_cluster_id": np.arange(n_clusters) + offset,
            "unitType": types[:n_clusters],
            "KSLabel": ks_label,
            "peak_channel": chan[:n_clusters],
            "peak_chan_x": pos[peak[:n_clusters], 0] if pos.size else 0,
            "peak_chan_y": pos[peak[:n_clusters], 1] if pos.size else 0,
            "unit_type_source": how,
            "block": i,
        }))

    order = np.argsort(np.concatenate(times), kind="stable")
    np.save(out_dir / "spike_times.npy", np.concatenate(times)[order])
    np.save(out_dir / "spike_clusters.npy", np.concatenate(clusters)[order])
    md = pd.concat(meta, ignore_index=True)
    md.to_csv(out_dir / "metadata.csv", index=False)
    if waves:
        np.save(out_dir / "waveforms.npy", np.concatenate(waves).astype(np.float32))
    return md


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:                            # noqa: BLE001
        return None


def _version(mod):
    """Module version, or a placeholder; never raises."""
    try:
        return __import__(mod).__version__
    except Exception as e:                       # noqa: BLE001
        return f"<unavailable: {type(e).__name__}>"


def provenance(out_dir, stem, paths, settings, timings, md):
    (out_dir / "sort_provenance.json").write_text(json.dumps({
        "recording": stem,
        "sorted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": [p.name for p in paths],
        "mea": _version("mea"),
        "kilosort": _version("kilosort"),
        "spikeinterface": _version("spikeinterface"),
        "torch": _version("torch"),
        "gpu": _gpu_name(),
        "host": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "settings": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                     for k, v in settings.items()},
        "numpy": __import__("numpy").__version__,   # gap fill's RNG stream depends on it

        "block_seconds": timings,
        "n_clusters": int(len(md)),
        "unit_type_counts": {str(k): int(v) for k, v in
                             md["unitType"].value_counts().sort_index().items()},
        "unit_type_source": sorted(md["unit_type_source"].unique().tolist()),
    }, indent=2))


def _default_n_jobs() -> int:
    """$SLURM_CPUS_PER_TASK, else 1. Output is identical at any worker count."""
    try:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    except (KeyError, ValueError):
        return 1


def build_parser():
    """Build the CLI parser (separate from main so defaults are testable)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory holding the .brw files")
    ap.add_argument("--stem", required=True, help="recording stem shared by its conditions")
    ap.add_argument("--out", required=True, help="where to write the sorted output")
    ap.add_argument("--scratch", help="unused; spikeinterface writes its "
                    "recording.dat inside each sorter folder. Kept so existing "
                    "SLURM scripts do not break.")
    ap.add_argument("--n-blocks", type=int, default=1,
                    help="split the array into this many channel blocks "
                         "(default 1: sort all 4096 at once, which is what "
                         "Kilosort 4 is designed for). Use 8 only if the whole "
                         "array will not fit on the GPU.")
    ap.add_argument("--dminx", type=float, default=PITCH_UM,
                    help=f"horizontal spacing of candidate spike locations, um "
                         f"(default {PITCH_UM:g}, the electrode pitch; "
                         "Kilosort's own default is 32)")
    ap.add_argument("--max-channel-distance", type=float, default=PITCH_UM,
                    help=f"drop candidate locations further than this from an "
                         f"electrode, um (default {PITCH_UM:g})")
    ap.add_argument("--th-universal", type=float, default=9.0,
                    help="detection threshold for universal templates (default 9)")
    ap.add_argument("--th-learned", type=float, default=8.0,
                    help="detection threshold for learned templates (default 8)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the gap fill; recorded in the provenance")
    ap.add_argument("--allow-kslabel-fallback", action="store_true",
                    help="accept Kilosort's good/mua when bombcell fails; the "
                         "pipeline's thresholds were set against bombcell")
    ap.add_argument("--no-drift-correction", dest="do_correction",
                    action="store_false",
                    help="turn off Kilosort 4's drift correction, as Yang's "
                         "2.5 script had it. On by default: on the "
                         "four-organoid test it left 5-12%% fewer clusters, "
                         "nearly all of them ones bombcell calls noise, and "
                         "the same biology.")
    ap.add_argument("--restore-gaps", action="store_true", default=False,
                    help="pad between conditions with the recorder-off time "
                         "the acquisition clocks report, instead of splicing "
                         "the files together. Lengthens the sort by 7-13%%.")
    ap.add_argument("--fill-gaps", default="noise",
                    choices=["noise", "zeros"],
                    help="what goes between stored stretches: 'noise' (default) "
                         "is mea.io.brw_fill's seeded draw at the channel's own "
                         "NoiseStdDev; 'zeros' is neo's constant 2048, kept only "
                         "to reproduce the diluted-whitening failure")
    ap.add_argument("--n-jobs", type=int, default=_default_n_jobs(),
                    help="workers spikeinterface uses to write the binary "
                         "(default: $SLURM_CPUS_PER_TASK, else 1). This step "
                         "dominates the wall time -- measured 4.6 s per 1 s "
                         "chunk at one worker, 3 h 47 for 268's 2,940 s -- and "
                         "it is the only part of the run that parallelises "
                         "without changing the answer.")
    ap.add_argument("--collect-only", action="store_true",
                    help="pool blocks already sorted by earlier jobs and write "
                         "the four files; sorts nothing. For the final task of "
                         "a per-block array job.")
    ap.add_argument("--blocks", metavar="SPEC",
                    help="sort only these channel blocks, e.g. '0' or '1-7' "
                         "(default all); the rest are left untouched")
    ap.add_argument("--tmax", type=float,
                    help="sort only the first N seconds, for a smoke test")
    ap.add_argument("--keep-scratch", action="store_true", default=False,
                    help="keep each block's recording.dat; about 59 GB per block")
    ap.add_argument("--drop-scratch", dest="keep_scratch", action="store_false",
                    help="delete the intermediates when the run finishes")
    ap.add_argument("--config", metavar="YAML",
                    help="pipeline config; use it to set dataset.discovery."
                         "condition_order when the files carry no usable clock")
    return ap


def main(argv=None):
    force_utf8()
    a = build_parser().parse_args(argv)

    from spikeinterface import concatenate_recordings
    from spikeinterface.extractors import read_biocam
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mea.io.brw_fill import open_brw_with_noise_fill

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    from mea import config as cfgmod
    cfg = cfgmod.load(a.config, [])
    paths = ordered_conditions(a.data, a.stem, cfg)
    print(f"=== {len(paths)} conditions, in acquisition order ===")
    for i, p in enumerate(paths, 1):
        print(f"  {i}. {p.name}")

    # Fill sparse gaps with seeded noise; neo's `zeros` dilutes whitening (see mea.io.brw_fill)
    recordings = []
    for p_ in paths:
        recordings.append(read_biocam(str(p_), fill_gaps_strategy="zeros")
                          if a.fill_gaps == "zeros"
                          else open_brw_with_noise_fill(p_, seed=a.seed))

    rates = {round(float(r.get_sampling_frequency()), 3) for r in recordings}
    if len(rates) > 1:
        raise SystemExit(
            f"conditions were recorded at different sampling rates ({sorted(rates)}). "
            "Spike times are indices into one concatenated stream, so a single "
            "rate is needed to place them.")
    if a.n_jobs != 1:
        import spikeinterface as _si
        _si.set_global_job_kwargs(n_jobs=a.n_jobs, chunk_duration="1s")
        if a.collect_only:
            print(f"{a.n_jobs} workers available (no binary to write)")
        else:
            print(f"writing the binary with {a.n_jobs} workers "
                  "(output is identical to 1; only the wall time changes)")

    gaps = None
    sequence = recordings
    if a.restore_gaps:
        from mea.io.gap_pad import restore_gaps
        sequence, gaps = restore_gaps(paths, recordings, seed=a.seed)
        if sequence is None:
            raise SystemExit(
                "--restore-gaps needs an acquisition timestamp in every file, "
                "and at least one has none. Restoring a gap without a clock "
                "would mean inventing the quantity this arm exists to measure.\n"
                "`mea conditions --input DIR` prints what the files carry.")
        fs_ = float(recordings[0].get_sampling_frequency())
        added = sum(gaps) / fs_
        print(f"restoring {added / 60:.1f} min of recorder-off time in "
              f"{sum(1 for g in gaps if g > 0)} gaps: "
              f"{', '.join(f'{g / fs_:.0f}s' for g in gaps[1:])}")
        print("  the conditions now sit on the wall clock, not end to end\n")

    full = concatenate_recordings(sequence)
    n_chan = full.get_num_channels()
    per_block = int(np.ceil(n_chan / a.n_blocks))
    if a.n_blocks == 1:
        print(f"{n_chan} channels, sorted whole\n")
    else:
        # Blocking removes real neighbours from units beside a cut
        print(f"{n_chan} channels in {a.n_blocks} blocks of {per_block}")
        print("NOTE: blocking is a memory workaround; --n-blocks 1 is the "
              "default\n")

    # dminx / max_channel_distance in um (KS4 default 32; this array is 60)
    params = {
        "do_CAR": True,
        "do_correction": a.do_correction,
        "skip_kilosort_preprocessing": False,
        "dminx": a.dminx,
        "max_channel_distance": a.max_channel_distance,
        "Th_universal": a.th_universal,
        "Th_learned": a.th_learned,
        # Kept for --collect-only; main() deletes it after collect() unless --keep-scratch
        "delete_recording_dat": False,
    }

    if a.tmax:
        # Smoke test writes to a separate folder
        full = full.frame_slice(0, int(a.tmax * full.get_sampling_frequency()))
        out_dir = out_dir.parent / f"smoke_{out_dir.name}_tmax{int(a.tmax)}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"SMOKE TEST: first {a.tmax:.0f} s only, writing to {out_dir}\n")

    # After the smoke redirect, so a smoke never writes into the real output
    df = write_recording_data(paths, recordings, out_dir,
                              total=full.get_num_samples(), gaps=gaps)
    print(f"total {df['num_samples'].sum():,} samples "
          f"({df['duration'].sum():.0f} s) at {rates.pop():.1f} Hz\n")

    wanted = parse_blocks(a.blocks, a.n_blocks)

    if a.collect_only:
        block_dirs = [out_dir / f"kilosort4_{b}" / "sorter_output" for b in wanted]
        missing = [str(d) for d in block_dirs if not (d / "spike_times.npy").exists()]
        if missing:
            raise SystemExit(
                "--collect-only needs every block already sorted; these have no "
                "spike_times.npy:\n  " + "\n  ".join(missing))
        timings = [0.0] * len(wanted)
        print(f"pooling {len(block_dirs)} already-sorted blocks, no sorting")
    else:
        block_dirs, timings = [], []
        for b in wanted:
            sl = slice(b * per_block, min((b + 1) * per_block, n_chan))
            print(f"--- block {b}: channels {sl.start}-{sl.stop - 1}", flush=True)
            d, secs = sort_block(full, sl, b, out_dir, params)
            print(f"    {secs / 60:.1f} min", flush=True)
            block_dirs.append(d); timings.append(round(secs, 1))

    if len(wanted) != a.n_blocks:
        print(f"\nNOTE: blocks {sorted(set(range(a.n_blocks)) - set(wanted))} "
              "were not sorted, so this output covers only part of the array.")

    # bombcell raises on single-spike clusters (presence_ratio takes the 90th
    # percentile of an empty array), which only a smoke produces.
    allow_fallback = a.allow_kslabel_fallback or bool(a.tmax)
    if allow_fallback and not a.allow_kslabel_fallback:
        print("smoke run: accepting Kilosort's labels if bombcell fails "
              "(too few spikes per cluster for its metrics)")
    md = collect(block_dirs, wanted, per_block, out_dir, a.stem,
                 full.get_sampling_frequency(), allow_fallback,
                 )
    provenance(out_dir, a.stem, paths, params, timings, md)

    print(f"\n=== {len(md)} clusters ===")
    print(md["unitType"].value_counts().sort_index().to_string())
    print(f"\nwrote {out_dir}")
    for f in ("spike_times.npy", "spike_clusters.npy", "metadata.csv",
              "recording_data.csv", "sort_provenance.json"):
        print(f"  {f}")

    # Delete recording.dat (hundreds of GB on shared scratch)
    if not a.keep_scratch:
        freed = 0
        for d in block_dirs:
            dat = Path(d) / "recording.dat"
            if dat.is_file():
                freed += dat.stat().st_size
                dat.unlink()
        if freed:
            print(f"\nrecording.dat: deleted ({freed / 1e9:.0f} GB freed)")
        else:
            print("\nrecording.dat: none to delete")
    else:
        print("\nrecording.dat: kept (--keep-scratch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
