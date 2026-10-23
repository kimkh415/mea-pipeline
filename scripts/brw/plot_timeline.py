#!/usr/bin/env python
"""Plot a recording's conditions on the wall clock (gaps shown) and the concatenated sample clock.

Activity is the stored-sample duty cycle, read from the TOC without decoding.

    python scripts/brw/plot_timeline.py --data <dir> --stem 257_het_100Hz \
        --out figures/257_timeline.png
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import h5py
import numpy as np

WELL = "Well_A1"


# Find mea: $MEA_REPO, else this checkout, else the installed package
for _cand in (os.environ.get("MEA_REPO"), Path(__file__).resolve().parents[2]):
    if _cand and (Path(_cand) / "src" / "mea").is_dir():
        sys.path.insert(0, str(Path(_cand) / "src"))
        break
try:
    from mea.conditions import acquisition_time      # noqa: E402
except ImportError:                                   # noqa: BLE001
    raise SystemExit(
        "cannot import `mea`. Either run this from the repo, set MEA_REPO to "
        "your checkout, or activate the environment 01_setup_env.sh built.")

# Kept for the tests; mea.conditions owns the decode
def dotnet_utc(ticks):
    """.NET DateTime: low 62 bits are 100 ns since 0001-01-01, top 2 are Kind."""
    from mea.conditions import _EPOCH, _TICK_MASK
    return _EPOCH + dt.timedelta(microseconds=(ticks & _TICK_MASK) // 10)


def read_profile(path):
    """Start, duration and per-block duty cycle; no decoding."""
    with h5py.File(path, "r") as h:
        fs = float(h.attrs["SamplingRate"])
        toc = h["TOC"][:]
        etoc = h[f"{WELL}/EventsBasedSparseRawTOC"][:]
        n_chan = int(h[f"{WELL}/StoredChIdxs"].shape[0])
        n_samples = int(toc[-1, 1])
    start = acquisition_time(path)
    if start is None:
        raise SystemExit(
            f"{Path(path).name} carries no acquisition timestamp, so it cannot "
            "be placed on a wall clock. `mea conditions --input DIR` lists what "
            "the files do carry.")

    # 2 bytes per stored sample
    stored = np.diff(etoc) / 2.0
    span = np.diff(toc[:, 0]).astype(float)
    span = np.append(span, max(toc[-1, 1] - toc[-1, 0], 1))
    duty = 100.0 * stored / (span[:len(stored)] * n_chan)
    t = toc[:len(duty), 0] / fs
    return dict(path=path, fs=fs, start=start, n_samples=n_samples,
                duration=n_samples / fs, t=t, duty=duty, n_chan=n_chan)


def gather(data_dir, stem):
    files = sorted(Path(data_dir).glob(f"{stem}*.brw"))
    if not files:
        raise SystemExit(f"no .brw files matching {stem}* in {data_dir}")
    profiles = [read_profile(f) for f in files]
    profiles.sort(key=lambda p: p["start"])      # acquisition order, not filename
    return profiles


def label_of(path, stem):
    name = Path(path).stem
    tail = name[len(stem):].strip("_").removesuffix("_00")
    return tail or "baseline"


def draw(profiles, stem, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = profiles[0]["start"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(profiles)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9.5))
    fig.suptitle(f"{stem} — where the time goes", fontsize=14, y=0.97)

    top = max(p["duty"].max() for p in profiles) * 1.18
    lane_hi, lane_lo = -top * 0.045, -top * 0.135      # the condition lane

    def lane(ax, x0, width, color, text):
        ax.add_patch(plt.Rectangle((x0, lane_lo), width, lane_hi - lane_lo,
                                   color=color, alpha=0.9, lw=0, clip_on=False))
        ax.text(x0 + width / 2, (lane_lo + lane_hi) / 2, text, ha="center",
                va="center", fontsize=8, color="white", weight="bold",
                clip_on=False)

    # --- top: wall clock, gaps shown ---
    gaps = []
    prev_end = None
    for p, c in zip(profiles, colors):
        off = (p["start"] - t0).total_seconds() / 60.0
        x = off + p["t"] / 60.0
        ax1.fill_between(x, 0, p["duty"], color=c, lw=0, alpha=0.9)
        lane(ax1, off, p["duration"] / 60.0, c, label_of(p["path"], stem))
        if prev_end is not None:
            g = (p["start"] - prev_end).total_seconds()
            gaps.append(g)
            ax1.axvspan(off - g / 60.0, off, color="#d62728", alpha=0.15, lw=0)
            ax1.text(off - g / 120.0, top * 0.94, f"{g:.0f}s", ha="center",
                     fontsize=8, color="#d62728", weight="bold")
        prev_end = p["start"] + dt.timedelta(seconds=p["duration"])

    rec = sum(p["duration"] for p in profiles)
    span = (prev_end - t0).total_seconds()
    ax1.set_xlim(-1, span / 60.0 + 1)
    ax1.set_ylabel("stored-sample duty cycle (%)")
    ax1.set_xlabel("wall-clock minutes from the start of the experiment")
    ax1.set_title(
        f"AS RECORDED — {span/60:.1f} min elapsed, {rec/60:.1f} min of it in a file. "
        f"Red = recorder off ({(span-rec)/60:.1f} min total, {len(gaps)} gaps).",
        fontsize=10, loc="left")

    # --- bottom: sample clock, gaps removed ---
    off = 0.0
    for i, (p, c) in enumerate(zip(profiles, colors)):
        x = off + p["t"] / 60.0
        ax2.fill_between(x, 0, p["duty"], color=c, lw=0, alpha=0.9)
        lane(ax2, off, p["duration"] / 60.0, c, label_of(p["path"], stem))
        if i:
            ax2.axvline(off, color="#d62728", ls="--", lw=1.4, alpha=0.9)
            ax2.text(off, top * 0.94, f" −{gaps[i-1]:.0f}s", fontsize=8,
                     color="#d62728", weight="bold", va="top")
        off += p["duration"] / 60.0

    ax2.set_xlim(-1, off + 1)
    ax2.set_ylabel("stored-sample duty cycle (%)")
    ax2.set_xlabel("minutes on the concatenated sample clock")
    ax2.set_title(
        "AS SORTED — the gaps are gone. Each dashed line is a splice where "
        f"{min(gaps):.0f}–{max(gaps):.0f} s of real time was removed.",
        fontsize=10, loc="left")

    for ax in (ax1, ax2):
        ax.set_ylim(lane_lo, top)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= top])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.xaxis.set_label_coords(0.5, -0.17)
        ax.tick_params(axis="x", pad=26)          # clear of the condition lane

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return gaps, rec, span


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    profiles = gather(a.data, a.stem)
    print(f"{'condition':<24s} {'start (UTC)':<20s} {'min':>7s} {'duty %':>8s} "
          f"{'quietest block':>15s}")
    for p in profiles:
        print(f"{label_of(p['path'], a.stem):<24s} "
              f"{p['start'].strftime('%Y-%m-%d %H:%M:%S'):<20s} "
              f"{p['duration']/60:>7.1f} {p['duty'].mean():>8.2f} "
              f"{p['duty'].min():>15.2f}")

    gaps, rec, span = draw(profiles, a.stem, a.out)
    print(f"\nrecorded {rec/60:.1f} min over a {span/60:.1f} min experiment; "
          f"{(span-rec)/60:.1f} min ({100*(span-rec)/span:.0f}%) exists in no file")
    print(f"gaps between conditions: {', '.join(f'{g:.0f}s' for g in gaps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
