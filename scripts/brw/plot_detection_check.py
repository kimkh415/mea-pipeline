#!/usr/bin/env python
"""Plot raw traces with stored samples shaded (rest is noise fill) and detected spikes marked.

    python scripts/brw/plot_detection_check.py --brw <file.brw> \
        --out figures/257.detection.png [--sorted <dir> --t0 120 --span 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sorted", help="sorted output dir, to mark detected spikes")
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--t0", type=float, default=60.0, help="window start, seconds")
    ap.add_argument("--span", type=float, default=1.0, help="window length, seconds")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mea.io.brw_fill import open_brw_with_noise_fill

    rec = open_brw_with_noise_fill(a.brw)
    fs = rec.get_sampling_frequency()
    i0 = int(a.t0 * fs)
    i1 = min(i0 + int(a.span * fs), rec.get_num_samples())
    if i1 <= i0:
        raise SystemExit(f"--t0 {a.t0} is past the end of the recording")

    # Pick the busiest channels in the window
    idx = rec.gap_index
    block = i0 // idx.block_samples
    ch_ids = rec.channel_ids
    chans, starts, ends = idx.stretches(min(block, idx.n_blocks - 1))
    busy = np.bincount(chans, minlength=len(ch_ids))
    picks = np.argsort(busy)[::-1][:a.channels]

    tr = rec.get_traces(start_frame=i0, end_frame=i1,
                        channel_ids=[ch_ids[i] for i in picks]).astype(float)
    t = np.arange(i0, i1) / fs

    spikes = None
    if a.sorted:
        d = Path(a.sorted)
        st = np.load(d / "spike_times.npy").ravel()
        spikes = st[(st >= i0) & (st < i1)] / fs

    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 1.7 * len(picks)),
                             sharex=True, squeeze=False)
    for k, ch in enumerate(picks):
        ax = axes[k][0]
        s_, e_ = idx.stored_ranges(min(block, idx.n_blocks - 1), int(ch))
        for lo, hi in zip(s_, e_):
            if hi > i0 and lo < i1:
                ax.axvspan(max(int(lo), i0) / fs, min(int(hi), i1) / fs,
                           color="#2ca02c", alpha=.16, lw=0, zorder=0)
        ax.plot(t, tr[:, k], lw=.5, color="#22223b", zorder=2)
        ax.set_ylabel(f"ch {ch}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if spikes is not None and spikes.size:
            ax.plot(spikes, np.full(spikes.size, ax.get_ylim()[1]), "v",
                    ms=3, color="#d62728", zorder=3)

    axes[-1][0].set_xlabel("seconds")
    fig.suptitle(f"{Path(a.brw).name} -- {a.span:g} s from {a.t0:g} s. "
                 "Green = stored samples; the rest is noise fill"
                 + ("; red = detected spikes (all channels)" if spikes is not None else ""),
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    print(f"  channels {list(map(int, picks))}, duty in window "
          f"{100 * sum(min(int(e), i1) - max(int(s), i0) for s, e in zip(*idx.stored_ranges(min(block, idx.n_blocks - 1), int(picks[0]))) if e > i0 and s < i1) / (i1 - i0):.1f}% on ch {int(picks[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
