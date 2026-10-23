#!/usr/bin/env python
"""Survey 3Brain .brw files (storage mode, duty cycle) straight from the HDF5.

    python scripts/brw/survey.py <dir-or-file> [...]
    python scripts/brw/survey.py data/ --pair 257_het_100Hz   # baseline vs its TTX
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from read_sparse import blocks, parse_block          # noqa: E402

GRID = 64


def settings(h):
    try:
        raw = h["ExperimentSettings"][0]
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:                                # noqa: BLE001
        return {}


def stretches(path, nblocks=8):
    h, raw, toc = blocks(path)
    ev = []
    for i in range(min(nblocks, toc.size - 1)):
        a, b = int(toc[i]), int(toc[i + 1])
        st, ok = parse_block(np.asarray(raw[a:b]))
        if not ok:
            raise SystemExit(f"{path}: block {i} did not parse; the layout differs")
        ev += st
    return h, np.array(ev, dtype=np.int64)


def describe(path, nblocks=8):
    h, ev = stretches(path, nblocks)
    s = settings(h)
    comp = s.get("DataSettings", {}).get("EventsBasedRawRanges", {})
    span = int(ev[:, 2].max() - ev[:, 1].min())
    n_chan = int(np.asarray(h["Well_A1/StoredChIdxs"]).size)
    return {
        "file": Path(path).name,
        "version": int(h.attrs.get("Version", -1)),
        "fs": float(h.attrs["SamplingRate"]),
        "n_chan": n_chan,
        "neighbors": comp.get("NeighborsScope", "?"),
        "thresholds": (comp.get("LowerNegativeStdFactor"),
                       comp.get("HigherNegativeStdFactor")),
        "stretches": len(ev),
        "seconds": span / float(h.attrs["SamplingRate"]),
        "duty": float((ev[:, 2] - ev[:, 1]).sum()) / (n_chan * span),
        "_ev": ev,
    }


def neighbours(ev, shuffle=False, seed=0, maxn=20000):
    """Mean co-active channels within 1 and 2 electrodes, observed or by chance."""
    ch, s0, s1 = ev[:, 0].copy(), ev[:, 1], ev[:, 2]
    if shuffle:
        ch = np.random.default_rng(seed).permutation(ch)
    o = np.argsort(s0)
    ch, s0, s1 = ch[o], s0[o], s1[o]
    row, col = ch // GRID, ch % GRID
    n, step = len(s0), max(1, len(s0) // maxn)
    a1 = a2 = cnt = 0
    for i in range(0, n, step):
        a, b = s0[i], s1[i]
        lo, hi = np.searchsorted(s0, a - 250), np.searchsorted(s0, b)
        j = np.arange(lo, hi)
        j = j[(s1[j] > a) & (j != i)]
        d = np.maximum(np.abs(row[j] - row[i]), np.abs(col[j] - col[i]))
        a1 += int((d == 1).sum())
        a2 += int(((d >= 1) & (d <= 2)).sum())
        cnt += 1
    return a1 / cnt, a2 / cnt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pair", metavar="STEM",
                    help="compare this recording's baseline against its TTX")
    ap.add_argument("--blocks", type=int, default=8,
                    help="TOC blocks to read per file (default 8)")
    a = ap.parse_args(argv)

    files = []
    for p in a.paths:
        p = Path(p)
        files += sorted(p.rglob("*.brw")) if p.is_dir() else [p]
    if not files:
        raise SystemExit("no .brw files found")

    if a.pair:
        base = [f for f in files if f.stem.startswith(a.pair)
                and not any(d in f.stem[len(a.pair):].lower()
                            for d in ("ttx", "gabazine", "cgp", "ap5", "dnqx"))]
        ttx = [f for f in files if f.stem.startswith(a.pair) and "ttx" in f.stem.lower()]
        if not base or not ttx:
            raise SystemExit(f"need a baseline and a TTX file for {a.pair}")
        print(f"=== {a.pair}: what fraction of stored events are spikes? ===\n")
        print(f"{'':10s} {'stretches':>10s} {'duty':>8s} {'nb 60um':>9s} {'chance':>8s} {'ratio':>7s}")
        for tag, f in (("baseline", base[0]), ("TTX", ttx[0])):
            d = describe(f, a.blocks)
            o1, _ = neighbours(d["_ev"])
            c1, _ = neighbours(d["_ev"], shuffle=True)
            print(f"{tag:10s} {d['stretches']:10,d} {d['duty']:8.2%} "
                  f"{o1:9.3f} {c1:8.3f} {o1 / max(c1, 1e-9):6.2f}x")
        print("\nTTX blocks every action potential, so its row is what pure noise"
              "\nlooks like.  A baseline close to it is mostly noise too.")
        return 0

    print(f"{'file':52s} {'ver':>4s} {'chan':>5s} {'neighbors':>12s} "
          f"{'stretches':>10s} {'duty':>7s}")
    for f in files:
        try:
            d = describe(f, a.blocks)
        except Exception as e:                       # noqa: BLE001
            print(f"{f.name[:52]:52s}  {type(e).__name__}: {e}")
            continue
        print(f"{d['file'][:52]:52s} {d['version']:4d} {d['n_chan']:5d} "
              f"{d['neighbors']:>12s} {d['stretches']:10,d} {d['duty']:7.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
