#!/usr/bin/env python
"""Decode one channel block of 3Brain `.brw` files into a flat uint16 binary for Kilosort.

Blanked samples are filled with `2048 + N(0, NoiseStdDev[ch])`, seeded per
(seed, file, channel, block); the fill is reproducible only on the same numpy build.
Pass conditions in acquisition order; they are concatenated as given.

    python scripts/brw/to_binary.py OUT.bin --files a.brw b.brw --block 6
    python scripts/brw/to_binary.py OUT.bin --files a.brw --block 6 --fill zeros
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from read_sparse import parse_block                         # noqa: E402

WELL = "Well_A1"
BASELINE = 2048.0          # the "0" of the unsigned 12-bit representation
DEFAULT_SD = 4.0           # for a channel the file never characterises


def noise_table(h, n_chan):
    """Per-TOC-block noise SD per channel, carried forward between noise updates.

    Returns (tables, median SD); unknown channels get the median.
    """
    g = h[WELL]
    toc = np.asarray(g["NoiseTOC"])
    idx = np.asarray(g["NoiseChIdxs"])
    sd = np.asarray(g["NoiseStdDev"])
    out = []
    cur = np.full(n_chan, np.nan)
    for i in range(toc.size - 1):
        a, b = int(toc[i]), int(toc[i + 1])
        if b > a:
            ch = idx[a:b]
            ok = (ch >= 0) & (ch < n_chan)
            cur = cur.copy()
            cur[ch[ok]] = sd[a:b][ok]
        out.append(cur)
    med = np.nanmedian(np.concatenate([c[np.isfinite(c)] for c in out])) \
        if any(np.isfinite(c).any() for c in out) else DEFAULT_SD
    return [np.where(np.isfinite(c), c, med) for c in out], float(med)


def block_extent(raw, toc):
    """Samples per TOC block (measured from stretch indices) and the block count."""
    firsts = []
    for i in range(toc.size - 1):
        a, b = int(toc[i]), int(toc[i + 1])
        st, ok = parse_block(np.asarray(raw[a:b]))
        if not ok:
            raise SystemExit(f"TOC block {i} did not parse; the layout differs")
        if st:
            firsts.append((i, min(s for _, s, _ in st)))
        if len(firsts) >= 3:
            break
    if len(firsts) < 2:
        raise SystemExit("too few populated TOC blocks to measure the block size")
    (i0, s0), (i1, s1) = firsts[0], firsts[1]
    size, rem = divmod(s1 - s0, i1 - i0)
    if rem or size <= 0:
        raise SystemExit(f"TOC blocks are not a uniform partition ({s1 - s0} over "
                         f"{i1 - i0} blocks); this reader assumes they are")
    return int(size), int(toc.size - 1)


def convert(path, channels, out, seed, fill):
    """Append one file's samples for `channels` to the open binary `out`.

    Returns (samples written, channels written).
    """
    h = h5py.File(str(path), "r")
    g = h[WELL]
    raw, toc = g["EventsBasedSparseRaw"], np.asarray(g["EventsBasedSparseRawTOC"])
    n_chan_total = int(np.asarray(g["StoredChIdxs"]).size)
    size, n_blocks = block_extent(raw, toc)
    sds, _ = noise_table(h, n_chan_total)
    want = np.asarray(channels, int)
    col = {int(c): k for k, c in enumerate(want)}
    name = Path(path).name

    for i in range(n_blocks):
        buf = np.empty((size, want.size), np.uint16)
        if fill == "zeros":
            buf[:] = np.uint16(BASELINE)
        else:
            sd = sds[i] if i < len(sds) else sds[-1]
            for k, c in enumerate(want):
                # Seed on (seed, file, channel, block), not call order, so bytes are reproducible
                r = np.random.default_rng(
                    [seed, _key(name), int(c), i])
                buf[:, k] = np.clip(
                    np.rint(BASELINE + r.normal(0.0, float(sd[c]), size)),
                    0, 4095).astype(np.uint16)
        _splice(raw, toc, i, col, buf, i * size)
        out.write(buf.tobytes(order="C"))
    h.close()
    return size * n_blocks, want.size


def _key(text):
    """A stable small integer from a filename, for seeding.  Not `hash`."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(),
                          "big")


def _splice(raw, toc, i, col, buf, lo):
    """Overwrite the fill with this TOC block's stored samples, in place."""
    a, b = int(toc[i]), int(toc[i + 1])
    data = np.asarray(raw[a:b])
    pos, n, hi = 0, data.size, lo + buf.shape[0]
    while pos + 8 <= n:
        ch = int(data[pos:pos + 4].view("<i4")[0]); pos += 4
        ln = int(data[pos:pos + 4].view("<i4")[0]); pos += 4
        end = pos + ln
        k = col.get(ch)
        if k is None:                      # not a channel we want: skip the payload
            pos = end
            continue
        while pos < end:
            s = int(data[pos:pos + 8].view("<i8")[0]); pos += 8
            e = int(data[pos:pos + 8].view("<i8")[0]); pos += 8
            cnt = e - s
            a0, b0 = max(s, lo), min(e, hi)
            if b0 > a0:
                v = data[pos:pos + 2 * cnt].view("<u2")
                buf[a0 - lo:b0 - lo, k] = v[a0 - s:b0 - s]
            pos += 2 * cnt
        pos = end


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="binary to write (uint16, samples x channels)")
    ap.add_argument("--files", nargs="+", required=True,
                    help="the .brw conditions, IN ACQUISITION ORDER")
    ap.add_argument("--block", type=int, required=True, help="channel block index")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--fill", choices=["noise", "zeros"], default="noise")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    with h5py.File(a.files[0], "r") as h:
        n_chan = int(np.asarray(h[f"{WELL}/StoredChIdxs"]).size)
        fs = float(h.attrs["SamplingRate"])
    per = int(np.ceil(n_chan / a.n_blocks))
    chans = np.arange(a.block * per, min((a.block + 1) * per, n_chan))
    print(f"block {a.block}: channels {chans[0]}-{chans[-1]} ({chans.size}), fill={a.fill}")

    total, rows = 0, []
    with open(a.out, "wb") as fh:
        for p in a.files:
            n, nc = convert(p, chans, fh, a.seed, a.fill)
            print(f"  {Path(p).name}: samples {total:,}..{total + n:,}")
            rows.append((Path(p).name, total, total + n))
            total += n
    print(f"wrote {a.out}: {total:,} samples x {chans.size} channels uint16 "
          f"({total * chans.size * 2 / 1e9:.1f} GB)")
    print(f"\nKilosort: n_chan_bin={chans.size}, data_dtype='uint16'")
    # Ranges from samples actually written, not the file's nominal duration
    csv = Path(a.out).with_suffix(".recording_data.csv")
    with open(csv, "w") as fh:
        fh.write("recording_name,duration,num_samples,sampling_freq,sample_start,sample_end\n")
        for nm, s0, s1 in rows:
            fh.write(f"{nm},{(s1 - s0) / fs:.7f},{s1 - s0},{fs},{s0},{s1}\n")
    print(f"wrote {csv} -- use THIS, not neo's sample counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
