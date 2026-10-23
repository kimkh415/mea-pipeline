"""Preprocessor that fills 3Brain noise-blanked samples with 2048 + N(0, NoiseStdDev[ch]).

Draws are seeded per (recording, channel, TOC block): re-reads and chunking give the same values.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

BASELINE = 2048.0
DEFAULT_SD = 4.0
WELL = "Well_A1"


def _key(*parts):
    raw = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "big")


class GapIndex:
    """Stored-stretch positions, parsed per TOC block on demand and cached."""

    CACHE = 8

    def __init__(self, path):
        import h5py
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "brw"))
        from read_sparse import parse_block                   # noqa: E402
        self._parse = parse_block

        self.path = str(path)
        self._h = h5py.File(self.path, "r")
        g = self._h[WELL]
        self._raw = g["EventsBasedSparseRaw"]
        self._toc = np.asarray(g["EventsBasedSparseRawTOC"])
        self.n_chan = int(np.asarray(g["StoredChIdxs"]).size)
        self.n_blocks = int(self._toc.size - 1)
        self._cache = {}
        self.block_samples = self._measure()
        self._sd = self._noise_table(g)

    # --- geometry ---
    def _measure(self):
        """Samples per TOC block, from the first two populated blocks."""
        firsts = []
        for i in range(self.n_blocks):
            ch, s, e = self.stretches(i)
            if s.size:
                firsts.append((i, int(s.min())))
            if len(firsts) >= 2:
                break
        if len(firsts) < 2:
            raise SystemExit(f"{self.path}: too few populated TOC blocks")
        (i0, s0), (i1, s1) = firsts
        size, rem = divmod(s1 - s0, i1 - i0)
        if rem or size <= 0:
            raise SystemExit(f"{self.path}: TOC blocks are not a uniform partition")
        return int(size)

    def _noise_table(self, g):
        """Per-channel noise SD per block, carried forward between estimates."""
        nt = np.asarray(g["NoiseTOC"])
        idx = np.asarray(g["NoiseChIdxs"])
        sdv = np.asarray(g["NoiseStdDev"])
        out, cur = [], np.full(self.n_chan, np.nan)
        for i in range(nt.size - 1):
            a, b = int(nt[i]), int(nt[i + 1])
            if b > a:
                ch = idx[a:b]
                ok = (ch >= 0) & (ch < self.n_chan)
                cur = cur.copy()
                cur[ch[ok]] = sdv[a:b][ok]
            out.append(cur)
        finite = [c[np.isfinite(c)] for c in out if np.isfinite(c).any()]
        med = float(np.median(np.concatenate(finite))) if finite else DEFAULT_SD
        return [np.where(np.isfinite(c), c, med) for c in out]

    def sd(self, block):
        t = self._sd
        return t[block] if block < len(t) else t[-1]

    # --- stretches ---
    def stretches(self, block):
        """(channels, starts, ends) for one TOC block, as int64 arrays."""
        hit = self._cache.get(block)
        if hit is not None:
            return hit
        a, b = int(self._toc[block]), int(self._toc[block + 1])
        st, ok = self._parse(np.asarray(self._raw[a:b]))
        if not ok:
            raise SystemExit(f"{self.path}: TOC block {block} did not parse")
        arr = (np.array([c for c, _, _ in st], np.int64),
               np.array([s for _, s, _ in st], np.int64),
               np.array([e for _, _, e in st], np.int64))
        if len(self._cache) >= self.CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[block] = arr
        return arr

    def stored_ranges(self, block, channel):
        ch, s, e = self.stretches(block)
        m = ch == channel
        return s[m], e[m]


# Module-level classes: spikeinterface rebuilds them by name in workers and on load
# (a factory-built class hangs n_jobs > 1 on spawn platforms)
from spikeinterface.preprocessing.basepreprocessor import (   # noqa: E402
    BasePreprocessor, BasePreprocessorSegment)


class BrwGapFillSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, owner):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.owner = owner

    def get_traces(self, start_frame, end_frame, channel_indices):
        o = self.owner
        tr = np.array(self.parent_recording_segment.get_traces(
            start_frame, end_frame, channel_indices), copy=True)
        n = tr.shape[0]
        if n == 0:
            return tr
        idx, size = o.gap_index, o.gap_index.block_samples
        chans = np.arange(o.get_num_channels())[channel_indices]
        b0 = start_frame // size
        b1 = min((end_frame - 1) // size, idx.n_blocks - 1)
        for bi in range(b0, b1 + 1):
            lo, hi = max(bi * size, start_frame), min((bi + 1) * size, end_frame)
            if hi <= lo:
                continue
            sd = idx.sd(bi)
            for k, ch in enumerate(chans):
                ch = int(ch)
                # draw the whole block, then slice, so chunking doesn't matter
                r = np.random.default_rng([o.seed, _key(o.label), ch, bi])
                block = np.clip(
                    np.rint(BASELINE + r.normal(0.0, float(sd[ch]), size)),
                    0, 4095).astype(tr.dtype)
                seg = block[lo - bi * size:hi - bi * size].copy()
                s_, e_ = idx.stored_ranges(bi, ch)
                for a_, b_ in zip(s_, e_):
                    x0, x1 = max(int(a_), lo), min(int(b_), hi)
                    if x1 > x0:
                        seg[x0 - lo:x1 - lo] = tr[x0 - start_frame:x1 - start_frame, k]
                tr[lo - start_frame:hi - start_frame, k] = seg
        return tr


class BrwGapFillRecording(BasePreprocessor):
    """`read_biocam` with blanked samples filled from the file's own noise.

    Takes the path, not a recording: the Biocam extractor drops `fill_gaps_strategy` from `_kwargs`, so it can't be rebuilt.
    """
    def __init__(self, brw_path, seed=0):
        from spikeinterface.extractors import read_biocam
        parent = read_biocam(str(brw_path), fill_gaps_strategy="zeros")
        BasePreprocessor.__init__(self, parent)
        self.gap_index = GapIndex(brw_path)
        self.seed = int(seed)
        self.label = Path(brw_path).name
        self.unfilled_tail_samples = int(
            parent.get_num_samples()
            - self.gap_index.n_blocks * self.gap_index.block_samples)
        for seg in parent._recording_segments:
            self.add_recording_segment(BrwGapFillSegment(seg, self))
        self._kwargs = dict(brw_path=str(brw_path), seed=int(seed))


def open_brw_with_noise_fill(brw_path, seed=0):
    """Read a 3Brain `.brw` with blanked gaps noise-filled.

    The trailing partial TOC block is left unfilled (see `unfilled_tail_samples`).
    """
    return BrwGapFillRecording(brw_path, seed=seed)
