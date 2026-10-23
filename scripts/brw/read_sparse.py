"""Decode 3Brain BRW v4 EventsBasedSparseRaw.

Layout per TOC block, little-endian:
    int32  channel index
    int32  byte length of this channel's payload
      repeated until payload consumed:
        int64 fromInclusive   (sample index)
        int64 toExclusive
        uint16 * (to-from)    samples
The parse is self-validating: every block must end exactly on its boundary.
"""
import numpy as np, h5py

def blocks(path, well="Well_A1"):
    h = h5py.File(path, "r")
    raw = h[f"{well}/EventsBasedSparseRaw"]
    toc = np.asarray(h[f"{well}/EventsBasedSparseRawTOC"])
    return h, raw, toc

def parse_block(buf):
    """-> list of (channel, from, to) and a flag saying the parse closed cleanly."""
    out, pos, n = [], 0, buf.size
    while pos + 8 <= n:
        ch = int(buf[pos:pos+4].view("<i4")[0]); pos += 4
        ln = int(buf[pos:pos+4].view("<i4")[0]); pos += 4
        if ln < 0 or pos + ln > n:
            return out, False
        end = pos + ln
        while pos < end:
            if pos + 16 > end: return out, False
            a = int(buf[pos:pos+8].view("<i8")[0]); pos += 8
            b = int(buf[pos:pos+8].view("<i8")[0]); pos += 8
            k = b - a
            if k <= 0 or pos + 2*k > end: return out, False
            out.append((ch, a, b))
            pos += 2*k
        pos = end
    return out, pos == n
