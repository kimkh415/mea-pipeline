"""Reader for 3Brain BRW Level-4 sparse files (only above-noise stretches are stored).

Chunk layout (offsets in EventsBasedSparseRawTOC), blocks repeated:

    int32  chIdx
    int32  size                 payload bytes
    payload, repeated until `size` is consumed:  # a block may hold several ranges
        int64  fromFrame        inclusive
        int64  toFrame          exclusive
        int16  data[toFrame - fromFrame]

Digital zero is 2048; ExperimentSettings.ValueConverter scales to uV.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field

import numpy as np

DIGITAL_ZERO = 2048
_HDR = np.dtype([("ch", "<i4"), ("size", "<i4")])


@dataclass
class BrwInfo:
    """Recording header fields."""
    path: str
    well: str
    n_channels: int
    sampling_hz: float
    n_frames: int
    duration_s: float
    acquired: datetime.datetime | None
    model: str | None
    biological_model: str | None
    highpass_hz: float | None
    std_factors: tuple | None
    scale: dict = field(default_factory=dict)

    def to_uv(self, digital):
        """Digital counts -> microvolts, per the file's own ValueConverter."""
        s = self.scale
        lo, hi = s.get("MinAnalogValue", -8000.0), s.get("MaxAnalogValue", 8000.0)
        dlo, dhi = s.get("MinDigitalValue", 0.0), s.get("MaxDigitalValue", 4095.0)
        span = (dhi - dlo) or 1.0
        return (np.asarray(digital, float) - dlo) * (hi - lo) / span + lo


def _settings(h):
    raw = h["ExperimentSettings"][0]
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _wells(h):
    return [k for k in h.keys() if k.startswith("Well_")]


def info(path, well=None):
    """Read the header only; no samples."""
    import h5py
    with h5py.File(str(path), "r") as h:
        d = _settings(h)
        w = well or (_wells(h) or ["Well_A1"])[0]
        toc = np.asarray(h["TOC"]) if "TOC" in h else None
        n_frames = int(toc[-1, 0]) if toc is not None and toc.size else 0
        fs = float(d.get("TimeConverter", {}).get("FrameRate")
                   or h.attrs.get("SamplingRate", 0.0))
        acq = d.get("ExperimentDateTime")
        if isinstance(acq, str):
            try:
                acq = datetime.datetime.fromisoformat(acq.replace("Z", "+00:00"))
            except ValueError:
                acq = None
        filt = d.get("SourceFilterSettings") or {}
        ev = (d.get("DataSettings") or {}).get("EventsBasedRawRanges") or {}
        stored = h.get(f"{w}/StoredChIdxs")
        return BrwInfo(
            path=str(path), well=w,
            n_channels=int(stored.shape[0]) if stored is not None else 0,
            sampling_hz=fs, n_frames=n_frames,
            duration_s=n_frames / fs if fs else 0.0,
            acquired=acq,
            model=(d.get("MeaPlate") or {}).get("Model"),
            biological_model=d.get("BiologicalModel"),
            highpass_hz=filt.get("HighPassFilterCutOffFrequency"),
            std_factors=(ev.get("LowerNegativeStdFactor"),
                         ev.get("HigherNegativeStdFactor")) if ev else None,
            scale=d.get("ValueConverter") or {},
        )


def iter_ranges(path, well=None, chunks=None, with_data=False):
    """Yield (chIdx, fromFrame, toFrame, data or None) for every stored range."""
    import h5py
    with h5py.File(str(path), "r") as h:
        w = well or (_wells(h) or ["Well_A1"])[0]
        raw = h[f"{w}/EventsBasedSparseRaw"]
        toc = np.asarray(h[f"{w}/EventsBasedSparseRawTOC"])
        n = raw.shape[0]
        idx = range(len(toc)) if chunks is None else chunks
        for c in idx:
            a = int(toc[c])
            b = int(toc[c + 1]) if c + 1 < len(toc) else n
            if b <= a:
                continue
            buf = np.asarray(raw[a:b]).tobytes()
            off, end_all = 0, len(buf)
            while off + 8 <= end_all:
                ch, size = np.frombuffer(buf, dtype=_HDR, count=1, offset=off)[0]
                p, end = off + 8, off + 8 + int(size)
                while p + 16 <= end:
                    fr, to = np.frombuffer(buf, dtype="<i8", count=2, offset=p)
                    cnt = int(to) - int(fr)
                    data = (np.frombuffer(buf, dtype="<i2", count=cnt, offset=p + 16)
                            if with_data and cnt > 0 else None)
                    yield int(ch), int(fr), int(to), data
                    p += 16 + 2 * cnt
                off = end


def n_chunks(path, well=None):
    import h5py
    with h5py.File(str(path), "r") as h:
        w = well or (_wells(h) or ["Well_A1"])[0]
        return int(np.asarray(h[f"{w}/EventsBasedSparseRawTOC"]).shape[0])


def sample_chunks(path, n=60, well=None):
    """`n` chunk indices spread evenly over this file's chunks."""
    total = n_chunks(path, well)
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n).astype(int).tolist()


def channel_activity(path, well=None, chunks=None):
    """Per-channel stored-range counts and rates, from range headers alone (no sorting).

    Returns dict with info, n_events, n_frames, seconds_scanned, event_rate_hz, occupancy.
    """
    nfo = info(path, well)
    n_ch = nfo.n_channels or 4096
    n_events = np.zeros(n_ch, np.int64)
    n_frames = np.zeros(n_ch, np.int64)
    for ch, fr, to, _ in iter_ranges(path, well, chunks, with_data=False):
        if 0 <= ch < n_ch:
            n_events[ch] += 1
            n_frames[ch] += to - fr
    # seconds actually scanned, so sampled scans give comparable rates
    total = n_chunks(path, well)
    scanned = total if chunks is None else len(chunks)
    seconds = nfo.duration_s * scanned / total if total else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"info": nfo, "n_events": n_events, "n_frames": n_frames,
                "seconds_scanned": seconds,
                "event_rate_hz": n_events / seconds if seconds else n_events * 0.0,
                # fraction of scanned time stored (above the recorder's noise band)
                "occupancy": (n_frames / (seconds * nfo.sampling_hz)
                              if seconds and nfo.sampling_hz else n_frames * 0.0)}
