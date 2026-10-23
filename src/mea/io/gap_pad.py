"""Optional noise padding for the recorder-off interval between conditions (off by default).

Uses the same 2048 + N(0, NoiseStdDev) fill as brw_fill; constant padding would skew whitening.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .brw_fill import BASELINE, GapIndex, _key


class GapPadSegment:
    """Fill-only traces; never reads the parent."""

    def __init__(self, owner, n_samples):
        self.owner = owner
        self.n_samples = int(n_samples)

    def get_num_samples(self):
        return self.n_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        o = self.owner
        start = 0 if start_frame is None else int(start_frame)
        end = self.n_samples if end_frame is None else int(end_frame)
        start, end = max(start, 0), min(end, self.n_samples)
        n = max(end - start, 0)
        chans = np.arange(o.get_num_channels())[channel_indices]
        out = np.empty((n, len(chans)), dtype=o.get_dtype())
        if n == 0:
            return out

        # fixed-size blocks then slice, so chunking doesn't matter
        size = o.block_samples
        for k, ch in enumerate(chans):
            ch = int(ch)
            sd = float(o.noise_sd[ch])
            for bi in range(start // size, (end - 1) // size + 1):
                lo, hi = max(bi * size, start), min((bi + 1) * size, end)
                if hi <= lo:
                    continue
                r = np.random.default_rng([o.seed, _key(o.label), ch, bi])
                blk = np.clip(np.rint(BASELINE + r.normal(0.0, sd, size)),
                              0, 4095).astype(o.get_dtype())
                out[lo - start:hi - start, k] = blk[lo - bi * size:hi - bi * size]
        return out


def _base_classes():
    from spikeinterface.core import BaseRecording, BaseRecordingSegment
    return BaseRecording, BaseRecordingSegment


class GapPadRecording:
    """Name only; instances come from :func:`make_gap_pad`."""


def make_gap_pad(template, brw_path, n_samples, seed=0, label=None):
    """`n_samples` of noise fill shaped like `template` (the condition before the gap)."""
    BaseRecording, BaseRecordingSegment = _base_classes()

    class _Segment(GapPadSegment, BaseRecordingSegment):
        def __init__(self, owner, n):
            BaseRecordingSegment.__init__(
                self, sampling_frequency=owner.get_sampling_frequency())
            GapPadSegment.__init__(self, owner, n)

    class _Recording(BaseRecording):
        def __init__(self, brw_path, n_samples, seed, label):
            BaseRecording.__init__(
                self,
                sampling_frequency=template.get_sampling_frequency(),
                channel_ids=template.channel_ids,
                dtype=template.get_dtype())
            idx = GapIndex(brw_path)
            self.block_samples = int(idx.block_samples)
            # last block's SD per channel; none exists while off
            self.noise_sd = np.asarray(idx.sd(idx.n_blocks - 1), dtype=float)
            self.seed = int(seed)
            self.label = str(label or f"pad:{Path(brw_path).name}:{n_samples}")
            self.add_recording_segment(_Segment(self, n_samples))
            probe = template.get_probe() if template.has_probe() else None
            if probe is not None:
                self.set_probe(probe, in_place=True)
            self._kwargs = dict(brw_path=str(brw_path), n_samples=int(n_samples),
                                seed=int(seed), label=self.label)

    rec = _Recording(str(brw_path), int(n_samples), int(seed), label)
    return rec


def gap_samples(paths, recordings, fs):
    """Recorder-off samples before each condition ([0] = 0); None if any file lacks a timestamp."""
    from ..conditions import acquisition_time

    times = [acquisition_time(p) for p in paths]
    if any(t is None for t in times):
        return None
    out = [0]
    for i in range(1, len(paths)):
        prev_end = times[i - 1].timestamp() + recordings[i - 1].get_num_samples() / fs
        out.append(max(int(round((times[i].timestamp() - prev_end) * fs)), 0))
    return out


def restore_gaps(paths, recordings, seed=0):
    """Interleave pads between conditions -> (sequence, gap_samples).

    Non-positive intervals (overlapping clocks) get no pad.
    """
    fs = float(recordings[0].get_sampling_frequency())
    gaps = gap_samples(paths, recordings, fs)
    if gaps is None:
        return None, None
    seq = []
    for i, (p, rec) in enumerate(zip(paths, recordings)):
        if gaps[i] > 0:
            seq.append(make_gap_pad(recordings[i - 1], paths[i - 1], gaps[i],
                                    seed=seed, label=f"pad{i}:{paths[i].name}"))
        seq.append(rec)
    return seq, gaps
