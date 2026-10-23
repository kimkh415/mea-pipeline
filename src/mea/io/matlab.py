"""Readers for legacy MATLAB .mat files (v7 via scipy; v7.3/HDF5 not yet supported).

MATLAB structs are unpacked into flat arrays here.
"""
from __future__ import annotations
import numpy as np, scipy.io as sio, h5py
from pathlib import Path


def _v73(path) -> bool:
    with open(path, "rb") as f:
        return b"MATLAB 7.3" in f.read(128)


def load_mat(path, variables=None):
    path = Path(path)
    if _v73(path):
        out = {}
        with h5py.File(path, "r") as h:
            for k in (variables or list(h.keys())):
                if k in h:
                    out[k] = h[k]
        raise NotImplementedError("v7.3 streaming reader lands with stage 3")
    d = sio.loadmat(path, squeeze_me=False, struct_as_record=True)
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _field(rec, name):
    try:
        return np.ravel(rec[name])
    except (KeyError, ValueError, IndexError):
        return np.array([])


def unpack_spike_timestamp(st):
    """spikeTimestamp struct array (1 x n_units) -> flat arrays.

    Returns t_ms (sorted within unit), unit_idx (struct position), offsets (n_units+1), per-unit fields.
    """
    st = np.atleast_1d(np.squeeze(st))
    n = st.shape[0]
    trains, meta = [], {k: [] for k in
                        ("n_units", "rec_length_s", "t_step_ms", "cluster",
                         "pos_x", "pos_y", "win_t0_s", "win_t1_s")}
    names = st.dtype.names or ()
    for i in range(n):
        rec = st[i]
        t = np.ravel(np.asarray(_field(rec, "timeStamp"), dtype=float))
        trains.append(np.sort(t))
        meta["n_units"].append(float(_field(rec, "nUnits")[0]) if "nUnits" in names and _field(rec, "nUnits").size else t.size)
        rl = _field(rec, "recLength"); meta["rec_length_s"].append(float(rl[0]) if rl.size else np.nan)
        ts = _field(rec, "t_step") if "t_step" in names else np.array([])
        meta["t_step_ms"].append(float(ts[0]) if ts.size else np.nan)
        cl = _field(rec, "cluster") if "cluster" in names else np.array([])
        meta["cluster"].append(float(cl[0]) if cl.size else i)
        xy = _field(rec, "positionXY") if "positionXY" in names else np.array([])
        meta["pos_x"].append(float(xy[0]) if xy.size >= 1 else np.nan)
        meta["pos_y"].append(float(xy[1]) if xy.size >= 2 else np.nan)
        w = _field(rec, "window") if "window" in names else np.array([])
        meta["win_t0_s"].append(float(w[0]) if w.size >= 1 else np.nan)
        meta["win_t1_s"].append(float(w[1]) if w.size >= 2 else np.nan)

    sizes = np.array([t.size for t in trains], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64); offsets[1:] = np.cumsum(sizes)
    t_ms = np.concatenate(trains) if n else np.array([])
    unit_idx = np.repeat(np.arange(n, dtype=np.int32), sizes)
    return {"t_ms": t_ms, "unit_idx": unit_idx, "offsets": offsets,
            "n_units_struct": n, **{k: np.array(v) for k, v in meta.items()}}


def _as_pair(w):
    w = np.ravel(np.asarray(w, float))
    return (float(w[0]), float(w[1])) if w.size >= 2 else None


def _first_window(obj, depth=0):
    """First `.window` inside a nested cell / struct array, as (t0, t1)."""
    if depth > 5 or obj is None:
        return None

    w = getattr(obj, "window", None)                 # mat_struct style
    if w is not None:
        return _as_pair(w)

    arr = np.asarray(obj)
    names = getattr(arr.dtype, "names", None)
    if names and "window" in names:                  # structured array
        flat = arr.ravel()
        for k in range(flat.size):
            got = _as_pair(flat[k]["window"])
            if got is not None:
                return got
        return None

    if arr.dtype == object:
        for e in np.atleast_1d(arr).ravel():
            got = _first_window(e, depth + 1)
            if got is not None:
                return got
    return None


def windows_from_indiv_cond(indiv_cond, n_cond=None):
    """Condition windows in seconds, from each Indiv_Cond `.window`.

    The condition axis is found by length: (n_cond, n_field) in raw files, (1, n_cond) in cleaned.
    """
    if indiv_cond is None:
        return []
    ic = np.asarray(indiv_cond, dtype=object)
    if ic.ndim >= 2:
        if n_cond is not None and ic.shape[1] == n_cond and ic.shape[0] != n_cond:
            ic = ic[0, :]                  # conditions along axis 1
        else:
            ic = ic[:, 0]                  # conditions along axis 0
    return [(_first_window(c) or (np.nan, np.nan)) for c in np.atleast_1d(ic)]


def unpack_rec_conditions(rc, indiv_cond=None):
    """rec_conditions -> list of (name, t0_s, t1_s); missing windows fall back to Indiv_Cond."""
    rc = np.atleast_2d(rc)
    out = []
    for row in rc:
        cells = [np.ravel(c) for c in np.atleast_1d(row)]
        name = str(cells[0][0]) if cells[0].size else ""
        t0 = t1 = np.nan
        if len(cells) > 1 and cells[1].size >= 2:
            t0, t1 = float(cells[1][0]), float(cells[1][1])
        out.append((name, t0, t1))

    if indiv_cond is not None and any(not np.isfinite(t) for _, t, _ in out):
        fallback = windows_from_indiv_cond(indiv_cond, n_cond=len(out))
        for i, (a, b) in enumerate(fallback[:len(out)]):
            if not np.isfinite(out[i][1]) and np.isfinite(a):
                out[i] = (out[i][0], a, b)
    return out
