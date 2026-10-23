"""Per-family network-burst metrics (port of NetworkBurstFeatures.m).

Spike lists come from padded windows; metrics use the unpadded event.
"""
from __future__ import annotations
import numpy as np

from ..core import matlab_semantics as ml
from ..core.matlab_semantics import sec_to_ms

BIN_REF = 0.1
K_SEC_MIN = 1.0 / 60.0


def _cv(x):
    x = np.asarray(x, float)
    m = ml.mean(x)
    return ml.std(x) / m if m else np.nan


def features(bursts, event_times, participants, member_mask, n_units_total,
             n_spikes_per_unit, n_channels_bursting, bin_s, opt_secondary=False,
             n_primary=None):
    """Metrics for the events selected by `member_mask`.

    bursts        : list of event dicts (peak / start / last / peakAmp, sec)
    event_times   : event_times[unit][event] -> spike times in ms (padded window)
    participants  : 0-based unit indices belonging to this family
    member_mask   : which events belong to this family
    """
    member = np.asarray(member_mask, bool)
    n_burst = int(member.sum())
    n_total = len(bursts)
    n_part = len(participants)
    out = {"nBurstTotal": n_total, "nBurst": n_burst,
           "nNeuron": {"totalRec": n_units_total,
                       "totalBurst": n_channels_bursting,
                       "Burst": n_part,
                       "normRec": n_channels_bursting / n_units_total if n_units_total else np.nan,
                       "normBurst": n_part / n_channels_bursting if n_channels_bursting else np.nan}}

    if not (n_total > 0 and n_burst > 0 and n_part > 0):
        for k in ("rate", "amplitude", "duration", "nSpikes", "neuronsAct",
                  "distance", "delay", "height"):
            out[k] = {"mean": np.nan, "cv": np.nan}
        out["indivMetric"] = []
        return out

    idx_event = np.flatnonzero(member)
    delay = np.full((n_burst, n_part), np.nan)
    isi_b = np.full(n_burst, np.nan)
    peak_b = np.full(n_burst, np.nan)
    dur_b = np.full(n_burst, np.nan)
    spk_b = np.full(n_burst, np.nan)
    neu_b = np.full(n_burst, np.nan)
    dist_b = np.full(n_burst, np.nan)
    hgt_b = np.full(n_burst, np.nan)
    indiv = []

    for k, idx in enumerate(idx_event):
        b = bursts[idx]
        lo_ms, hi_ms = sec_to_ms(b["start"]), sec_to_ms(b["last"])
        peak_ms = sec_to_ms(b["peak"])
        peak_b[k] = b["peak"]

        n_spk = np.zeros(n_part)
        dist = np.full(n_part, np.nan)
        isi = np.full(n_part, np.nan)
        n_active = 0
        for j, ch in enumerate(participants):
            t = np.asarray(event_times[ch][idx], float)
            t = t[(t > lo_ms) & (t < hi_ms)]         # strict, as MATLAB
            n_spk[j] = t.size
            if t.size > 1:
                isi[j] = float(np.median(np.diff(t)))
            if t.size > 0:
                dist[j] = peak_ms - float(np.median(t))
                delay[k, j] = lo_ms - t[0]
                n_active += 1

        d = {"nSpikes": ml.mean(n_spk),
             "duration": b["last"] - b["start"],
             "distance": ml.mean(np.abs(dist)),
             "isi": ml.mean(isi),
             "nNeuronsAct": n_active / n_part}
        if opt_secondary and "parent" in b:
            d["parent"] = b["parent"]
        indiv.append(d)

        isi_b[k], dur_b[k] = d["isi"], d["duration"]
        spk_b[k], neu_b[k] = d["nSpikes"], d["nNeuronsAct"]
        dist_b[k] = d["distance"]

        charge = float(np.nansum(n_spk))
        factor = bin_s / BIN_REF
        amp = b.get("peakAmp", np.nan)
        hgt_b[k] = (amp / factor / charge) / d["duration"] \
            if charge and d["duration"] else np.nan

    out["indivMetric"] = indiv
    spikes_part = float(sum(n_spikes_per_unit[c] for c in participants))
    spikes_all = float(np.sum(n_spikes_per_unit))

    if n_burst > 1:
        ibi = np.diff(peak_b)
        m = ml.mean(ibi)
        out["rate"] = {"mean": 1.0 / (m * K_SEC_MIN) if m else np.nan, "cv": _cv(ibi)}
    else:
        out["rate"] = {"mean": np.nan, "cv": np.nan}

    m_isi = ml.mean(isi_b)
    out["amplitude"] = {"mean": 1.0 / (m_isi / 1000.0) if m_isi else np.nan,
                        "cv": _cv(isi_b)}
    out["duration"] = {"mean": ml.mean(dur_b), "cv": _cv(dur_b)}
    mean_spk = ml.mean(spk_b)
    out["nSpikes"] = {
        "mean": mean_spk,
        "norm": mean_spk * n_burst / (spikes_part / n_part) if spikes_part else np.nan,
        "normTotal": mean_spk * n_burst * n_part / spikes_all if spikes_all else np.nan}
    out["neuronsAct"] = {"mean": ml.mean(neu_b), "cv": _cv(neu_b)}
    out["distance"] = {"mean": ml.mean(dist_b), "cv": _cv(dist_b)}
    per_event_delay = np.array([ml.mean(np.abs(delay[k])) for k in range(n_burst)])
    out["delay"] = {"mean": ml.mean(per_event_delay), "cv": _cv(per_event_delay),
                    "values": delay}
    out["height"] = {"mean": ml.mean(hgt_b), "cv": _cv(hgt_b)}

    if opt_secondary and n_primary:
        daughter = np.zeros(n_primary)
        for d in indiv:
            if "parent" in d:
                daughter[d["parent"]] += 1
        out["PrimarywithSecondary"] = float((daughter > 0).sum()) / n_primary
        out["SecondaryperPrimary"] = ml.mean(daughter)
    return out
