"""Stage 7 -- compile.  Every recording's metrics into one table.

Stage-6 metrics.parquet from every store -> compiled.wide.csv (one row per recording x condition).
Groups come from an optional sample sheet; thin network coverage is flagged, not dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = "quantify.metrics.parquet"
MIN_NETBURST = 3            # fewer network bursts -> few_events


def find_stores(root):
    """Every `<recording>.mea` store under `root` that stage 6 has written."""
    root = Path(root)
    return sorted(p for p in root.rglob("*.mea") if (p / "artifacts" / METRICS).exists())


def read_sample_sheet(path):
    """Optional per-recording metadata CSV, indexed by its `recording` column."""
    if path is None:
        return None
    df = pd.read_csv(path)
    if "recording" not in df.columns:
        raise ValueError(
            f"{path} needs a `recording` column naming the .mea store "
            f"(found {list(df.columns)})")
    return df.set_index("recording")


def _provenance(store):
    meta = {}
    f = store / "meta.json"
    if f.exists():
        try:
            meta = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return {"source": meta.get("source"),
            "n_cluster_original": meta.get("n_cluster_original"),
            "entered_at": meta.get("produced_by")}


def _network_burst_counts(long):
    """nBurst per (recording, condition), for the coverage rule."""
    nb = long[(long["level"] == "C_network") & (long["metric"] == "nBurst")]
    return nb.set_index(["recording", "condition"])["value"].to_dict()


def collect(root, cfg):
    """Long table of all stores: recording, condition, treatment, level, metric, value."""
    from ..dataset import label_conditions

    frames = []
    for store in find_stores(root):
        rec = store.name[:-4]                       # strip .mea
        df = pd.read_parquet(store / "artifacts" / METRICS)
        df.insert(0, "recording", rec)

        # Label conditions with the shared treatment list
        conds = list(dict.fromkeys(df["condition"]))
        labels = dict(zip(conds, label_conditions(conds, cfg)))
        df.insert(2, "treatment", df["condition"].map(labels))

        for k, v in _provenance(store).items():
            df[k] = v
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["recording", "condition", "treatment",
                                     "level", "metric", "value"])
    long = pd.concat(frames, ignore_index=True)

    # Level-C coverage: NaN nBurst = not network bursting; small nBurst = few events
    counts = _network_burst_counts(long)
    n_burst = np.array([counts.get((r, c), np.nan) for r, c in
                        zip(long["recording"], long["condition"])], float)
    is_c = long["level"].str.startswith("C_").to_numpy()
    long["n_network_burst"] = n_burst
    long["not_network_bursting"] = is_c & ~np.isfinite(n_burst)
    long["few_events"] = is_c & np.isfinite(n_burst) & (n_burst < MIN_NETBURST)
    # single filter column for either reason
    long["thin"] = long["not_network_bursting"] | long["few_events"]
    return long


# Level letters dropped; each metric name is unique on its own:
#   unit     population.entropy  -> entropy
#   burst    burst.rate_mean     -> burst.rate_mean   (already qualified)
#   network  nBurst              -> network.nBurst
#   families quality             -> families.quality
#   family1  duration.mean       -> family1.duration.mean
LEVEL_ORDER = {"unit": 0, "burst": 1, "network": 2, "families": 3}


def public_names(long):
    """Drop level letters and give each metric a unique public name."""
    out = long.copy()
    level = out["level"].str.replace(r"^[A-Z]_", "", regex=True)
    metric = out["metric"]
    metric = metric.where(level != "unit", metric.str.replace(r"^population\.", "", regex=True))
    metric = metric.where(level.isin(["unit", "burst"]), level + "." + metric)
    out["level"], out["metric"] = level, metric
    clash = out.groupby("metric")["level"].nunique()
    if (clash > 1).any():
        raise ValueError(f"metric names shared by two levels: {list(clash[clash > 1].index)}")
    return out


# Leading columns of the wide table
INDEX_COLS = ["recording", "condition", "treatment",
              "t_start_s", "t_end_s", "n_units"]


def wide(long):
    """One row per recording and condition, every metric as a column (see README Metrics)."""
    if long.empty:
        return pd.DataFrame()
    idx = [c for c in INDEX_COLS if c in long.columns]
    out = (long.pivot_table(index=idx, columns="metric", values="value",
                            aggfunc="first")
               .reset_index())
    out.columns.name = None
    # Order columns by level, then family number, then name
    level_of = dict(zip(long["metric"], long["level"]))
    metrics = [c for c in out.columns if c not in idx]
    def key(c):                   # family2 before family10
        lvl = level_of[c]
        n = lvl[len("family"):]
        return (LEVEL_ORDER.get(lvl, 4), int(n) if n.isdigit() else 0, c)
    metrics.sort(key=key)
    # conditions in acquisition order
    rows = [c for c in ("recording", "t_start_s") if c in idx]
    if rows:
        out = out.sort_values(rows, kind="stable").reset_index(drop=True)
    return out[idx + metrics]


def run(root, cfg, out_dir, sample_sheet=None):
    """Write compiled.wide.csv and return counts.

    Sample-sheet columns are inserted right after `recording`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long = collect(root, cfg)
    if long.empty:
        return {"n_recordings": 0, "written": [], "note": "no stage-6 output found"}
    long = public_names(long)

    w = wide(long)
    sheet = read_sample_sheet(sample_sheet)
    if sheet is not None:
        missing = sorted(set(w["recording"]) - set(sheet.index))
        extra = [c for c in sheet.columns if c not in w.columns]
        w = w.merge(sheet[extra], left_on="recording", right_index=True, how="left")
        w = w[["recording"] + extra + [c for c in w.columns if c not in extra and c != "recording"]]
    q = out_dir / "compiled.wide.csv"
    w.to_csv(q, index=False)

    res = {"n_recordings": int(long["recording"].nunique()),
           "n_conditions": int(long.groupby("recording")["condition"].nunique().sum()),
           "n_metrics": int(long["metric"].nunique()),
           "n_not_bursting": int(long.loc[long["not_network_bursting"], ["recording", "condition"]].drop_duplicates().shape[0]),
           "n_few_events": int(long.loc[long["few_events"], ["recording", "condition"]].drop_duplicates().shape[0]),
           "written": [str(q)]}
    if sheet is not None:
        res["not_in_sheet"] = missing
    return res
