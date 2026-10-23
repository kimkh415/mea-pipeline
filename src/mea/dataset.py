"""Condition labels and roles, resolved from the `dataset:` config."""
from __future__ import annotations
import numpy as np, warnings, difflib


# --- conditions ---
def match_condition(name, cfg):
    """Fuzzy-match a condition name to a label -> (label, score); keeps the name below match_ratio."""
    d = cfg["dataset"]["conditions"]
    labels = d.get("labels") or []
    if not labels:
        return name, None
    low = str(name).lower()
    best, score = None, 0.0
    for lab in labels:
        s = max(
            difflib.SequenceMatcher(None, low, lab.lower()).ratio(),
            1.0 if lab.lower() in low else 0.0,      # substring is a full match
        )
        if s > score:
            best, score = lab, s
    return (best, score) if score >= d["match_ratio"] else (name, score)


def label_conditions(names, cfg, order=None):
    """Label conditions; treatments by name, baseline = first condition matching no treatment."""
    d = cfg["dataset"]["conditions"]
    base = d["roles"].get("baseline")
    treatments = [l for l in (d.get("labels") or []) if l != base]
    sub = dict(cfg)
    sub["dataset"] = {**cfg["dataset"],
                      "conditions": {**d, "labels": treatments}}
    out = []
    for i, n in enumerate(names):
        lab, _ = match_condition(n, sub)
        out.append(lab)
    unmatched = [i for i, (l, n) in enumerate(zip(out, names)) if l == n]
    if base and unmatched:
        # earliest unmatched condition is the baseline; any others keep their name
        out[unmatched[0]] = base
        unmatched = unmatched[1:]
    if unmatched:
        # keep the filename rather than guess, but warn
        warnings.warn(
            "these conditions match no label in dataset.conditions.labels and "
            "keep their filename, so they will not group with the same drug in "
            f"other recordings: {[names[i] for i in unmatched]}. Fix the "
            "filename, add the spelling to labels, or lower "
            f"dataset.conditions.match_ratio (currently {d['match_ratio']}).")
    return out


def role_mask(labels, cfg, role):
    """Mask of conditions filling a role ('silencing', 'baseline'); warns if none match."""
    want = cfg["dataset"]["conditions"]["roles"].get(role)
    labels = list(labels)
    if not want:
        return np.zeros(len(labels), bool)
    mask = np.array([str(l).lower() == str(want).lower() for l in labels])
    if not mask.any():
        warnings.warn(
            f"dataset.conditions.roles.{role} = {want!r} matches none of "
            f"{labels}. Any step depending on it will be skipped, not guessed.")
    return mask
