"""Config loading: packaged defaults, user overrides, dotted --set overrides."""
from __future__ import annotations
import warnings
import yaml, copy, hashlib, json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _deep_update(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load(user=None, overrides=None):
    cfg = yaml.safe_load(DEFAULT.read_text())
    if user:
        _deep_update(cfg, yaml.safe_load(Path(user).read_text()))
    for item in (overrides or []):                     # "network_burst.threshold_sd=4.0"
        key, _, val = item.partition("=")
        node = cfg
        *parts, last = key.split(".")
        for p in parts:
            node = node.setdefault(p, {})
        try:
            node[last] = yaml.safe_load(val)
        except Exception:
            node[last] = val
    _check(cfg)
    return cfg


# Retired qc_general keys -> replacement hint; rejected so stale configs fail loudly
_RETIRED = {
    "ttx_test": "the ratio/ratio_baseline/poisson choice is gone; there is one "
                "rule, `ttx_min_drop` against the peak rate",
    "ttx_effect": "use ttx_min_drop = 1 - ttx_effect  (0.10 -> 0.90)",
    "ttx_fdr_q": "the binomial test and its Benjamini-Hochberg cut are gone",
}


def _check(cfg):
    """Reject configs that ask for retired behaviour; warn on a lenient TTX bar."""
    g = cfg.get("qc_general") or {}
    stale = [k for k in _RETIRED if k in g]
    if stale:
        raise ValueError(
            "qc_general sets retired key(s) "
            + ", ".join(sorted(stale)) + ":\n  "
            + "\n  ".join(f"{k}: {_RETIRED[k]}" for k in sorted(stale)))

    drop = g.get("ttx_min_drop")
    if drop is None:
        return
    drop = float(drop)
    if not 0.0 < drop < 1.0:
        raise ValueError(f"qc_general.ttx_min_drop must be between 0 and 1, "
                         f"got {drop}")
    if drop < 0.90:
        warnings.warn(
            f"qc_general.ttx_min_drop={drop:g} is below the 0.90 the field "
            f"uses. TTX blocks sodium channels, so a neuron stops firing; a "
            f"lower bar keeps units the drug did not silence, and they will "
            f"be counted as neurons downstream.",
            stacklevel=2)


def subtree_hash(cfg, *sections):
    """Hash only the config sections a stage depends on."""
    sub = {s: cfg.get(s) for s in sections}
    blob = json.dumps(sub, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
