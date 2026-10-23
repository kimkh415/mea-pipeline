"""Group-by via a single sort."""
from __future__ import annotations
import numpy as np


def group_by_sorted(keys, n_keys=None):
    """Return (order, bounds) so group k is order[bounds[k]:bounds[k+1]].

    keys: non-negative ints. Stable sort, so within-group order is kept.
    """
    keys = np.asarray(keys)
    if n_keys is None:
        n_keys = int(keys.max()) + 1 if keys.size else 0
    order = np.argsort(keys, kind="stable")
    bounds = np.searchsorted(keys[order], np.arange(n_keys + 1))
    return order, bounds


def group_sizes(keys, n_keys=None):
    """Count per key."""
    keys = np.asarray(keys)
    if n_keys is None:
        n_keys = int(keys.max()) + 1 if keys.size else 0
    return np.bincount(keys, minlength=n_keys)
