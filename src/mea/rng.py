"""Reproducible randomness: one root seed, independent streams per work item."""
from __future__ import annotations
import hashlib

import numpy as np


class Streams:
    def __init__(self, seed: int, label: str = ""):
        self.seed = int(seed)
        self.label = label
        self._root = np.random.SeedSequence(self.seed)

    def for_item(self, *key) -> np.random.Generator:
        """Generator bound to a work item, e.g. for_item('pair', i, j)."""
        # blake2b, not hash(): builtin str hashing is salted per process
        raw = "\x1f".join((self.label, *map(str, key))).encode()
        h = int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "big")
        return np.random.default_rng(np.random.SeedSequence([self.seed, h]))

    def spawn(self, n) -> list:
        return [np.random.default_rng(s) for s in self._root.spawn(n)]
