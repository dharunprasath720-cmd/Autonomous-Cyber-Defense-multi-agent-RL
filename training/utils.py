"""Shared training utilities: seeds, plateau-based stopping, CSV logging."""
from __future__ import annotations

import csv
import os

import numpy as np

# Seed conventions (documented in README):
#   training episodes : 100_000 + 1_000_000 * run_seed + k   (< 10_000_000)
#   tuning episodes   : 5_000_000 + k                         (heuristic threshold search)
#   held-out eval     : 10_000_000 + k, k = 0..N-1            (never seen in training)
TRAIN_SEED_BASE = 100_000
TUNE_SEED_BASE = 5_000_000
EVAL_SEED_BASE = 10_000_000


def eval_seeds(n=50):
    return [EVAL_SEED_BASE + k for k in range(n)]


class PlateauStopper:
    """Stop when the smoothed return has not improved by > tol (relative) for `patience` updates.

    `min_steps` guards against stopping while the curve is still rising; `max_steps` is only a
    safety cap (reported in the logs when it triggers instead of a genuine plateau).
    """

    def __init__(self, window=10, patience=40, tol=0.01, min_steps=200_000, max_steps=3_000_000):
        self.window, self.patience, self.tol = window, patience, tol
        self.min_steps, self.max_steps = min_steps, max_steps
        self.hist: list[float] = []
        self.best = -np.inf
        self.since = 0
        self.reason = None

    def update(self, ep_return: float, steps: int) -> bool:
        self.hist.append(ep_return)
        if len(self.hist) < self.window:
            return False
        ma = float(np.mean(self.hist[-self.window:]))
        if ma > self.best + self.tol * abs(self.best if np.isfinite(self.best) else ma):
            self.best, self.since = ma, 0
        else:
            self.since += 1
        if steps >= self.max_steps:
            self.reason = f"max_steps cap ({self.max_steps}) reached"
            return True
        if steps >= self.min_steps and self.since >= self.patience:
            self.reason = f"plateau: no >{self.tol:.0%} improvement of {self.window}-update MA for {self.patience} updates"
            return True
        return False


class CSVLogger:
    def __init__(self, path, fields):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()

    def log(self, **row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
