"""Rule-based baseline. Works only from *observable* features (no ground-truth stage).

Compromise stage is inferred from the smoothed anomaly score s = (traffic+user)/2:
    s >= t_breach  -> BREACHED est.   -> restore_backup
    s >= t_esc     -> ESCALATED est.  -> isolate_machine (then restore_backup once isolated)
    s >= t_early   -> EARLY est.      -> block_ip
    unpatched and vuln >= t_vuln      -> patch_system
    otherwise                         -> no_op
One action per step on the highest-priority host.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from env.core import (BLOCK_IP, FEATS_PER_HOST, ISOLATE, N_TYPES, NO_OP, PATCH, RESTORE)


@dataclass
class HeuristicParams:
    t_early: float = 0.32
    t_esc: float = 0.48
    t_breach: float = 0.62
    t_vuln: float = 0.5


_TUNED = os.path.join(os.path.dirname(__file__), "heuristic_tuned.json")


def tuned_params() -> HeuristicParams:
    """Thresholds found by evaluation/tune_heuristic.py (grid search on tuning seeds only)."""
    if os.path.exists(_TUNED):
        return HeuristicParams(**json.load(open(_TUNED)))
    return HeuristicParams()


class HeuristicAgent:
    def __init__(self, params: HeuristicParams | None = None):
        self.p = params or tuned_params()

    def host_choice(self, feats: np.ndarray):
        """feats: (k_hosts, FEATS_PER_HOST). Returns (priority, host_idx, action)."""
        p = self.p
        best = (0.0, 0, NO_OP)
        for h, f in enumerate(feats):
            s = 0.5 * (f[N_TYPES] + f[N_TYPES + 1])
            vuln, patched, isolated = f[N_TYPES + 2], f[N_TYPES + 3] > 0.5, f[N_TYPES + 4] > 0.5
            if s >= p.t_breach:
                cand = (4.0 + s, h, RESTORE)
            elif s >= p.t_esc:
                cand = (3.0 + s, h, RESTORE if isolated else ISOLATE)
            elif s >= p.t_early and not isolated:
                cand = (2.0 + s, h, BLOCK_IP)
            elif not patched and vuln >= p.t_vuln:
                cand = (1.0 + vuln, h, PATCH)
            else:
                cand = (0.0, h, NO_OP)
            if cand[0] > best[0]:
                best = cand
        return best

    # single-agent flat observation -> flat action index
    def predict(self, obs, deterministic=True):
        feats = np.asarray(obs)[:-1].reshape(-1, FEATS_PER_HOST)
        _, h, a = self.host_choice(feats)
        return h * 7 + a
