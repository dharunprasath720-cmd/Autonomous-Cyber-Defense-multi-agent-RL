"""Single-agent Gymnasium environment: ``CyberDefenseEnv``.

The agent picks one (host, action) pair per step via a flat Discrete(n_hosts * 7) space.
Observation: per-host [type one-hot, traffic_anom, user_anom, vuln, patched, isolated, honeypot]
plus one global threat scalar. The true compromise stage is hidden (POMDP).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import FEATS_PER_HOST, N_ACTIONS, NetworkSim, SimConfig


class CyberDefenseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_hosts: int = 8, horizon: int = 100, cfg: SimConfig | None = None, **kw):
        super().__init__()
        self.cfg = cfg or SimConfig(n_hosts=n_hosts, horizon=horizon, **kw)
        self.sim = NetworkSim(self.cfg)
        self.n_hosts = self.cfg.n_hosts
        self.action_space = spaces.Discrete(self.n_hosts * N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, (self.n_hosts * FEATS_PER_HOST + 1,), np.float32)

    def decode(self, a: int):
        return divmod(int(a), N_ACTIONS)  # (host, action)

    def _obs(self):
        return np.concatenate([self.sim.host_features().ravel(),
                               [self.sim.threat_obs]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed)
        return self._obs(), {}

    def step(self, action):
        rz = self.sim.step([self.decode(action)])
        truncated = self.sim.t >= self.cfg.horizon
        info = {}
        if truncated:
            info["metrics"] = self.sim.episode_metrics()
        return self._obs(), float(rz.sum()), False, truncated, info
