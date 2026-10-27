"""Multi-agent PettingZoo ParallelEnv: ``MultiCyberDefenseEnv`` (one agent per network zone).

Zones: 0 = DMZ (servers/cloud), 1 = internal workstation zone, 2 = data tier (databases).
Each agent sees ONLY its own zone's host features (padded to the largest zone) plus a
shared summary [global threat level, cross-zone alert count]. Cross-zone dynamics live in
``NetworkSim`` (a compromised, un-isolated host with stage>=ESCALATED in zone z raises
intrusion probability in zone z+1 on the next step).

``full_obs=True`` (ablation) gives every agent the whole network instead.
``state()`` returns a privileged global state (true stages included) for CTDE critics.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .core import FEATS_PER_HOST, N_ACTIONS, NO_OP, NetworkSim, SimConfig

ALERT_THRESH = 0.45


class MultiCyberDefenseEnv(ParallelEnv):
    metadata = {"name": "multi_cyber_defense_v0"}

    def __init__(self, cfg: SimConfig | None = None, full_obs: bool = False, **kw):
        self.cfg = cfg or SimConfig(**kw)
        self.sim = NetworkSim(self.cfg)
        self.full_obs = full_obs
        self.nz = self.cfg.n_zones
        self.possible_agents = [f"zone_{z}" for z in range(self.nz)]
        self.agents = list(self.possible_agents)
        self.max_zone = max(len(h) for h in self.sim.zone_hosts)
        self.n_hosts = self.cfg.n_hosts
        self.obs_dim = (self.n_hosts if full_obs else self.max_zone) * FEATS_PER_HOST + 2 + (self.nz if full_obs else 0)
        self.act_dim = self.max_zone * N_ACTIONS
        self.state_dim = self.n_hosts * (FEATS_PER_HOST + 4) + 2 + 1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.act_dim)

    # ------------------------------------------------------------------ obs
    def _alerts_outside(self, z):
        score = self._alert
        other = np.ones(self.n_hosts, bool)
        other[self.sim.zone_hosts[z]] = False
        return float(score[other].sum()) / self.n_hosts

    def _obs_one(self, z):
        feats = self._feats
        shared = np.array([self.sim.threat_obs, self._alerts_outside(z)], dtype=np.float32)
        if self.full_obs:
            # whole network, zone-ordered, plus own-zone one-hot
            return np.concatenate([feats.ravel(), shared, np.eye(self.nz, dtype=np.float32)[z]])
        local = np.zeros((self.max_zone, FEATS_PER_HOST), dtype=np.float32)
        hosts = self.sim.zone_hosts[z]
        local[: len(hosts)] = feats[hosts]
        return np.concatenate([local.ravel(), shared])

    def _obs(self):
        self._feats = self.sim.host_features()               # computed once per step, shared
        self._alert = self.sim.alert_score() > ALERT_THRESH
        return {a: self._obs_one(z) for z, a in enumerate(self.possible_agents)}

    def state(self):
        """Privileged global state: all observable features + TRUE stage one-hot + time."""
        s = self.sim
        stage = np.eye(4, dtype=np.float32)[s.stage]
        feats = np.concatenate([self._feats, stage], axis=1).ravel()
        alerts = float(self._alert.sum()) / self.n_hosts
        return np.concatenate([feats, [s.threat_obs, alerts, s.t / self.cfg.horizon]]).astype(np.float32)

    # ------------------------------------------------------------- dynamics
    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.sim.reset(seed)
        return self._obs(), {a: {} for a in self.agents}

    def decode(self, z, action):
        """Zone-local action index -> (global host, action); padded hosts become no_op."""
        local, a = divmod(int(action), N_ACTIONS)
        hosts = self.sim.zone_hosts[z]
        if local >= len(hosts):
            return int(hosts[0]), NO_OP
        return int(hosts[local]), a

    def step(self, actions):
        acts = [self.decode(z, actions[a]) for z, a in enumerate(self.possible_agents)]
        rz = self.sim.step(acts)
        trunc = self.sim.t >= self.cfg.horizon
        obs = self._obs()
        rewards = {a: float(rz[z]) for z, a in enumerate(self.possible_agents)}
        infos = {a: {} for a in self.possible_agents}
        if trunc:
            m = self.sim.episode_metrics()
            for a in infos:
                infos[a]["metrics"] = m
            self.agents = []
        terms = {a: False for a in self.possible_agents}
        truncs = {a: trunc for a in self.possible_agents}
        return obs, rewards, terms, truncs, infos
