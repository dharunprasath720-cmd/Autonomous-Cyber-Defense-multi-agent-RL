"""Uniform policy wrappers + episode runners used by every evaluation script."""
from __future__ import annotations

import glob
import os

import numpy as np

from agents.heuristic import HeuristicAgent
from agents.marl import MAPolicy
from env import CyberDefenseEnv, SimConfig
from env.core import FEATS_PER_HOST, N_ACTIONS
from env.multi_env import MultiCyberDefenseEnv

MODELS = "results/models"

# generalization shifts relative to the training distribution
SETTINGS = {
    "in_dist": SimConfig(),
    "high_vuln": SimConfig(vuln_shift=0.20),           # every host +0.20 base vulnerability
    "boundary": SimConfig(attack_profile="boundary"),  # attacker concentrates on zone-gateway hosts
    "adaptive_red": SimConfig(attack_profile="adaptive"),  # reactive rule-based red agent (never seen in training)
}


# ----------------------------------------------------------------- single-agent policies
class RandomSingle:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return int(self.rng.integers(CyberDefenseEnv().action_space.n))


class HeuristicSingle:
    def __init__(self):
        self.h = HeuristicAgent()

    def __call__(self, obs):
        return self.h.predict(obs)


class SB3Policy:
    def __init__(self, name):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        self.model = PPO.load(f"{MODELS}/{name}.zip", device="cpu")
        self.vn = None
        p = f"{MODELS}/{name}_vecnormalize.pkl"
        if os.path.exists(p):
            self.vn = VecNormalize.load(p, DummyVecEnv([lambda: CyberDefenseEnv()]))
            self.vn.training = False

    def __call__(self, obs):
        o = obs[None]
        if self.vn is not None:
            o = self.vn.normalize_obs(o)
        return int(self.model.predict(o, deterministic=True)[0][0])


# ----------------------------------------------------------------- multi-agent policies
class RandomMulti:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return {a: int(self.rng.integers(21)) for a in obs}


class HeuristicMulti:
    """Same rule-based agent, applied independently inside each zone (3 actions/step)."""

    def __init__(self):
        self.h = HeuristicAgent()

    def __call__(self, obs):
        out = {}
        for a, o in obs.items():
            feats = o[:-2].reshape(-1, FEATS_PER_HOST)
            _, h, act = self.h.host_choice(feats)
            out[a] = h * N_ACTIONS + act
        return out


class TorchMulti:
    def __init__(self, name):
        self.pol, self.ck = MAPolicy.load(f"{MODELS}/{name}.pt")
        self.full_obs = self.ck.get("full_obs", False)

    def __call__(self, obs):
        return self.pol.act(obs, deterministic=True)


# ----------------------------------------------------------------- runners
def run_single(policy, cfg, seeds):
    env = CyberDefenseEnv(cfg=cfg)
    rows = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, R, info = False, 0.0, {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            R += r
            done = term or trunc
        rows.append(dict(seed=s, reward=R, **info["metrics"]))
    return rows


def run_multi(policy, cfg, seeds, full_obs=False):
    env = MultiCyberDefenseEnv(cfg, full_obs=full_obs)
    rows = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        R, info = 0.0, {}
        while env.agents:
            obs, rew, _, _, info = env.step(policy(obs))
            R += sum(rew.values())
        rows.append(dict(seed=s, reward=R, **info["zone_0"]["metrics"]))
    return rows


def discover(prefix, ext):
    return sorted(glob.glob(f"{MODELS}/{prefix}_s*{ext}"))
