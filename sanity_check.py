"""Sanity check: random and no-op baselines (and the heuristic) on CyberDefenseEnv."""
import argparse

import numpy as np

from agents.heuristic import HeuristicAgent
from env import CyberDefenseEnv


def run(policy, n_ep=20, seed0=0, **env_kw):
    env = CyberDefenseEnv(**env_kw)
    rets, breach = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed0 + ep)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, env))
            R += r
            done = term or trunc
        rets.append(R)
        breach.append(info["metrics"]["breach_steps"])
    return np.mean(rets), np.std(rets), np.mean(breach)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    heur = HeuristicAgent()
    rng = np.random.default_rng(0)
    pols = {
        "no_op": lambda o, e: 0,
        "random": lambda o, e: int(rng.integers(e.action_space.n)),
        "heuristic": lambda o, e: heur.predict(o),
    }
    print(f"{'policy':<10} {'mean_reward':>13} {'std':>10} {'breach_steps/ep':>16}")
    for name, pol in pols.items():
        np.random.seed(0)
        m, s, b = run(pol, a.episodes, a.seed0)
        print(f"{name:<10} {m:>13.1f} {s:>10.1f} {b:>16.1f}")
