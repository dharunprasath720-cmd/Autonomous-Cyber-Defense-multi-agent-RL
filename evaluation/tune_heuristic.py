"""Grid-search the heuristic thresholds on TUNING seeds (disjoint from training and held-out eval seeds)."""
import itertools, json, sys
import numpy as np
from agents.heuristic import HeuristicAgent, HeuristicParams
from env import CyberDefenseEnv
from training.utils import TUNE_SEED_BASE

def score(p, n_ep=20):
    ag, env, R = HeuristicAgent(p), CyberDefenseEnv(), []
    for k in range(n_ep):
        o, _ = env.reset(seed=TUNE_SEED_BASE + k); d = False; tot = 0.0
        while not d:
            o, r, _, d, _ = env.step(ag.predict(o)); tot += r
        R.append(tot)
    return float(np.mean(R))

if __name__ == "__main__":
    grid = itertools.product([0.26, 0.30, 0.34, 0.38], [0.42, 0.48, 0.54], [0.58, 0.65, 0.72], [0.45, 0.60, 2.0])
    res = []
    for te, ts, tb, tv in grid:
        if not te < ts < tb: continue
        p = HeuristicParams(te, ts, tb, tv); res.append((score(p), p))
    res.sort(key=lambda x: -x[0])
    print("default:", score(HeuristicParams()))
    for s, p in res[:6]: print(f"{s:10.1f}  {p}")
    json.dump(res[0][1].__dict__, open("agents/heuristic_tuned.json", "w"), indent=1)
    print("saved best ->", "agents/heuristic_tuned.json")
