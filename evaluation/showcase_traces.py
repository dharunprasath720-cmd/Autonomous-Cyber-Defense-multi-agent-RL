"""Record real episode traces (blue policy vs red attacker) for the interactive showcase.

    python -m evaluation.showcase_traces        # -> showcase/traces.json

Nothing here is mocked: every event comes from NetworkSim running the actual trained policies.
"""
import json

import numpy as np

from env import CyberDefenseEnv, SimConfig
from env.core import ACTION_NAMES, HOST_TYPE_NAMES
from env.multi_env import MultiCyberDefenseEnv
from evaluation import policies as P
from training.utils import EVAL_SEED_BASE

BLUES = {
    "none": ("No defense (all no-op)", "single"),
    "heuristic": ("Rule-based heuristic", "single"),
    "ppo": ("PPO (single agent, ppo_norm_s0)", "single"),
    "mappo": ("MAPPO (3 zone agents, mappo_s2)", "multi"),
}
REDS = {"stochastic": SimConfig()}
SEEDS = [EVAL_SEED_BASE + 3, EVAL_SEED_BASE + 11, EVAL_SEED_BASE + 21]


def snap(sim, rz, events):
    f = sim.host_features()
    return dict(
        stage=sim.stage.tolist(), traffic=np.round(sim.traffic, 2).tolist(), user=np.round(sim.user, 2).tolist(),
        iso=(sim.isolated_t > 0).astype(int).tolist(), patched=sim.patched.astype(int).tolist(),
        honey=(sim.honey_t > 0).astype(int).tolist(), block=(sim.block_t > 0).astype(int).tolist(),
        redirect=(sim.redirect_t > 0).astype(int).tolist(), vuln=np.round(sim.vuln, 2).tolist(),
        rew=float(np.sum(rz)), threat=round(float(sim.threat_true), 2), ev=events)


def run(blue, red_cfg, seed):
    kind = BLUES[blue][1]
    steps = []
    if kind == "single":
        env = CyberDefenseEnv(cfg=red_cfg)
        pol = {"none": lambda o: 0, "heuristic": P.HeuristicSingle(), "ppo": P.SB3Policy("ppo_norm_s0")}[blue]
        obs, _ = env.reset(seed=seed)
        env.sim.tracing = True
        done = False
        while not done:
            obs, r, _, done, info = env.step(pol(obs))
            steps.append(dict(snap(env.sim, [r], env.sim._ev), reward_parts=r))
        sim = env.sim
    else:
        env = MultiCyberDefenseEnv(red_cfg)
        pol = P.TorchMulti("mappo_s2")
        obs, _ = env.reset(seed=seed)
        env.sim.tracing = True
        while env.agents:
            obs, rew, _, _, info = env.step(pol(obs))
            steps.append(snap(env.sim, list(rew.values()), env.sim._ev))
        sim = env.sim
    for k, st in enumerate(steps):      # the ledger must add up exactly to the reward the agent received
        led = sum(e["amt"] for e in st["ev"] if e["side"] == "reward")
        assert abs(led - st["rew"]) < 1e-6, (blue, seed, k, led, st["rew"])
    m = sim.episode_metrics()
    return dict(blue=blue, red="stochastic",
                seed=int(seed), steps=steps,
                summary=dict(reward=round(sum(s["rew"] for s in steps)), breach_steps=m["breach_steps"],
                             intrusions=m["n_intrusions"], detected=m["n_detected"], contained=m["n_contained"],
                             trapped=m["trapped"], actions=m["n_actions"], downtime=m["downtime_cost"]))


if __name__ == "__main__":
    sim0 = CyberDefenseEnv().sim
    out = dict(
        hosts=[dict(id=int(h), type=HOST_TYPE_NAMES[int(sim0.types[h])], zone=int(sim0.zone_of[h])) for h in range(sim0.n)],
        zones=["DMZ", "Internal", "Data tier"], actions=ACTION_NAMES,
        blues={k: v[0] for k, v in BLUES.items()}, scenarios=[])
    for sd in SEEDS:
        for red, cfg in REDS.items():
            for blue in BLUES:
                sc = run(blue, cfg, sd)
                out["scenarios"].append(sc)
                print(f"seed={sd} red={red:<10} blue={blue:<10} {sc['summary']}", flush=True)
    json.dump(out, open("showcase/traces.json", "w"), separators=(",", ":"))
    print("saved showcase/traces.json")
