"""Training-progress data for the demo's Learning tab, read from the real training logs in results/logs/.

Nothing is simulated here: every point is a row the trainers wrote (SB3 `ep_rew_mean` for PPO; the mean team
return of 8 parallel episodes per update for IPPO/MAPPO). Series are smoothed with a short moving average and
down-sampled to a fixed number of points so the page stays small.
"""
import numpy as np
import pandas as pd

LOGS = "results/logs"
N = 80
METHODS = {"ppo": ("PPO", "ppo_norm_s{}", 3), "ippo": ("Independent PPO", "ippo_s{}", 10), "mappo": ("MAPPO", "mappo_s{}", 10)}


def _g(v, sig=4):
    return float(f"{v:.{sig}g}")


def _series(x, y, end, n=N):
    xs = np.linspace(0, end, n)
    return xs, np.interp(xs, x, y)


def learning_data():
    comp = pd.read_csv("results/comparison_table.csv").set_index("method")
    ref = {"heur1": float(comp.loc["heuristic", "reward_mean"]), "heur3": float(comp.loc["heuristic_ma", "reward_mean"]),
           "rand1": float(comp.loc["random", "reward_mean"]), "rand3": float(comp.loc["random_ma", "reward_mean"])}
    out = {"ref": ref, "methods": {}, "entropy": {}, "metrics": {}}
    for key, (label, pat, win) in METHODS.items():
        runs = [pd.read_csv(f"{LOGS}/{pat.format(s)}.csv").bfill() for s in range(3)]
        for r in runs:
            r["sm"] = r.ep_rew_mean.rolling(win, min_periods=1).mean()
        ends = [float(r.timesteps.iloc[-1]) for r in runs]
        seeds = []
        for r, e in zip(runs, ends):
            xs, ys = _series(r.timesteps.to_numpy(float), r.sm.to_numpy(float), e)
            seeds.append({"x": [int(v) for v in xs], "y": [round(float(v)) for v in ys]})
        emin = min(ends)
        grid = np.linspace(0, emin, N)
        stack = np.array([np.interp(grid, r.timesteps.to_numpy(float), r.sm.to_numpy(float)) for r in runs])
        mean = stack.mean(0)
        out["methods"][key] = {"label": label, "seeds": seeds,
                               "mean": {"x": [int(v) for v in grid], "y": [round(float(v)) for v in mean]}}
        # headline metrics
        start = float(np.mean([r.ep_rew_mean.head(3).mean() for r in runs]))
        finals = [float(r.ep_rew_mean.tail(10).mean()) for r in runs]
        final = float(np.mean(finals))
        target = ref["heur3"] if key != "ppo" else ref["heur1"]
        def first(cond):
            idx = np.where(cond)[0]
            return int(grid[idx[0]]) if len(idx) else None
        out["metrics"][key] = {
            "label": label, "runs": 3, "start": round(start), "final": round(final), "final_sd": round(float(np.std(finals, ddof=1))),
            "plateau": int(np.mean(ends)), "beat": first(mean >= target), "beat_ref": round(target),
            "t90": first(mean >= start + 0.9 * (final - start)),
            "zero": first(mean >= 0),
            "heldout_reward": round(float(comp.loc[key, "reward_mean"])), "heldout_breach": round(float(comp.loc[key, "breach_rate"]), 3),
        }
        if key in ("ippo", "mappo"):
            ent = [r.entropy.rolling(10, min_periods=1).mean() for r in runs]
            stack = np.array([np.interp(grid, r.timesteps.to_numpy(float), e.to_numpy(float)) for r, e in zip(runs, ent)])
            out["entropy"][key] = {"x": [int(v) for v in grid], "y": [round(float(v), 3) for v in stack.mean(0)]}
    # reward-scale ablation: raw vs normalized PPO, same seed
    ab = {}
    for name, f in (("raw", "ppo_raw_s0"), ("norm", "ppo_norm_s0")):
        d = pd.read_csv(f"{LOGS}/{f}.csv").bfill()      # the first SB3 row has no training stats yet (NaN)
        e = float(d.timesteps.iloc[-1])
        xs = np.linspace(0, e, N)
        t = d.timesteps.to_numpy(float)
        ab[name] = {"x": [int(v) for v in xs], "end": int(e),
                    "rew": [round(float(v)) for v in np.interp(xs, t, d.ep_rew_mean.to_numpy(float))],
                    "clip": [_g(v) for v in np.interp(xs, t, d.clip_fraction.to_numpy(float))],
                    "kl": [_g(v) for v in np.interp(xs, t, d.approx_kl.to_numpy(float))],
                    "vloss": [_g(v) for v in np.interp(xs, t, d.value_loss.to_numpy(float))]}
    out["ablation"] = ab
    out["xmax"] = int(max(max(m["seeds"][i]["x"][-1] for i in range(3)) for m in out["methods"].values()))
    return out


if __name__ == "__main__":
    import json
    d = learning_data()
    print(json.dumps(d["metrics"], indent=1))
    print("bytes:", len(json.dumps(d, separators=(",", ":"))))
