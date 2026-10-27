"""Full evaluation suite: 5 methods (+2 capacity-matched references) x in-distribution and
shifted-distribution settings over >=50 held-out seeds.

    python -m evaluation.evaluate --episodes 50

Methods
  random, heuristic, ppo (SB3, 1 action/step)      -> CyberDefenseEnv
  ippo, mappo (3 zone agents, 1 action/zone/step)  -> MultiCyberDefenseEnv
  random_ma, heuristic_ma                          -> reference rows with the SAME action capacity as the MARL methods
Trained methods are evaluated for every available training seed; episodes are pooled.
"""
import argparse
import os

import numpy as np
import pandas as pd

from evaluation import policies as P
from training.utils import eval_seeds

OUT = "results"


def collect(n_ep, settings=None, models=None, methods=None):
    seeds = eval_seeds(n_ep)
    settings = settings or list(P.SETTINGS)
    rows = []
    ppo_models = models.get("ppo") if models else [os.path.basename(p)[:-4] for p in P.discover("ppo_norm", ".zip")]
    ippo_models = models.get("ippo") if models else [os.path.basename(p)[:-3] for p in P.discover("ippo", ".pt")]
    mappo_models = models.get("mappo") if models else [os.path.basename(p)[:-3] for p in P.discover("mappo", ".pt")]
    for sname in settings:
        cfg = P.SETTINGS[sname]
        jobs = [("random", "single", "-", P.RandomSingle(0)), ("heuristic", "single", "-", P.HeuristicSingle()),
                ("random_ma", "multi", "-", P.RandomMulti(0)), ("heuristic_ma", "multi", "-", P.HeuristicMulti())]
        jobs += [("ppo", "single", m, P.SB3Policy(m)) for m in ppo_models]
        jobs += [("ippo", "multi", m, P.TorchMulti(m)) for m in ippo_models]
        jobs += [("mappo", "multi", m, P.TorchMulti(m)) for m in mappo_models]
        for method, kind, model, pol in jobs:
            if methods and method not in methods:
                continue
            res = P.run_single(pol, cfg, seeds) if kind == "single" else P.run_multi(pol, cfg, seeds)
            for r in res:
                rows.append(dict(setting=sname, method=method, model=model, **r))
            print(f"  [{sname}] {method:<13}{model:<14} mean reward {np.mean([r['reward'] for r in res]):>11.1f}", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (setting, method), g in df.groupby(["setting", "method"], sort=False):
        ttd = np.concatenate([np.asarray(x, float) for x in g["ttd"]]) if len(g) else np.array([])
        ttc = np.concatenate([np.asarray(x, float) for x in g["ttc"]]) if len(g) else np.array([])
        per_model = g.groupby("model")["reward"].mean()
        out.append(dict(
            setting=setting, method=method, n_models=g["model"].nunique(), n_episodes=len(g),
            reward_mean=g["reward"].mean(), reward_std=g["reward"].std(),
            reward_std_across_train_seeds=per_model.std() if len(per_model) > 1 else np.nan,
            breach_rate=g["breached"].mean(), breach_steps_per_ep=g["breach_steps"].mean(),
            ttd_mean=ttd.mean() if len(ttd) else np.nan,
            detect_rate=g["n_detected"].sum() / max(g["n_intrusions"].sum(), 1),
            ttc_mean=ttc.mean() if len(ttc) else np.nan,
            contain_rate=g["n_contained"].sum() / max(g["n_detected"].sum(), 1),
            downtime_cost=g["downtime_cost"].mean(), actions_per_ep=g["n_actions"].mean()))
    return pd.DataFrame(out)


ORDER = ["random", "heuristic", "ppo", "ippo", "mappo", "random_ma", "heuristic_ma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--methods", nargs="*", default=None, help="subset (interim runs only)")
    ap.add_argument("--settings", nargs="*", default=None)
    a = ap.parse_args()
    assert a.episodes >= 30, "need >= 30 held-out seeds"
    if not a.methods and not a.settings:
        assert a.episodes >= 50, "full suite requires >= 50 held-out seeds"
    df = collect(a.episodes, a.settings, None, a.methods)
    if a.methods or a.settings:      # interim run: don't overwrite the final artefacts
        pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
        print(summarize(df).to_string(index=False))
        return
    df.to_pickle(f"{OUT}/eval_episodes.pkl")
    df.drop(columns=["ttd", "ttc"]).to_csv(f"{OUT}/eval_episodes.csv", index=False)
    s = summarize(df)
    s["method"] = pd.Categorical(s["method"], ORDER, ordered=True)
    s = s.sort_values(["setting", "method"])
    s.to_csv(f"{OUT}/comparison_table_full.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    main_t = s[s.setting == "in_dist"].drop(columns="setting")
    main_t.to_csv(f"{OUT}/comparison_table.csv", index=False)
    print("\n=== IN-DISTRIBUTION (held-out seeds) ===\n", main_t.to_string(index=False))
    gen = s.pivot(index="method", columns="setting", values="reward_mean").reindex(ORDER)
    gen_b = s.pivot(index="method", columns="setting", values="breach_rate").reindex(ORDER)
    gen = gen.join(gen_b, rsuffix="_breach_rate")
    gen.to_csv(f"{OUT}/generalization_table.csv")
    print("\n=== GENERALIZATION (mean reward | breach rate) ===\n", gen.to_string())
    plot(s)


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = s[s.setting == "in_dist"].set_index("method").reindex(ORDER)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = ["#999999", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#cccccc", "#9ecae1"]
    ax[0].bar(d.index, d.reward_mean, yerr=d.reward_std, color=colors, capsize=3)
    ax[0].set_yscale("symlog", linthresh=1000)
    ax[0].set_ylabel("mean episode reward (symlog)")
    ax[0].set_title("Reward, 50 held-out seeds (± per-episode std)")
    ax[1].bar(d.index, d.breach_rate, color=colors)
    ax[1].set_ylabel("breach rate (episodes with >=1 breach-step)")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Breach rate")
    for a_ in ax:
        a_.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_comparison.png", dpi=150)
    # generalization chart
    g = s.pivot(index="method", columns="setting", values="reward_mean").reindex(ORDER)
    ax2 = g.plot(kind="bar", figsize=(10, 4.2), logy=False)
    ax2.set_yscale("symlog", linthresh=1000)
    ax2.set_ylabel("mean episode reward (symlog)")
    ax2.set_title("Generalization: reward under distribution shift")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_generalization.png", dpi=150)


if __name__ == "__main__":
    main()
