"""Learning-curve plots + the two ablations.

    python -m evaluation.plots_and_ablations

Ablation A (reward scale):  raw -500/step-scale reward  vs  VecNormalize (same PPO, same seed)
Ablation B (observation):   full network visibility     vs  zone-limited visibility (IPPO & MAPPO, seed 0)
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import policies as P
from training.utils import eval_seeds

LOGS, OUT = "results/logs", "results"


def read(name):
    return pd.read_csv(f"{LOGS}/{name}.csv")


def smooth(x, k=10):
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()


def curves(prefix_list, ax, label, color):
    runs = [read(os.path.basename(p)[:-4]) for p in sorted(prefix_list)]
    if not runs:
        return
    for r in runs:
        ax.plot(r.timesteps, smooth(r.ep_rew_mean), color=color, alpha=0.25, lw=1)
    n = min(len(r) for r in runs)
    m = np.mean([smooth(r.ep_rew_mean)[:n] for r in runs], 0)
    ax.plot(runs[0].timesteps[:n], m, color=color, lw=2, label=f"{label} (mean of {len(runs)} seeds)")


def learning_curves():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, ax in enumerate(axs):
        curves(glob.glob(f"{LOGS}/ppo_norm_s*.csv"), ax, "single-agent PPO", "#2ca02c")
        curves(glob.glob(f"{LOGS}/ippo_s*.csv"), ax, "independent PPO", "#ff7f0e")
        curves(glob.glob(f"{LOGS}/mappo_s*.csv"), ax, "MAPPO", "#d62728")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("environment steps"); ax.set_ylabel("training episode return (team, raw)")
    axs[0].set_title("Learning curves (10-update MA; faint = single seeds)")
    axs[1].set_ylim(-1500, 3500); axs[1].set_title("Zoom on the plateau"); axs[1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_learning_curves.png", dpi=150)


def eval_models(items, n=50, setting="in_dist"):
    cfg, seeds, rows = P.SETTINGS[setting], eval_seeds(n), []
    for label, name, kind, full in items:
        pol = P.SB3Policy(name) if kind == "sb3" else P.TorchMulti(name)
        res = P.run_single(pol, cfg, seeds) if kind == "sb3" else P.run_multi(pol, cfg, seeds, full_obs=full)
        rows.append(dict(config=label, model=name, reward_mean=np.mean([r["reward"] for r in res]),
                         reward_std=np.std([r["reward"] for r in res]),
                         breach_rate=np.mean([r["breached"] for r in res]),
                         breach_steps=np.mean([r["breach_steps"] for r in res])))
    return pd.DataFrame(rows)


def ablation_reward():
    raw, norm = read("ppo_raw_s0"), read("ppo_norm_s0")
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for r, lab, c in ((raw, "raw reward", "#d62728"), (norm, "VecNormalize", "#2ca02c")):
        ax[0].plot(r.timesteps, smooth(r.ep_rew_mean), label=lab, color=c)
        ax[1].plot(r.timesteps, smooth(r.clip_fraction), color=c)
        ax[2].plot(r.timesteps, smooth(r.approx_kl), color=c)
        ax[3].plot(r.timesteps, smooth(r.value_loss), color=c)
    ax[0].set_yscale("symlog", linthresh=1000); ax[0].set_title("ep_rew_mean (raw units)"); ax[0].legend()
    ax[1].set_title("clip_fraction"); ax[2].set_title("approx_kl")
    ax[3].set_yscale("log"); ax[3].set_title("value_loss")
    for a_ in ax: a_.set_xlabel("steps")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation_reward_scale.png", dpi=150)
    diag = pd.DataFrame({
        "config": ["raw reward", "VecNormalize"],
        "train_steps": [raw.timesteps.iloc[-1], norm.timesteps.iloc[-1]],
        "final_ep_rew_mean(last10)": [raw.ep_rew_mean.tail(10).mean(), norm.ep_rew_mean.tail(10).mean()],
        "mean_clip_fraction": [raw.clip_fraction.mean(), norm.clip_fraction.mean()],
        "mean_approx_kl": [raw.approx_kl.mean(), norm.approx_kl.mean()],
        "mean_explained_var": [raw.explained_variance.tail(50).mean(), norm.explained_variance.tail(50).mean()],
        "mean_value_loss": [raw.value_loss.mean(), norm.value_loss.mean()]})
    ev = eval_models([("raw reward", "ppo_raw_s0", "sb3", False), ("VecNormalize", "ppo_norm_s0", "sb3", False)])
    return diag, ev


def ablation_obs():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a_, algo in zip(ax, ("ippo", "mappo")):
        for nm, lab, c in ((f"{algo}_s0", "zone-limited", "#1f77b4"), (f"fullobs_{algo}_s0", "full visibility", "#ff7f0e")):
            if os.path.exists(f"{LOGS}/{nm}.csv"):
                r = read(nm); a_.plot(r.timesteps, smooth(r.ep_rew_mean), label=lab, color=c)
        a_.set_title(algo.upper()); a_.set_xlabel("steps"); a_.legend()
    ax[0].set_ylabel("training return")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation_observation.png", dpi=150)
    seeds, rows = eval_seeds(50), []
    for algo in ("ippo", "mappo"):
        for label, prefix, full in ((f"{algo} zone-limited", algo, False), (f"{algo} full visibility", f"fullobs_{algo}", True)):
            per = []
            for sd in range(3):
                nm = f"{prefix}_s{sd}"
                if not os.path.exists(f"results/models/{nm}.pt"):
                    continue
                res = P.run_multi(P.TorchMulti(nm), P.SETTINGS["in_dist"], seeds, full_obs=full)
                per.append((np.mean([r["reward"] for r in res]), np.mean([r["breached"] for r in res]),
                            np.mean([r["breach_steps"] for r in res])))
            a = np.array(per)
            rows.append(dict(config=label, n_train_seeds=len(a), reward_mean=a[:, 0].mean(), reward_std_across_seeds=a[:, 0].std(ddof=1) if len(a) > 1 else np.nan,
                             breach_rate=a[:, 1].mean(), breach_steps=a[:, 2].mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    learning_curves()
    diag, ev = ablation_reward()
    print("\n=== ABLATION A: reward scale (training diagnostics) ===\n", diag.to_string(index=False))
    print("\n=== ABLATION A: held-out evaluation (50 seeds) ===\n", ev.to_string(index=False))
    diag.to_csv(f"{OUT}/ablation_reward_diagnostics.csv", index=False); ev.to_csv(f"{OUT}/ablation_reward_eval.csv", index=False)
    eo = ablation_obs()
    print("\n=== ABLATION B: observation design (50 held-out seeds, mean over training seeds) ===\n", eo.to_string(index=False))
    eo.to_csv(f"{OUT}/ablation_observation_eval.csv", index=False)
