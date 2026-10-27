"""Single-agent PPO (Stable-Baselines3) on CyberDefenseEnv.

    python -m training.train_ppo --reward norm --seed 0 --name ppo_norm
    python -m training.train_ppo --reward raw  --seed 0 --name ppo_raw     # ablation (reward-scale issue)

`--reward norm` wraps the env in VecNormalize (reward + observation normalization);
`--reward raw` trains on the raw -500/step-scale reward. Training stops on a plateau of
`ep_rew_mean` (see training.utils.PlateauStopper) rather than at a fixed arbitrary step count.
"""
import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from env import CyberDefenseEnv
from training.utils import TRAIN_SEED_BASE, CSVLogger, PlateauStopper

MODELS = "results/models"
LOGS = "results/logs"


class SeedingEnv(CyberDefenseEnv):
    """Gives every training episode a fresh seed from a reserved range (never eval seeds)."""

    def __init__(self, base, **kw):
        super().__init__(**kw)
        self._next = base

    def reset(self, *, seed=None, options=None):
        seed = self._next if seed is None else seed
        self._next += 1
        return super().reset(seed=seed, options=options)


class PlateauCallback(BaseCallback):
    def __init__(self, stopper, logger_csv):
        super().__init__()
        self.stopper, self.csv = stopper, logger_csv
        self.stop = False

    def _on_step(self):
        return not self.stop   # stop flag is set at rollout end; honoured at the next step

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if len(buf) == 0:
            return
        r = float(np.mean([e["r"] for e in buf]))
        l = float(np.mean([e["l"] for e in buf]))
        lg = self.model.logger.name_to_value
        self.csv.log(timesteps=self.num_timesteps, ep_rew_mean=r, ep_len_mean=l,
                     clip_fraction=lg.get("train/clip_fraction", float("nan")),
                     approx_kl=lg.get("train/approx_kl", float("nan")),
                     explained_variance=lg.get("train/explained_variance", float("nan")),
                     value_loss=lg.get("train/value_loss", float("nan")))
        if self.stopper.update(r, self.num_timesteps):
            self.model.stop_reason = self.stopper.reason
            self.stop = True


def build_vec(seed, reward, n_envs=8):
    def mk(i):
        return lambda: SeedingEnv(TRAIN_SEED_BASE + 1_000_000 * seed + i * 200_000)
    venv = VecMonitor(DummyVecEnv([mk(i) for i in range(n_envs)]))
    if reward == "norm":
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    return venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["norm", "raw"], default="norm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--min-steps", type=int, default=400_000)
    ap.add_argument("--max-steps", type=int, default=3_000_000)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    name = a.name or f"ppo_{a.reward}_s{a.seed}"
    torch.set_num_threads(a.threads)
    os.makedirs(MODELS, exist_ok=True)

    venv = build_vec(a.seed, a.reward)
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, n_epochs=10, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, seed=a.seed, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128]), device="cpu")
    csv_log = CSVLogger(f"{LOGS}/{name}.csv", ["timesteps", "ep_rew_mean", "ep_len_mean", "clip_fraction",
                                               "approx_kl", "explained_variance", "value_loss"])
    stopper = PlateauStopper(min_steps=a.min_steps, max_steps=a.max_steps, patience=a.patience)
    cb = PlateauCallback(stopper, csv_log)
    model.stop_reason = None
    model.learn(total_timesteps=a.max_steps, callback=cb, progress_bar=False)
    csv_log.close()
    model.save(f"{MODELS}/{name}")
    if a.reward == "norm":
        venv.save(f"{MODELS}/{name}_vecnormalize.pkl")
    print(f"[{name}] stopped at {model.num_timesteps} steps -- {model.stop_reason or 'max_steps'}")


if __name__ == "__main__":
    main()
