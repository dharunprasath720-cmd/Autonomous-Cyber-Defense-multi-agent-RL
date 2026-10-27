"""Multi-agent PPO on MultiCyberDefenseEnv: independent PPO and MAPPO (custom minimal implementation).

    python -m training.train_marl --algo ippo  --seed 0 --name ippo_s0
    python -m training.train_marl --algo mappo --seed 0 --name mappo_s0
    python -m training.train_marl --algo ippo --full-obs ...            # observation ablation

IPPO : one actor AND one critic per zone agent, each on its own local observation and its own
       zone-local reward. No parameter sharing, no information exchange.
MAPPO: one actor per zone agent (local obs, no parameter sharing) + ONE shared centralized
       critic on the global state (all hosts' features + TRUE compromise stages + time),
       trained on the team reward (sum of zone rewards). Execution remains decentralized.

Everything else (hyper-parameters, GAE, clip, reward scaling, plateau stopping) is identical.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from agents.marl import mlp
from env import SimConfig
from env.multi_env import MultiCyberDefenseEnv
from training.utils import TRAIN_SEED_BASE, CSVLogger, PlateauStopper

MODELS, LOGS = "results/models", "results/logs"


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        bm, bv, bc = x.mean(), x.var(), x.size
        d, tot = bm - self.mean, self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ippo", "mappo"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--full-obs", action="store_true")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--min-steps", type=int, default=400_000)
    ap.add_argument("--max-steps", type=int, default=2_000_000)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    name = a.name or f"{a.algo}_s{a.seed}"
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    E, H = a.envs, 128
    gamma, lam, clip, ent_c, vf_c, lr = 0.99, 0.95, 0.2, 0.01, 0.5, 3e-4
    epochs, mb = 10, 256
    cfg = SimConfig()
    envs = [MultiCyberDefenseEnv(cfg, full_obs=a.full_obs) for _ in range(E)]
    agents = envs[0].possible_agents
    A, T = len(agents), cfg.horizon
    obs_dim, act_dim, state_dim = envs[0].obs_dim, envs[0].act_dim, envs[0].state_dim
    central = a.algo == "mappo"

    actors = [mlp(obs_dim, act_dim, H, 0.01) for _ in range(A)]
    critics = [mlp(state_dim, 1, H, 1.0)] if central else [mlp(obs_dim, 1, H, 1.0) for _ in range(A)]
    params = [p for m in actors + critics for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5)

    n_rs = 1 if central else A                       # reward-scaling streams
    rms = [RunningMeanStd() for _ in range(n_rs)]
    ret_acc = np.zeros((E, n_rs))

    seed_counter = TRAIN_SEED_BASE + 1_000_000 * a.seed
    csv = CSVLogger(f"{LOGS}/{name}.csv", ["timesteps", "ep_rew_mean", "ep_len_mean", "policy_loss", "value_loss", "entropy"])
    stopper = PlateauStopper(min_steps=a.min_steps, max_steps=a.max_steps, patience=a.patience)
    total, upd = 0, 0
    os.makedirs(MODELS, exist_ok=True)

    def save():
        torch.save(dict(actors=[m.state_dict() for m in actors], obs_dim=obs_dim, act_dim=act_dim, hidden=H,
                        agents=list(agents), algo=a.algo, full_obs=a.full_obs, steps=total), f"{MODELS}/{name}.pt")

    while True:
        # ------------------------------------------------ rollout (one full episode per env)
        O = np.zeros((T, E, A, obs_dim), np.float32)
        S = np.zeros((T, E, state_dim), np.float32)
        ACT = np.zeros((T, E, A), np.int64)
        LP = np.zeros((T, E, A), np.float32)
        R = np.zeros((T, E, A), np.float64)
        obs_now = []
        for e in envs:
            o, _ = e.reset(seed=seed_counter)
            seed_counter += 1
            obs_now.append(o)
        ep_ret = np.zeros(E)
        for t in range(T):
            ob = np.stack([[obs_now[e][ag] for ag in agents] for e in range(E)])        # (E,A,obs)
            O[t] = ob
            S[t] = np.stack([envs[e].state() for e in range(E)])
            with torch.no_grad():
                for i in range(A):
                    dist = torch.distributions.Categorical(logits=actors[i](torch.from_numpy(ob[:, i])))
                    act = dist.sample()
                    ACT[t, :, i] = act.numpy()
                    LP[t, :, i] = dist.log_prob(act).numpy()
            for e in range(E):
                acts = {ag: int(ACT[t, e, i]) for i, ag in enumerate(agents)}
                o, r, _, _, _ = envs[e].step(acts)
                obs_now[e] = o
                R[t, e] = [r[ag] for ag in agents]
        ep_ret = R.sum(axis=(0, 2))                                                  # team return per env
        total += E * T

        # bootstrap values (truncation, not termination)
        ob_last = np.stack([[obs_now[e][ag] for ag in agents] for e in range(E)])
        S_last = np.stack([envs[e].state() for e in range(E)])

        with torch.no_grad():
            if central:
                V = critics[0](torch.from_numpy(S.reshape(T * E, -1))).view(T, E).numpy()
                Vl = critics[0](torch.from_numpy(S_last)).view(E).numpy()[None]
                V, Vl = V[..., None], Vl[..., None]                                 # (T,E,1)
            else:
                V = np.stack([critics[i](torch.from_numpy(O[:, :, i].reshape(T * E, -1))).view(T, E).numpy()
                              for i in range(A)], -1)                              # (T,E,A)
                Vl = np.stack([critics[i](torch.from_numpy(ob_last[:, i])).view(E).numpy() for i in range(A)], -1)[None]

        Rtr = R.sum(-1, keepdims=True) if central else R                             # (T,E,K)
        # reward scaling (VecNormalize-style: divide by running std of discounted return)
        Rs = np.zeros_like(Rtr)
        for t in range(T):
            ret_acc = ret_acc * gamma + Rtr[t]
            for k in range(n_rs):
                rms[k].update(ret_acc[:, k])
            Rs[t] = np.clip(Rtr[t] / np.sqrt(np.array([m.var for m in rms]) + 1e-8), -10, 10)
        ret_acc[:] = 0                                                               # episode boundary

        Vext = np.concatenate([V, Vl], 0)
        adv = np.zeros_like(Rs)
        last = 0
        for t in reversed(range(T)):
            delta = Rs[t] + gamma * Vext[t + 1] - Vext[t]
            last = delta + gamma * lam * last
            adv[t] = last
        ret = adv + V

        # ------------------------------------------------ PPO update
        N = T * E
        b_adv = torch.from_numpy(adv.astype(np.float32))
        b_ret = torch.from_numpy(ret.astype(np.float32))
        pl_acc, vl_acc, en_acc, nb = 0.0, 0.0, 0.0, 0
        for _ in range(epochs):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                ti, ei = idx // E, idx % E
                loss = 0.0
                if central:
                    v = critics[0](torch.from_numpy(S[ti.numpy(), ei.numpy()])).squeeze(-1)
                    vl = ((v - b_ret[ti, ei, 0]) ** 2).mean()
                    loss = loss + vf_c * vl
                for i in range(A):
                    ob_b = torch.from_numpy(O[ti.numpy(), ei.numpy(), i])
                    dist = torch.distributions.Categorical(logits=actors[i](ob_b))
                    act_b = torch.from_numpy(ACT[ti.numpy(), ei.numpy(), i])
                    ratio = torch.exp(dist.log_prob(act_b) - torch.from_numpy(LP[ti.numpy(), ei.numpy(), i]))
                    ad = b_adv[ti, ei, 0 if central else i]
                    ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                    pl = -torch.min(ratio * ad, torch.clamp(ratio, 1 - clip, 1 + clip) * ad).mean()
                    ent = dist.entropy().mean()
                    loss = loss + pl - ent_c * ent
                    if not central:
                        v = critics[i](ob_b).squeeze(-1)
                        vl = ((v - b_ret[ti, ei, i]) ** 2).mean()
                        loss = loss + vf_c * vl
                    pl_acc += pl.item(); en_acc += ent.item(); vl_acc += vl.item(); nb += 1
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
        upd += 1
        csv.log(timesteps=total, ep_rew_mean=float(ep_ret.mean()), ep_len_mean=T,
                policy_loss=pl_acc / nb, value_loss=vl_acc / nb, entropy=en_acc / nb)
        if upd % 25 == 0:
            save()                                   # periodic checkpoint (final save overwrites)
        if stopper.update(float(ep_ret.mean()), total):
            break

    csv.close()
    save()
    print(f"[{name}] stopped at {total} steps -- {stopper.reason}")


if __name__ == "__main__":
    main()
