"""OFFLINE sim-to-real gap analysis on labeled CICIDS2017 flows (replay inference only).

No retraining, no live action execution, no claim of real deployment. A trained simulator policy is
shown observation vectors built from real labeled flows (see data/cicids_loader.py); we record the
response it implies for every host at every step and compare against the ground-truth stage derived
from the flow label.

    python -m evaluation.real_data_replay --policy ppo_norm_s0

Implied response per host = argmax over that host's 7 action logits (the policy normally picks one host
per step; the per-host view lets us score every flow). Replay stream: each of the 8 host slots receives
contiguous label-homogeneous segments (10-30 steps; 60% benign) of real flows, smoothed with the
simulator's sensor EMA, so temporal context matches what the policy saw in training.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from agents.heuristic import HeuristicAgent
from data.cicids_loader import STAGE_OF, AnomalyMapper, load_subset
from env.core import (ACTION_NAMES, BLOCK_IP, FEATS_PER_HOST, ISOLATE, N_ACTIONS, N_TYPES, REDIRECT, RESTORE,
                      STAGE_NAMES, make_layout)
from env.multi_env import MultiCyberDefenseEnv
from evaluation import policies as P

RESPONSIVE = {BLOCK_IP, ISOLATE, REDIRECT, RESTORE}   # actions that actively respond to an incident
EMA = 0.6


def build_streams(df, tr, us, n_steps, n_hosts, rng):
    labels = [l for l in df.Label.unique() if l != "BENIGN"]
    pools = {l: np.where(df.Label.to_numpy() == l)[0] for l in df.Label.unique()}
    idx = np.zeros((n_hosts, n_steps), int)
    for h in range(n_hosts):
        seq = []
        while len(seq) < n_steps:
            lab = "BENIGN" if rng.random() < 0.6 else labels[rng.integers(len(labels))]
            seq += list(rng.choice(pools[lab], int(rng.integers(10, 31))))
        idx[h] = seq[:n_steps]
    return idx, tr[idx], us[idx], df.stage.to_numpy()[idx], df.Label.to_numpy()[idx]


def host_obs(types, traffic, user, threat):
    f = np.zeros((len(types), FEATS_PER_HOST), np.float32)
    f[np.arange(len(types)), types] = 1
    f[:, N_TYPES], f[:, N_TYPES + 1], f[:, N_TYPES + 2] = traffic, user, 0.5   # vuln fixed at mid-level
    return f, np.float32(threat)


def sb3_host_logits(pol, feats, threat):
    obs = np.concatenate([feats.ravel(), [threat]]).astype(np.float32)[None]
    if pol.vn is not None:
        obs = pol.vn.normalize_obs(obs)
    with torch.no_grad():
        dist = pol.model.policy.get_distribution(torch.as_tensor(obs))
        return dist.distribution.logits[0].numpy().reshape(-1, N_ACTIONS)


def marl_host_logits(pol, feats, threat, zone_hosts, max_zone):
    out = np.zeros((feats.shape[0], N_ACTIONS))
    shared_alert = (0.5 * (feats[:, N_TYPES] + feats[:, N_TYPES + 1]) > 0.45)
    for z, hosts in enumerate(zone_hosts):
        local = np.zeros((max_zone, FEATS_PER_HOST), np.float32)
        local[: len(hosts)] = feats[hosts]
        other = np.ones(len(feats), bool); other[hosts] = False
        o = np.concatenate([local.ravel(), [threat, shared_alert[other].sum() / len(feats)]]).astype(np.float32)
        with torch.no_grad():
            lg = pol.pol.actors[z](torch.as_tensor(o)[None])[0].numpy().reshape(max_zone, N_ACTIONS)
        out[hosts] = lg[: len(hosts)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="e.g. ppo_norm_s0 | ippo_s0 | mappo_s1")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = load_subset()
    mapper = AnomalyMapper(df)
    tr, us = mapper.transform(df)
    types, zone_of = make_layout(8)
    nH = 8
    idx, T, U, S, L = build_streams(df, tr, us, a.steps, nH, rng)

    is_ppo = a.policy.startswith("ppo")
    pol = P.SB3Policy(a.policy) if is_ppo else P.TorchMulti(a.policy)
    zone_hosts = [np.where(zone_of == z)[0] for z in range(3)]
    heur = HeuristicAgent()

    sm_t, sm_u = np.full(nH, 0.1), np.full(nH, 0.1)
    threat_s = 0.3
    rows = []
    for t in range(a.steps):
        sm_t = EMA * T[:, t] + (1 - EMA) * sm_t
        sm_u = EMA * U[:, t] + (1 - EMA) * sm_u
        threat_s = 0.5 * float(np.clip(2 * np.mean(0.5 * (T[:, t] + U[:, t])), 0, 1)) + 0.5 * threat_s
        feats, thr = host_obs(types, sm_t, sm_u, threat_s)
        lg = sb3_host_logits(pol, feats, thr) if is_ppo else marl_host_logits(pol, feats, thr, zone_hosts, 3)
        for h in range(nH):
            rows.append(dict(t=t, host=h, stage=int(S[h, t]), label=L[h, t],
                             policy_action=int(np.argmax(lg[h])),
                             heuristic_action=int(heur.host_choice(feats[h:h + 1])[2])))
    r = pd.DataFrame(rows)
    r["policy_responds"] = r.policy_action.isin(RESPONSIVE)
    r["heur_responds"] = r.heuristic_action.isin(RESPONSIVE)
    r["is_attack"] = r.stage > 0

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"\nOFFLINE REPLAY (sim-to-real gap analysis; NOT a deployment claim) -- policy={a.policy}, "
          f"{len(r)} host-steps from real labeled CICIDS2017 flows\n")
    for who, col in (("TRAINED POLICY", "policy_action"), ("HEURISTIC (reference)", "heuristic_action")):
        ct = pd.crosstab(r.stage.map(dict(enumerate(STAGE_NAMES))), r[col].map(dict(enumerate(ACTION_NAMES))))
        ct = ct.reindex(index=STAGE_NAMES, columns=ACTION_NAMES, fill_value=0)
        ct.index.name = "ground-truth stage \\ implied action"
        print(f"--- {who}: counts ---\n{ct.to_string()}\n")
        print(f"--- {who}: row-normalised (%) ---\n{(100 * ct.div(ct.sum(axis=1), axis=0)).round(1).to_string()}\n")
    summ = {}
    for who, col in (("policy", "policy_responds"), ("heuristic", "heur_responds")):
        tp = int((r[col] & r.is_attack).sum()); fp = int((r[col] & ~r.is_attack).sum())
        fn = int((~r[col] & r.is_attack).sum()); tn = int((~r[col] & ~r.is_attack).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        summ[who] = dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec,
                         f1=2 * prec * rec / max(prec + rec, 1e-9), false_alarm_rate=fp / max(fp + tn, 1),
                         recall_by_stage={STAGE_NAMES[s]: float(r[(r.stage == s)][col].mean()) for s in (1, 2, 3)})
        print(f"[{who}] binary 'responds vs attack' confusion: TP={tp} FP={fp} FN={fn} TN={tn} | "
              f"precision={prec:.3f} recall={rec:.3f} F1={summ[who]['f1']:.3f} FAR={summ[who]['false_alarm_rate']:.3f}")
        print("        recall by stage:", {k: round(v, 3) for k, v in summ[who]['recall_by_stage'].items()})
    per_label = r.groupby("label")[["policy_responds", "heur_responds"]].mean().round(3)
    print("\nResponse rate by true label (fraction of host-steps where an active response is implied):\n",
          per_label.to_string())
    out = f"results/real_data_{a.policy}"
    pd.crosstab(r.stage.map(dict(enumerate(STAGE_NAMES))), r.policy_action.map(dict(enumerate(ACTION_NAMES)))) \
        .reindex(index=STAGE_NAMES, columns=ACTION_NAMES, fill_value=0).to_csv(out + "_confusion.csv")
    per_label.to_csv(out + "_per_label.csv")
    json.dump(summ, open(out + "_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
