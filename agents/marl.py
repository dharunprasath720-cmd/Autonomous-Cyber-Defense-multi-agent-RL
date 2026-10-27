"""Networks and inference wrapper for the multi-agent PPO variants (IPPO / MAPPO)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(i, o, h=128, out_gain=1.0):
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_gain)
    return net


class MAPolicy:
    """Per-agent actors (separate parameters). Inference only needs the actors."""

    def __init__(self, actors, agents):
        self.actors, self.agents = actors, list(agents)

    @torch.no_grad()
    def act(self, obs: dict, deterministic=True):
        out = {}
        for actor, a in zip(self.actors, self.agents):
            logits = actor(torch.as_tensor(obs[a], dtype=torch.float32).unsqueeze(0))[0]
            out[a] = int(torch.argmax(logits)) if deterministic else int(torch.distributions.Categorical(logits=logits).sample())
        return out

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        actors = []
        for sd in ck["actors"]:
            net = mlp(ck["obs_dim"], ck["act_dim"], ck["hidden"])
            net.load_state_dict(sd)
            net.eval()
            actors.append(net)
        return cls(actors, ck["agents"]), ck
