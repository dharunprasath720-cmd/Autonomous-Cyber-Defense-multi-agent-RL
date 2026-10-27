"""Shared simulation core for the enterprise-network cyber-defense environments.

Both ``CyberDefenseEnv`` (single agent) and ``MultiCyberDefenseEnv`` (one agent per
zone) wrap :class:`NetworkSim`, so dynamics and reward constants are identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- enumerations -----------------------------------------------------------
SERVER, WORKSTATION, DATABASE, CLOUD, IOT = range(5)
HOST_TYPE_NAMES = ["server", "workstation", "database", "cloud", "iot"]
N_TYPES = 5

CLEAN, EARLY, ESCALATED, BREACHED = range(4)
STAGE_NAMES = ["UNCOMPROMISED", "EARLY_INTRUSION", "ESCALATED", "BREACHED"]

NO_OP, BLOCK_IP, ISOLATE, PATCH, REDIRECT, HONEYPOT, RESTORE = range(7)
ACTION_NAMES = ["no_op", "block_ip", "isolate_machine", "patch_system",
                "redirect_traffic", "deploy_honeypot", "restore_backup"]
N_ACTIONS = 7

MALWARE, PHISHING, DDOS = range(3)  # initial-access vectors; privesc/data-theft drive progression
ATTACK_NAMES = ["malware", "phishing", "ddos"]

# ---- reward constants (identical for single- and multi-agent) ----------------
R_PREVENT = 100.0      # attack reversed before breach (host returns to clean from EARLY/ESCALATED)
R_DETECT = 50.0        # early detection (first relevant action on an EARLY/ESCALATED host, or honeypot trap)
R_BREACH = -500.0      # per step, per host that is BREACHED at end of step
R_DOWNTIME = -200.0    # isolate_machine
R_RESTORE_DOWNTIME = -50.0  # restore_backup (brief outage); design choice, see report
R_STEP = -1.0          # per non-no-op action

FEATS_PER_HOST = N_TYPES + 6  # type one-hot + traffic, user, vuln, patched, isolated, honeypot

BASE_VULN = {SERVER: 0.45, WORKSTATION: 0.60, DATABASE: 0.35, CLOUD: 0.40, IOT: 0.75}
# target weights per attack vector, indexed by host type
TARGET_W = {
    MALWARE:  np.array([1.0, 1.5, 0.5, 1.0, 1.5]),
    PHISHING: np.array([0.1, 3.0, 0.1, 0.5, 0.1]),
    DDOS:     np.array([2.5, 0.05, 0.05, 2.5, 1.0]),
}
LAND_P = {MALWARE: 0.55, PHISHING: 0.65, DDOS: 0.50}
VECTOR_P = np.array([0.45, 0.35, 0.20])
TRAFFIC_BASE = np.array([0.10, 0.30, 0.50, 0.70])
USER_BASE = np.array([0.10, 0.35, 0.55, 0.75])


@dataclass
class SimConfig:
    n_hosts: int = 8
    horizon: int = 100
    attack_rate: float = 1.5     # mean landing attempts per step (scaled by threat level)
    vuln_shift: float = 0.0      # generalization: added to every host's base vulnerability
    attack_profile: str = "default"   # "default" | "boundary" (zone gateways) | "adaptive" (reactive red agent)
    prog_early: float = 0.30     # per-step base prob EARLY -> ESCALATED (privilege escalation)
    prog_esc: float = 0.25       # per-step base prob ESCALATED -> BREACHED (data theft)
    sig_noise: float = 0.22      # std of anomaly-signal noise (partial observability)
    false_alarm: float = 0.04    # per host-step probability of a spurious anomaly spike
    ema: float = 0.6             # weight of the newest reading in the sensor smoothing
    n_zones: int = 3


def make_layout(n_hosts: int, n_zones: int = 3):
    """Return (host_types, zone_of). Zones: 0=DMZ, 1=internal workstations, 2=data tier."""
    sizes = [n_hosts // n_zones + (1 if i < n_hosts % n_zones else 0) for i in range(n_zones)]
    palette = [[SERVER, CLOUD], [WORKSTATION, IOT], [DATABASE]]
    types, zones = [], []
    for z, sz in enumerate(sizes):
        pal = palette[z % len(palette)]
        for k in range(sz):
            # internal zone: 2 workstations per IoT device -> (WS, WS, IOT); dmz: (SERVER, CLOUD, SERVER)
            if z == 1:
                types.append(WORKSTATION if k % 3 != 2 else IOT)
            else:
                types.append(pal[k % len(pal)])
            zones.append(z)
    return np.array(types), np.array(zones)


class NetworkSim:
    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.n = self.cfg.n_hosts
        self.types, self.zone_of = make_layout(self.n, self.cfg.n_zones)
        self.nz = self.cfg.n_zones
        self.zone_hosts = [np.where(self.zone_of == z)[0] for z in range(self.nz)]
        self.gateways = [int(h[0]) for h in self.zone_hosts]  # first host of each zone = boundary host
        self.rng = np.random.default_rng()
        self.tracing = False      # showcase only: record structured events (consumes no RNG)
        self._ev: list = []

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r, n = self.rng, self.n
        self.t = 0
        base = np.array([BASE_VULN[int(t)] for t in self.types])
        self.vuln = np.clip(base + r.uniform(-0.1, 0.1, n) + self.cfg.vuln_shift, 0.05, 0.98)
        self.stage = np.zeros(n, dtype=int)
        self.atype = -np.ones(n, dtype=int)
        self.patched = np.zeros(n, dtype=bool)
        self.isolated_t = np.zeros(n, dtype=int)
        self.block_t = np.zeros(n, dtype=int)
        self.redirect_t = np.zeros(n, dtype=int)
        self.honey_t = np.zeros(n, dtype=int)       # remaining lifetime of honeypot (0 = none)
        self._wvec = []
        for v in range(3):
            w = TARGET_W[v][self.types].copy()
            if self.cfg.attack_profile == "boundary":
                for z in range(1, self.nz):
                    w[self.gateways[z]] *= 8.0
            self._wvec.append(w)
        self.threat_hi = False
        self.threat_obs = 0.3
        self.traffic = np.full(n, 0.1)
        self.user = np.full(n, 0.1)
        self.cur_event = [None] * n
        self.events: list[dict] = []
        self.stats = dict(breach_steps=0, downtime_cost=0.0, n_actions=0, trapped=0,
                          action_counts=np.zeros(N_ACTIONS, dtype=int))
        self._sense()
        return self

    @property
    def threat_true(self):
        return 0.85 if self.threat_hi else 0.25

    # ------------------------------------------------------------ helpers
    def _veff(self, h):
        return self.vuln[h] * (0.5 if self.patched[h] else 1.0)

    def _detect(self, h, rz):
        ev = self.cur_event[h]
        if ev is not None and ev["detect"] is None:
            ev["detect"] = self.t
            if self.stage[h] in (EARLY, ESCALATED):
                rz[self.zone_of[h]] += R_DETECT
                self._rw('detect', h, R_DETECT)

    def _clean(self, h, rz, prevented):
        ev = self.cur_event[h]
        if ev is not None:
            ev["clean"] = self.t
            self.cur_event[h] = None
        if prevented:
            rz[self.zone_of[h]] += R_PREVENT
            self._rw('prevent', h, R_PREVENT)
        self.stage[h] = CLEAN
        self.atype[h] = -1

    def _apply(self, h, a, rz):
        r = self.rng
        z = self.zone_of[h]
        s = self.stage[h]
        if a in (BLOCK_IP, ISOLATE, REDIRECT, RESTORE) and s >= EARLY:
            self._detect(h, rz)
        if a == BLOCK_IP:
            self.block_t[h] = 5
            if s == EARLY and r.random() < 0.7:
                self._clean(h, rz, True)
            elif s == ESCALATED and r.random() < 0.3:
                self.stage[h] = EARLY
        elif a == ISOLATE:
            self.isolated_t[h] = 5
            rz[z] += R_DOWNTIME
            self._rw('downtime_isolate', h, R_DOWNTIME)
            self.stats["downtime_cost"] += -R_DOWNTIME
        elif a == PATCH:
            if not self.patched[h]:
                self.patched[h] = True
                self.vuln[h] *= 0.6
        elif a == REDIRECT:
            self.redirect_t[h] = 4
            ddos = self.atype[h] == DDOS
            if s == EARLY and r.random() < (0.6 if ddos else 0.15):
                self._clean(h, rz, True)
            elif s == ESCALATED and ddos and r.random() < 0.2:
                self.stage[h] = EARLY
        elif a == HONEYPOT:
            if self.honey_t[h] == 0:
                active = np.where(self.honey_t > 0)[0]
                if len(active) >= 3:                       # cap: oldest decoy is retired
                    self.honey_t[active[np.argmin(self.honey_t[active])]] = 0
                self.honey_t[h] = 20
            if s in (EARLY, ESCALATED) and r.random() < 0.5:
                self._detect(h, rz)
        elif a == RESTORE:
            rz[z] += R_RESTORE_DOWNTIME
            self._rw('downtime_restore', h, R_RESTORE_DOWNTIME)
            self.stats["downtime_cost"] += -R_RESTORE_DOWNTIME
            if s >= EARLY:
                self._clean(h, rz, s in (EARLY, ESCALATED))
            self.isolated_t[h] = 0

    def _rw(self, kind, host, amt):
        """Showcase-only reward-ledger entry (no RNG use, no effect on dynamics)."""
        if self.tracing:
            self._ev.append(dict(side='reward', kind=kind, host=int(host), amt=float(amt)))

    # ------------------------------------------------------------ red agent
    def _red_choose(self):
        """Reactive rule-based attacker ('adaptive' profile). Uses what a real attacker can probe:
        host weakness, visible defenses (decoys, block lists, scrubbing, isolation) and footholds."""
        r = self.rng
        w = (0.05 + np.array([self._veff(h) for h in range(self.n)]))
        why = ["weak host"]
        w = np.where(self.stage > CLEAN, w * 0.05, w)                 # already owned: move on
        defended = (self.block_t > 0) | (self.redirect_t > 0) | (self.isolated_t > 0)
        w = np.where(defended, w * 0.2, w)                            # avoid hosts that are actively defended
        w = np.where(self.honey_t > 0, w * 0.15, w)                   # fingerprint & avoid decoys (not perfectly)
        foothold = np.zeros(self.n, bool)
        for z in range(1, self.nz):                                   # lateral movement into the next zone
            if ((self.stage[self.zone_hosts[z - 1]] >= EARLY) & (self.isolated_t[self.zone_hosts[z - 1]] == 0)).any():
                foothold[self.zone_hosts[z]] = True
        w = np.where(foothold, w * 2.5, w)
        w = np.where(self.types == DATABASE, w * 1.5, w)              # high-value data
        c = np.cumsum(w)
        h = int(np.searchsorted(c, r.random() * c[-1], side="right"))
        t = self.types[h]
        u = r.random()
        if t in (SERVER, CLOUD):
            vec = DDOS if u < 0.5 else MALWARE
        elif t == WORKSTATION:
            vec = PHISHING if u < 0.7 else MALWARE
        else:
            vec = MALWARE
        return h, int(vec), dict(defended=bool(defended[h]), decoy=bool(self.honey_t[h] > 0),
                                 foothold=bool(foothold[h]), weak=float(self._veff(h)))

    # ------------------------------------------------------------------ step
    def step(self, actions):
        """actions: iterable of (host, action). Returns per-zone reward vector."""
        r, cfg = self.rng, self.cfg
        rz = np.zeros(self.nz)
        self._ev = []
        for h, a in actions:
            self.stats["action_counts"][a] += 1
            if a == NO_OP:
                continue
            self.stats["n_actions"] += 1
            rz[self.zone_of[h]] += R_STEP
            self._rw('step_cost', h, R_STEP)
            if self.tracing:
                s0, ev0 = int(self.stage[h]), self.cur_event[h]
                had = ev0 is not None and ev0["detect"] is not None
            self._apply(int(h), int(a), rz)
            if self.tracing:
                self._ev.append(dict(side="blue", host=int(h), action=int(a), before=s0, after=int(self.stage[h]),
                                     detected=bool(ev0 is not None and not had and ev0["detect"] == self.t)))

        # ---- cross-zone pressure (from state at start of attack phase = "next step" effect)
        active_bad = (self.stage >= ESCALATED) & (self.isolated_t == 0)
        zmult = np.ones(self.n)
        for z in range(1, self.nz):
            k = int(active_bad[self.zone_hosts[z - 1]].sum())
            zmult[self.zone_hosts[z]] = 1.0 + 0.6 * min(2, k)

        # ---- attack landing
        any_honey = bool((self.honey_t > 0).any())
        n_att = r.poisson(cfg.attack_rate * (0.3 + 1.4 * self.threat_true))
        for _ in range(n_att):
            if cfg.attack_profile == "adaptive":
                h, vec, why = self._red_choose()
            else:
                u = r.random()
                vec = 0 if u < 0.45 else (1 if u < 0.80 else 2)          # VECTOR_P = [.45,.35,.20]
                w = self._wvec[vec]
                if any_honey:
                    w = w * (1.0 + (self.honey_t > 0))                    # decoys attract attackers
                c = np.cumsum(w)
                h = int(np.searchsorted(c, r.random() * c[-1], side="right"))
                why = None
            if self.stage[h] != CLEAN or self.isolated_t[h] > 0:
                if self.tracing:
                    self._ev.append(dict(side="red", host=h, vec=int(vec), outcome="busy", why=why,
                                         isolated=bool(self.isolated_t[h] > 0)))
                continue
            p = LAND_P[vec] * self._veff(h) * zmult[h]
            if cfg.attack_profile == "boundary" and h in self.gateways[1:]:
                p *= 1.3
            mods = []
            if self.block_t[h] > 0:
                p *= 0.25; mods.append("blocked")
            if self.redirect_t[h] > 0:
                p *= 0.1 if vec == DDOS else 0.6; mods.append("redirected")
            if self.patched[h]:
                mods.append("patched")
            if r.random() < min(p, 0.95):
                if self.honey_t[h] > 0 and r.random() < 0.8:      # decoy traps the attacker
                    rz[self.zone_of[h]] += R_DETECT
                    self._rw('trap', h, R_DETECT)
                    self.stats["trapped"] += 1
                    if self.tracing:
                        self._ev.append(dict(side="red", host=h, vec=int(vec), outcome="trapped", why=why, mods=mods, p=float(p)))
                    continue
                self.stage[h] = EARLY
                self.atype[h] = vec
                ev = dict(host=h, land=self.t, detect=None, clean=None, breach=None)
                self.events.append(ev)
                self.cur_event[h] = ev
                if self.tracing:
                    self._ev.append(dict(side="red", host=h, vec=int(vec), outcome="landed", why=why, mods=mods, p=float(p)))
            elif self.tracing:
                self._ev.append(dict(side="red", host=h, vec=int(vec), outcome="failed", why=why, mods=mods, p=float(p)))

        # ---- progression (privilege escalation, data theft)
        for h in np.where((self.stage >= EARLY) & (self.stage < BREACHED) & (self.isolated_t == 0))[0]:
            s = self.stage[h]
            vf = (0.4 + 0.6 * self.vuln[h]) * (0.6 if self.patched[h] else 1.0)
            vec = self.atype[h]
            if s == EARLY:
                p = cfg.prog_early * vf * (1.2 if vec == PHISHING else 0.6 if vec == DDOS else 1.0)
            else:
                p = cfg.prog_esc * vf * (1.4 if vec == DDOS else 1.0)
                if self.types[h] in (DATABASE, CLOUD):
                    p *= 1.5
            if self.honey_t[h] > 0:
                p *= 0.8
            if r.random() < p:
                self.stage[h] += 1
                if self.tracing:
                    self._ev.append(dict(side="red", host=int(h), outcome="progressed", vec=int(vec), before=int(s),
                                         after=int(self.stage[h])))
                if self.stage[h] == BREACHED:
                    self.cur_event[h]["breach"] = self.t

        # ---- background dynamics
        if self.threat_hi:
            self.threat_hi = r.random() > 0.10
        else:
            self.threat_hi = r.random() < 0.05
        self.patched &= ~(r.random(self.n) < 0.02)                     # new CVEs un-patch hosts
        self.vuln = np.minimum(self.vuln + 0.15 * (r.random(self.n) < 0.01), 0.98)

        # ---- timers
        for arr in (self.isolated_t, self.block_t, self.redirect_t, self.honey_t):
            np.subtract(arr, 1, out=arr, where=arr > 0)

        # ---- breach penalty
        breached = self.stage == BREACHED
        for z in range(self.nz):
            rz[z] += R_BREACH * int(breached[self.zone_hosts[z]].sum())
            for _h in self.zone_hosts[z][breached[self.zone_hosts[z]]]:
                self._rw('breach', _h, R_BREACH)
        self.stats["breach_steps"] += int(breached.sum())

        self.t += 1
        self._sense()
        return rz

    # ----------------------------------------------------------- sensors
    def _sense(self):
        r, cfg = self.rng, self.cfg
        s = self.stage
        tb = TRAFFIC_BASE[s].copy()
        ub = USER_BASE[s].copy()
        comp = s >= EARLY
        tb[comp & (self.atype == DDOS)] += 0.25
        ub[comp & (self.atype == DDOS)] -= 0.10
        ub[comp & (self.atype == PHISHING)] += 0.20
        honey = self.honey_t > 0
        tb[honey & comp] += 0.15
        ub[honey & comp] += 0.15
        noise_t = np.where(honey, 0.5, np.where(self.redirect_t > 0, 0.6, 1.0)) * cfg.sig_noise
        noise_u = np.where(honey, 0.5, 1.0) * cfg.sig_noise
        raw_t = tb + r.normal(0, 1, self.n) * noise_t
        raw_u = ub + r.normal(0, 1, self.n) * noise_u
        fa = r.random(self.n) < cfg.false_alarm
        spike_t = r.random(self.n) < 0.5
        raw_t += 0.4 * (fa & spike_t)
        raw_u += 0.4 * (fa & ~spike_t)
        raw_t = np.where(self.isolated_t > 0, 0.3 * raw_t, raw_t)       # isolated host: traffic cut
        a = cfg.ema
        self.traffic = np.minimum(np.maximum(a * raw_t + (1 - a) * self.traffic, 0.0), 1.0)
        self.user = np.minimum(np.maximum(a * raw_u + (1 - a) * self.user, 0.0), 1.0)
        thr = self.threat_true + 0.15 * r.standard_normal()
        self.threat_obs = min(1.0, max(0.0, 0.5 * thr + 0.5 * self.threat_obs))

    def host_features(self) -> np.ndarray:
        """(n_hosts, FEATS_PER_HOST) observable features (no compromise stage!)."""
        f = np.zeros((self.n, FEATS_PER_HOST), dtype=np.float32)
        f[np.arange(self.n), self.types] = 1.0
        f[:, N_TYPES] = self.traffic
        f[:, N_TYPES + 1] = self.user
        f[:, N_TYPES + 2] = self.vuln
        f[:, N_TYPES + 3] = self.patched
        f[:, N_TYPES + 4] = self.isolated_t > 0
        f[:, N_TYPES + 5] = self.honey_t > 0
        return f

    def alert_score(self):
        return 0.5 * (self.traffic + self.user)

    # ----------------------------------------------------------- metrics
    def episode_metrics(self) -> dict:
        ttd = [e["detect"] - e["land"] for e in self.events if e["detect"] is not None]
        ttc = [e["clean"] - e["detect"] for e in self.events
               if e["detect"] is not None and e["clean"] is not None]
        return dict(
            breach_steps=self.stats["breach_steps"],
            breached=int(self.stats["breach_steps"] > 0),
            downtime_cost=self.stats["downtime_cost"],
            n_intrusions=len(self.events),
            n_detected=len(ttd),
            n_contained=len(ttc),
            ttd=ttd, ttc=ttc,
            trapped=self.stats["trapped"],
            n_actions=self.stats["n_actions"],
        )
