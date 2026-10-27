"""CICIDS2017 -> CyberDefenseEnv observation schema (offline sim-to-real gap analysis only).

    python -m data.cicids_loader          # builds data/cicids_subset.csv from data/raw/extracted

Feature mapping (label-free; calibrated on BENIGN flows only):
  traffic_anom : mean two-sided benign-ECDF deviation of {Flow Duration, Total Fwd/Bwd Packets, total bytes,
                 Flow Bytes/s, Flow Packets/s, SYN / RST / FIN flag counts}
  user_anom    : same construction over behaviour-style features {Flow IAT Mean, Idle Mean, Down/Up Ratio,
                 PSH/ACK flag counts, Init_Win_bytes_forward, Average Packet Size, Fwd Packet Length Mean}
Each deviation score is rescaled so the benign median -> 0.10 (the sim's clean level) and the benign
95th percentile -> ~0.45, then clipped to [0, 1].

Label -> ground-truth compromise stage (a modelling assumption, documented in the report):
  BENIGN                                   -> 0 UNCOMPROMISED
  PortScan, Bot, Web Attack (all 3)        -> 1 EARLY_INTRUSION   (reconnaissance / initial foothold)
  FTP-Patator, SSH-Patator, Infiltration   -> 2 ESCALATED         (credential attack / internal activity)
  DoS *, DDoS, Heartbleed                  -> 3 BREACHED          (service impact / data exposure)
"""
import glob
import os

import numpy as np
import pandas as pd

RAW = "data/raw/extracted/MachineLearningCVE"
SUBSET = "data/cicids_subset.csv"

TRAFFIC_FEATS = ["Flow Duration", "Total Packets", "Total Bytes", "Flow Bytes/s", "Flow Packets/s",
                 "SYN Flag Count", "RST Flag Count", "FIN Flag Count"]
USER_FEATS = ["Flow IAT Mean", "Idle Mean", "Down/Up Ratio", "PSH Flag Count", "ACK Flag Count",
              "Init_Win_bytes_forward", "Average Packet Size", "Fwd Packet Length Mean"]

STAGE_OF = {"BENIGN": 0, "PortScan": 1, "Bot": 1, "Web Attack - Brute Force": 1, "Web Attack - XSS": 1,
            "Web Attack - Sql Injection": 1, "FTP-Patator": 2, "SSH-Patator": 2, "Infiltration": 2,
            "DoS Hulk": 3, "DoS GoldenEye": 3, "DoS slowloris": 3, "DoS Slowhttptest": 3, "DDoS": 3,
            "Heartbleed": 3}


def _clean_label(s):
    return s.replace("�", "-").strip()


def build_subset(per_attack=1500, benign=9000, seed=0):
    rng = np.random.default_rng(seed)
    base = TRAFFIC_FEATS + USER_FEATS
    raw_cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
                "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "RST Flag Count",
                "FIN Flag Count", "Flow IAT Mean", "Idle Mean", "Down/Up Ratio", "PSH Flag Count", "ACK Flag Count",
                "Init_Win_bytes_forward", "Average Packet Size", "Fwd Packet Length Mean", "Label"]
    parts = []
    files = sorted(glob.glob(f"{RAW}/*.csv"))
    for f in files:
        d = pd.read_csv(f, skipinitialspace=True, usecols=lambda c: c.strip() in raw_cols)
        d.columns = [c.strip() for c in d.columns]
        d["Label"] = d["Label"].map(_clean_label)
        d["Total Packets"] = d["Total Fwd Packets"] + d["Total Backward Packets"]
        d["Total Bytes"] = d["Total Length of Fwd Packets"] + d["Total Length of Bwd Packets"]
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=base)
        d["source_file"] = os.path.basename(f)
        for lab, g in d.groupby("Label"):
            k = benign // len(files) if lab == "BENIGN" else per_attack
            parts.append(g.sample(min(len(g), k), random_state=int(rng.integers(1 << 30))))
        del d
    out = pd.concat(parts)
    # attack classes appear in several files; cap per class overall
    out = pd.concat([g.sample(min(len(g), benign if lab == "BENIGN" else per_attack), random_state=seed)
                     for lab, g in out.groupby("Label")])
    out = out[base + ["Label", "source_file"]].reset_index(drop=True)
    out["stage"] = out["Label"].map(STAGE_OF)
    assert out["stage"].notna().all(), set(out.loc[out.stage.isna(), "Label"])
    out["stage"] = out["stage"].astype(int)
    out.to_csv(SUBSET, index=False)
    return out


class AnomalyMapper:
    """Benign-referenced ECDF deviation -> [0,1] anomaly score in the simulator's scale."""

    def __init__(self, df):
        ben = df[df.Label == "BENIGN"]
        self.ref = {c: np.sort(ben[c].to_numpy(float)) for c in TRAFFIC_FEATS + USER_FEATS}
        self.cal = {}
        for name, feats in (("traffic", TRAFFIC_FEATS), ("user", USER_FEATS)):
            s = self._raw(ben, feats)
            self.cal[name] = (np.median(s), np.percentile(s, 95))

    def _raw(self, df, feats):
        dev = []
        for c in feats:
            ref = self.ref[c]
            u = np.searchsorted(ref, df[c].to_numpy(float), side="right") / len(ref)
            dev.append(2 * np.abs(u - 0.5))
        return np.mean(dev, axis=0)

    def transform(self, df):
        out = {}
        for name, feats in (("traffic", TRAFFIC_FEATS), ("user", USER_FEATS)):
            med, p95 = self.cal[name]
            z = (self._raw(df, feats) - med) / max(p95 - med, 1e-6)
            out[name] = np.clip(0.10 + 0.35 * z, 0.0, 1.0)
        return out["traffic"], out["user"]


def load_subset():
    if not os.path.exists(SUBSET):
        build_subset()
    return pd.read_csv(SUBSET)


if __name__ == "__main__":
    d = build_subset()
    print(d.groupby(["stage", "Label"]).size().to_string())
    m = AnomalyMapper(d)
    t, u = m.transform(d)
    d["traffic_anom"], d["user_anom"] = t, u
    print(d.groupby("stage")[["traffic_anom", "user_anom"]].agg(["mean", "std"]).round(3))
    print(d.groupby("Label")[["traffic_anom", "user_anom"]].mean().round(3).to_string())
