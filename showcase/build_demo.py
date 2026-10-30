"""Assemble showcase/demo.html (self-contained) and showcase/artifact.html (publish-ready, no document wrapper)."""
import re

import json
D = json.load(open("showcase/traces.json"))
for sc in D["scenarios"]:                      # slim the payload: drop unused fields, store anomaly scores as int percent
    for st in sc["steps"]:
        for k in ("vuln", "threat", "reward_parts"):
            st.pop(k, None)
        st["traffic"] = [round(x * 100) for x in st["traffic"]]
        st["user"] = [round(x * 100) for x in st["user"]]
        for e in st["ev"]:
            if e["side"] == "odds":
                continue
            if e.get("p") is not None:
                e["p"] = round(e["p"], 3)
            if e.get("why") is None:
                e.pop("why", None)
            if e.get("mods") == []:
                e.pop("mods", None)
        st["rew"] = int(st["rew"])
        # attack-odds data: compact arrays (see template: renderOdds)
        od = {"m": [0.25, 0, 0, 0, 0, 0], "l": [], "g": []}
        for e in st["ev"]:
            if e["side"] != "odds":
                continue
            if e["kind"] == "meta":
                od["m"] = [e["theta"], round(e["lam"], 3), e["n"]] + list(e["k"])
            elif e["kind"] == "land":
                bb, v, pat, Z, mb, mr = e["comp"]
                od["l"].append([e["host"], e["vec"], {"failed": 0, "landed": 1, "trapped": 2}[e["out"]], round(e["p"], 4),
                                round(e["u"], 4), round(v, 3), int(pat), round(Z, 2), mb, mr,
                                -1 if e["tr"] is None else round(e["tr"], 3), bb])
            elif e["kind"] == "prog":
                od["g"].append([e["host"], e["before"], round(e["p"], 4), round(e["u"], 4), int(e["ok"]), e["base"], round(e["v"], 3),
                                int(e["pat"]), e["mvec"], e["mtype"], e["mhp"], e["vec"]])
        st["od"] = od
        st["ev"] = [e for e in st["ev"] if e["side"] != "odds"]
        # breach penalties are derivable (one -500 per host with stage 3) and wasted attempts are compact tuples
        st["busy"] = [[e["host"], e["vec"], int(bool(e.get("isolated")))] for e in st["ev"]
                      if e["side"] == "red" and e["outcome"] == "busy"]
        st["ev"] = [e for e in st["ev"] if not (e["side"] == "red" and e["outcome"] == "busy")
                    and not (e["side"] == "reward" and e["kind"] == "breach")]
        st["iso"] = st["iso"] if any(st["iso"]) else 0
        for k in ("patched", "honey", "block", "redirect"):
            st[k] = st[k] if any(st[k]) else 0
traces = json.dumps(D, separators=(",", ":"))
html = open("showcase/template.html", encoding="utf-8").read()
import sys
sys.path.insert(0, "showcase")
from build_learning import learning_data
html = html.replace("/*DATA*/", traces.replace("</", r"<\/"))
html = html.replace("/*LEARN*/", json.dumps(learning_data(), separators=(",", ":"), allow_nan=False))
import os
os.makedirs("showcase/site", exist_ok=True)
open("showcase/site/index.html", "w", encoding="utf-8").write(html)
open("showcase/demo.html", "w", encoding="utf-8").write(html)
print("built showcase/site/index.html and showcase/demo.html", len(html) // 1024, "KB")
