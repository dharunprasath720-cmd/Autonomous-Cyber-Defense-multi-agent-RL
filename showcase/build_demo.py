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
            if e.get("p") is not None:
                e["p"] = round(e["p"], 3)
            if e.get("why") is None:
                e.pop("why", None)
            if e.get("mods") == []:
                e.pop("mods", None)
        st["rew"] = int(st["rew"])
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
html = html.replace("/*DATA*/", traces.replace("</", r"<\/"))
import os
os.makedirs("showcase/site", exist_ok=True)
open("showcase/site/index.html", "w", encoding="utf-8").write(html)
open("showcase/demo.html", "w", encoding="utf-8").write(html)
print("built showcase/site/index.html and showcase/demo.html", len(html) // 1024, "KB")
