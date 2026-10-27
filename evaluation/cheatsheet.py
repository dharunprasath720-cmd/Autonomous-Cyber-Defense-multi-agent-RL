"""Render results/cheatsheet_actions.png: every defensive action with its mechanics, costs, downtime and rewards.
All numbers come from env/core.py (constants imported, mechanics mirrored from NetworkSim._apply)."""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from env.core import (R_BREACH, R_DETECT, R_DOWNTIME, R_PREVENT, R_RESTORE_DOWNTIME, R_STEP)

INK, MUTE, LINE, BG = "#1b2430", "#5b6675", "#d9dee5", "#f6f7f9"
GREEN, RED, BLUE, AMBER, PURPLE = "#2e9e5b", "#c92a2a", "#1c7ed6", "#e0a800", "#5c3d99"
STAGE = ["#2e9e5b", "#e0a800", "#e8590c", "#c92a2a"]


def fmt(v):
    return f"{'+' if v > 0 else '−' if v < 0 else ''}{abs(v):g}"


ROWS = [
    dict(name="no_op", color="#7b8794",
         does="Do nothing this step.",
         cost="0", down="none", last="–",
         rewards="No rule fires. Any ongoing breach still costs −500 per host.",
         net="0"),
    dict(name="block_ip", color=BLUE,
         does="Blocks the attacker's address. EARLY intruder removed with 70% chance; ESCALATED pushed back to EARLY with 30%. "
              "No effect on a BREACHED host. Landing chance on this host ×0.25 afterwards.",
         cost=fmt(R_STEP), down="none", last="5 steps (landing ×0.25)",
         rewards=f"{fmt(R_DETECT)} first detection · {fmt(R_PREVENT)} if the host returns to clean",
         net="EARLY intruder: +149 (70%) / +49 (30%), expected +119"),
    dict(name="isolate_machine", color=PURPLE,
         does="Takes the host off the network: progression and new landings are frozen. "
              "Does not clean anything; the intruder is still there afterwards.",
         cost=fmt(R_STEP), down=fmt(R_DOWNTIME) + " (service offline)", last="5 steps",
         rewards=f"{fmt(R_DETECT)} first detection (on EARLY/ESCALATED). No +100: nothing is reversed.",
         net="ESCALATED host: −151, then needs restore_backup (+49): net −102"),
    dict(name="patch_system", color=GREEN,
         does="Fixes the weakness: vulnerability ×0.6, landing chance ×0.5, progression ×0.6. "
              "A new CVE un-patches a host with 2% chance per step. Does not remove an existing intruder.",
         cost=fmt(R_STEP), down="none", last="until un-patched (2%/step)",
         rewards="No direct reward. It is prevention: fewer future landings and slower escalation.",
         net="−1 now, pays off later"),
    dict(name="redirect_traffic", color="#0b7285",
         does="Sends traffic through scrubbing. EARLY intruder removed with 60% if it came from DDoS, 15% otherwise; "
              "ESCALATED DDoS pushed back with 20%. Landing chance ×0.6 (×0.1 against DDoS).",
         cost=fmt(R_STEP), down="none", last="4 steps",
         rewards=f"{fmt(R_DETECT)} first detection · {fmt(R_PREVENT)} if the host returns to clean",
         net="EARLY DDoS intruder: expected +109; other vector: expected +64"),
    dict(name="deploy_honeypot", color=AMBER,
         does="Plants a decoy on the host: 80% of attacks landing on it are trapped. Attackers pick it twice as often; "
              "progression ×0.8; its signals are less noisy. At most 3 active (oldest is retired). "
              "On an already-compromised host it exposes the intruder with 50% chance.",
         cost=fmt(R_STEP), down="none", last="20 steps",
         rewards=f"{fmt(R_DETECT)} per trapped attack · {fmt(R_DETECT)} if it exposes an existing intruder",
         net="−1, then +50 each time a landing is trapped"),
    dict(name="restore_backup", color=RED,
         does="Wipes the host back to a clean state and ends any isolation. Works at every stage; "
              "the only way out of BREACHED.",
         cost=fmt(R_STEP), down=fmt(R_RESTORE_DOWNTIME) + " (brief outage)", last="instant",
         rewards=f"{fmt(R_PREVENT)} if the host was EARLY/ESCALATED · {fmt(R_DETECT)} first detection",
         net="EARLY intruder: +99 · BREACHED: −51 (stops −500/step) · clean host: −51 wasted"),
]

COLS = [("Action", 17), ("What it does", 51), ("Step cost", 11), ("Downtime", 15), ("Lasts", 13), ("Rewards it can trigger", 34), ("Typical net reward", 29)]
W = sum(c[1] for c in COLS)
fig = plt.figure(figsize=(20, 15.2), dpi=130, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W + 6); ax.set_ylim(0, 152); ax.axis("off")
X0 = 3

ax.text(X0, 148, "Cyber Defense Cheat Sheet: actions, rewards and downtime", fontsize=26, fontweight="bold", color=INK, va="center")
ax.text(X0, 143.6, "Simulated 8-host network, 3 zones, 100-step episodes. One (host, action) pair per agent per step. "
        "Rewards are summed over all hosts and steps; the agent's goal is the highest total.", fontsize=12.5, color=MUTE, va="center")

# ---- reward constants banner
consts = [(fmt(R_PREVENT), "attack reversed\nbefore breach", GREEN), (fmt(R_DETECT), "early detection\n(first relevant action)", GREEN),
          (fmt(R_DETECT), "honeypot trap", GREEN), (fmt(R_STEP), "any action\nexcept no_op", RED),
          (fmt(R_RESTORE_DOWNTIME), "restore_backup\ndowntime", RED), (fmt(R_DOWNTIME), "isolate_machine\ndowntime", RED),
          (fmt(R_BREACH), "per breached host\nper step", RED)]
bw = (W - 6 * 1.5) / 7
for k, (val, lab, col) in enumerate(consts):
    x = X0 + k * (bw + 1.5)
    ax.add_patch(FancyBboxPatch((x, 128.2), bw, 11.6, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=col, lw=2))
    ax.text(x + bw / 2, 135.6, val, ha="center", va="center", fontsize=25, fontweight="bold", color=col)
    ax.text(x + bw / 2, 130.9, lab, ha="center", va="center", fontsize=10.3, color=INK, linespacing=1.15)

# ---- table
y = 124.2
x = X0
for name, w in COLS:
    ax.add_patch(Rectangle((x, y - 4.2), w, 4.2, fc=INK, ec="white", lw=1))
    ax.text(x + 1, y - 2.1, name, color="white", fontsize=12, fontweight="bold", va="center")
    x += w
y -= 4.2
RH = 11.8
for r_i, r in enumerate(ROWS):
    top = y - r_i * RH
    fc = "white" if r_i % 2 == 0 else "#f0f2f6"
    ax.add_patch(Rectangle((X0, top - RH), W, RH, fc=fc, ec=LINE, lw=1))
    x = X0
    cells = [r["name"], r["does"], r["cost"], r["down"], r["last"], r["rewards"], r["net"]]
    for ci, ((cname, w), txt) in enumerate(zip(COLS, cells)):
        if ci == 0:
            ax.add_patch(FancyBboxPatch((x + 0.8, top - RH + 2.2), w - 1.6, RH - 4.4, boxstyle="round,pad=0,rounding_size=1",
                                        fc=r["color"], ec="none"))
            ax.text(x + w / 2, top - RH / 2, r["name"], ha="center", va="center", color="white", fontsize=(11.2 if len(r["name"]) > 14 else 13.5), fontweight="bold")
        else:
            size = {1: 11, 2: 17, 3: 11.2, 4: 11, 5: 11, 6: 11}[ci]
            chars = {1: 60, 2: 6, 3: 15, 4: 12, 5: 36, 6: 32}[ci]
            col = INK
            weight = "normal"
            if ci == 2:
                weight, col = "bold", (RED if txt.startswith("−") else MUTE)
            if ci == 3:
                col = RED if txt.startswith("−") else MUTE
                weight = "bold" if txt.startswith("−") else "normal"
            if ci == 2 or (ci == 3 and txt.startswith("−")):
                ax.text(x + w / 2, top - RH / 2, "\n".join(textwrap.wrap(txt, chars)), ha="center", va="center",
                        fontsize=size, color=col, fontweight=weight)
            else:
                ax.text(x + 1, top - RH / 2, "\n".join(textwrap.wrap(txt, chars)), ha="left", va="center",
                        fontsize=size, color=col, fontweight=weight, linespacing=1.3)
        x += w
y_end = y - len(ROWS) * RH

# ---- bottom panels
py = y_end - 2.5
ph = py - 3.6
# compromise stages
ax.add_patch(FancyBboxPatch((X0, 3.6), 70, ph, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=LINE, lw=1.5))
ax.text(X0 + 2, py - 3, "Compromise stages (hidden from the defender)", fontsize=13, fontweight="bold", color=INK, va="center")
stages = [("UNCOMPROMISED", "clean"), ("EARLY_INTRUSION", "attacker has a foothold"),
          ("ESCALATED", "attacker gained higher privileges"), ("BREACHED", "−500 every step until restored")]
for k, (nm, ds) in enumerate(stages):
    yy = py - 8.6 - k * 5.7
    ax.add_patch(FancyBboxPatch((X0 + 2, yy - 2), 5, 4, boxstyle="round,pad=0,rounding_size=1", fc=STAGE[k], ec="none"))
    ax.text(X0 + 9, yy, nm, fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(X0 + 32, yy, ds, fontsize=11, color=MUTE, va="center")
    if k < 3:
        ax.annotate("", xy=(X0 + 4.5, yy - 3.8), xytext=(X0 + 4.5, yy - 2.1), arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.3))

# attacks
bx = X0 + 72
bw2 = 58
ax.add_patch(FancyBboxPatch((bx, 3.6), bw2, ph, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=LINE, lw=1.5))
ax.text(bx + 2, py - 3, "How the attacker (red) acts", fontsize=13, fontweight="bold", color=INK, va="center")
atk = ["Landing attempts per step ≈ 1.5 × threat level (calm 0.25 · campaign 0.85).",
       "Malware 45% · phishing 35% · DDoS 20% (phishing hits workstations, DDoS hits servers/cloud).",
       "P(land) = vector base (0.55 / 0.65 / 0.50) × vulnerability × patch factor × zone pressure.",
       "EARLY → ESCALATED 0.30/step · ESCALATED → BREACHED 0.25/step (×1.5 on databases, cloud).",
       "Lateral movement: each escalated host raises odds in the next zone by +0.6 (max two hosts)."]
for k, t in enumerate(atk):
    ax.text(bx + 2, py - 9.0 - k * 4.9, "•  " + textwrap.fill(t, 76, subsequent_indent="    "), fontsize=10.2, color=INK, va="center")

# reading tips
cx = bx + bw2 + 2
cw = X0 + W - cx
ax.add_patch(FancyBboxPatch((cx, 3.6), cw, ph, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=LINE, lw=1.5))
ax.text(cx + 2, py - 3, "Rules worth remembering", fontsize=13, fontweight="bold", color=INK, va="center")
tips = ["+50 detection is paid once per intrusion, and only on EARLY/ESCALATED hosts.",
        "+100 needs the host to return to clean before it is breached.",
        "One breached host costs −500 every step, which is why speed beats thrift.",
        "The defender sees only noisy anomaly scores, never the true stage."]
for k, t in enumerate(tips):
    ax.text(cx + 2, py - 9.4 - k * 5.9, "•  " + textwrap.fill(t, 40, subsequent_indent="    "), fontsize=10.2, color=INK, va="center")

ax.text(X0 + W, 1.7, "Source: env/core.py (constants and mechanics). Net rewards are worked examples from these rules, not measured averages.",
        fontsize=9.5, color=MUTE, ha="right", va="center")
fig.savefig("results/cheatsheet_actions.png", dpi=130, facecolor=BG)
print("saved results/cheatsheet_actions.png")
