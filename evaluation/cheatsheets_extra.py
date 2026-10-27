"""Two more cheat sheets (PNG), all values from env/core.py:

  results/cheatsheet_action_outcomes.png : what every action does at every compromise stage (outcome + net reward)
  results/cheatsheet_zones.png           : which zone each action touches, who may use it, cross-zone spill-over
"""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTE, LINE, BG = "#1b2430", "#5b6675", "#d9dee5", "#f6f7f9"
GREEN, RED, BLUE, AMBER, PURPLE, TEAL, GREY = "#2e9e5b", "#c92a2a", "#1c7ed6", "#e0a800", "#5c3d99", "#0b7285", "#7b8794"
STAGE = ["#2e9e5b", "#e0a800", "#e8590c", "#c92a2a"]
ACT = [("no_op", GREY), ("block_ip", BLUE), ("isolate_machine", PURPLE), ("patch_system", GREEN),
       ("redirect_traffic", TEAL), ("deploy_honeypot", AMBER), ("restore_backup", RED)]


def canvas(w, h, units_w, units_h):
    fig = plt.figure(figsize=(w, h), dpi=130, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, units_w); ax.set_ylim(0, units_h); ax.axis("off")
    return fig, ax


def chip(ax, x, y, w, h, text, color, size=12.5, tcolor="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcolor, fontsize=size, fontweight="bold")


def wrap(t, n):
    return "\n".join(textwrap.wrap(t, n))


# =====================================================================================================
# Sheet A: outcomes by stage
# =====================================================================================================
# (net label, tone, text) ; tone: +, -, 0
CELLS = {
    "no_op": [("0", "0", "Nothing happens."),
              ("0", "0", "Intruder keeps progressing (0.30 per step to ESCALATED)."),
              ("0", "0", "Intruder keeps progressing (0.25 per step to BREACHED)."),
              ("−500 / step", "-", "The breach penalty continues every step.")],
    "block_ip": [("−1", "-", "Pre-emptive: landing chance on this host ×0.25 for 5 steps."),
                 ("+149  or  +49", "+", "70%: intruder removed (+50 detect, +100 reversed, −1). 30%: still there (+50, −1)."),
                 ("+49", "+", "+50 detection, −1. 30%: pushed back to EARLY. 70%: unchanged."),
                 ("−1", "-", "No effect on a breached host. Penalty continues.")],
    "isolate_machine": [("−201", "-", "Wasted: −1, −200 downtime. Host frozen for 5 steps."),
                        ("−151", "-", "+50 detection, −1, −200. Frozen 5 steps, intruder remains: restore needed."),
                        ("−151", "-", "+50 detection, −1, −200. Frozen 5 steps, intruder remains: restore needed."),
                        ("−201", "-", "No detection reward at this stage. Penalty continues.")],
    "patch_system": [("−1", "-", "Vulnerability ×0.6, landing ×0.5, progression ×0.6."),
                     ("−1", "-", "Hardens the host and slows escalation ×0.6. Intruder stays."),
                     ("−1", "-", "Hardens the host and slows breach ×0.6. Intruder stays."),
                     ("−1", "-", "Does not clean. Penalty continues.")],
    "redirect_traffic": [("−1", "-", "Scrubbing for 4 steps: landing ×0.6 (×0.1 for DDoS)."),
                         ("+149  or  +49", "+", "DDoS origin: 60% removed. Other vectors: 15%. Removed = +50 +100 −1, else +49."),
                         ("+49", "+", "+50 detection, −1. DDoS origin: 20% pushed back to EARLY."),
                         ("−1", "-", "No effect on a breached host.")],
    "deploy_honeypot": [("−1", "-", "Decoy for 20 steps: 80% of landings trapped (+50 each)."),
                        ("−1  or  +49", "0", "50%: intruder exposed (+50). Otherwise just −1."),
                        ("−1  or  +49", "0", "50%: intruder exposed (+50). Otherwise just −1."),
                        ("−1", "-", "No detection chance at this stage.")],
    "restore_backup": [("−51", "-", "Wasted: −1 and −50 downtime on a clean host."),
                       ("+99", "+", "Cleaned: +50 detection, +100 reversed, −1, −50 downtime."),
                       ("+99", "+", "Cleaned: +50 detection, +100 reversed, −1, −50 downtime."),
                       ("−51", "-", "Host cleaned, so −500 per step stops. Costs −1, −50.")],
}
TONE = {"+": (GREEN, "#e6f5ec"), "-": (RED, "#fbe9e9"), "0": (MUTE, "#eef0f4")}
STAGES = [("UNCOMPROMISED", "clean host", STAGE[0]), ("EARLY_INTRUSION", "attacker has a foothold", STAGE[1]),
          ("ESCALATED", "attacker has higher privileges", STAGE[2]), ("BREACHED", "−500 every step", STAGE[3])]


def sheet_outcomes():
    cw0, cw = 20, 36
    W = cw0 + 4 * cw
    RH = 14.2
    fig, ax = canvas(20, 13.6, W + 6, 136)
    ax.set_ylim(10, 136)
    fig.set_size_inches(20, 12.6)
    X0 = 3
    ax.text(X0, 131.5, "Cheat Sheet: what each action does at each stage", fontsize=25, fontweight="bold", color=INK, va="center")
    ax.text(X0, 126.8, "Net = reward from this one action (detection +50, reversal +100, step cost −1, downtime). "
            "Columns are the host's TRUE stage, which the defender cannot see.", fontsize=12.2, color=MUTE, va="center")
    y = 122.5
    ax.add_patch(Rectangle((X0, y - 7), cw0, 7, fc=INK, ec="white", lw=1))
    ax.text(X0 + cw0 / 2, y - 3.5, "Action  \\  host stage", color="white", fontsize=11.5, fontweight="bold", ha="center", va="center")
    for k, (nm, ds, col) in enumerate(STAGES):
        x = X0 + cw0 + k * cw
        ax.add_patch(Rectangle((x, y - 7), cw, 7, fc=col, ec="white", lw=1))
        ax.text(x + cw / 2, y - 2.6, nm, color="white", fontsize=12.2, fontweight="bold", ha="center", va="center")
        ax.text(x + cw / 2, y - 5.2, ds, color="white", fontsize=9.4, ha="center", va="center")
    y -= 7
    for ri, (an, ac) in enumerate(ACT):
        top = y - ri * RH
        ax.add_patch(Rectangle((X0, top - RH), W, RH, fc="white" if ri % 2 == 0 else "#f0f2f6", ec=LINE, lw=1))
        chip(ax, X0 + 0.8, top - RH + 2.6, cw0 - 1.6, RH - 5.2, an, ac, size=(11 if len(an) > 14 else 12.5))
        for k, (net, tone, txt) in enumerate(CELLS[an]):
            x = X0 + cw0 + k * cw
            tcol, fill = TONE[tone]
            ax.add_patch(FancyBboxPatch((x + 1.0, top - 4.9), 14.5 if len(net) > 6 else 9.5, 3.6,
                                        boxstyle="round,pad=0,rounding_size=0.8", fc=fill, ec=tcol, lw=1.2))
            ax.text(x + 1.0 + (14.5 if len(net) > 6 else 9.5) / 2, top - 3.1, net, ha="center", va="center", fontsize=11.2,
                    fontweight="bold", color=tcol)
            ax.text(x + 1.2, top - 9.3, wrap(txt, 40), ha="left", va="center", fontsize=9.9, color=INK, linespacing=1.28)
    yb = y - len(ACT) * RH - 3.2
    note = ("Detection (+50) is paid once per intrusion: if the host was already detected, subtract 50 from the figures above. "
            "Breach penalties (−500 per breached host) are charged at the end of every step on top of these numbers. "
            "Chances are the simulator's probabilities (env/core.py).")
    ax.text(X0, yb, wrap(note, 215), fontsize=10, color=MUTE, va="top", linespacing=1.3)
    fig.savefig("results/cheatsheet_action_outcomes.png", dpi=130, facecolor=BG)
    plt.close(fig)
    print("saved results/cheatsheet_action_outcomes.png")


# =====================================================================================================
# Sheet B: zones
# =====================================================================================================
ZONES = [("DMZ", "zone_0 agent", [("H0", "server"), ("H1", "cloud"), ("H2", "server")], "#e7f0fb", BLUE),
         ("Internal", "zone_1 agent", [("H3", "workstation"), ("H4", "workstation"), ("H5", "IoT")], "#fff4e0", AMBER),
         ("Data tier", "zone_2 agent", [("H6", "database"), ("H7", "database")], "#f3e9fb", PURPLE)]
ROWS_B = [
    ("no_op", "Nothing.", "None.", "Anyone.", "None."),
    ("block_ip", "One host: landing ×0.25 for 5 steps. May remove an EARLY intruder or push ESCALATED to EARLY.",
     "If an ESCALATED host drops to EARLY or clean, the next zone loses a lateral-pressure source.",
     "Single agent: any host. Zone agent: own zone's hosts only.", "Zone of the target host."),
    ("isolate_machine", "One host: frozen 5 steps (no landing, no progression).",
     "Strongest spill-over control: an isolated host stops counting as a pressure source for 5 steps, so the NEXT zone is protected even though the host stays compromised.",
     "Single agent: any host. Zone agent: own zone only.", "−200 lands in the target host's zone."),
    ("patch_system", "One host: vulnerability ×0.6, landing ×0.5, progression ×0.6.",
     "None directly. Fewer escalated hosts later means less pressure downstream.",
     "Single agent: any host. Zone agent: own zone only.", "−1 in the target host's zone."),
    ("redirect_traffic", "One host: scrubbing 4 steps (landing ×0.6, ×0.1 for DDoS).",
     "Minor: DDoS-origin ESCALATED host drops to EARLY with 20%, easing the next zone.",
     "Single agent: any host. Zone agent: own zone only.", "−1 in the target host's zone."),
    ("deploy_honeypot", "One host: 20-step decoy; 80% of landings on it are trapped.",
     "SHARED LIMIT: at most 3 decoys exist network-wide. A new decoy in one zone retires the oldest decoy, possibly in another zone.",
     "Single agent: any host. Zone agent: own zone only.", "−1 in the target host's zone; +50 per trap in the trapped host's zone."),
    ("restore_backup", "One host: cleaned at any stage, isolation ends.",
     "Removes an ESCALATED/BREACHED host as a pressure source for the next zone.",
     "Single agent: any host. Zone agent: own zone only.", "−50 downtime in the target host's zone."),
]
BCOLS = [("Action", 17), ("Direct effect (which zone)", 40), ("Effect on OTHER zones", 50), ("Who may use it", 27), ("Where the reward lands", 32)]


def sheet_zones():
    BW = sum(c[1] for c in BCOLS)
    fig, ax = canvas(20, 15.6, BW + 6, 156)
    ax.set_ylim(6.4, 156)
    fig.set_size_inches(20, 14.95)
    X0 = 3
    ax.text(X0, 152, "Cheat Sheet: which zone each action affects", fontsize=25, fontweight="bold", color=INK, va="center")
    ax.text(X0, 147.3, "Every action targets exactly ONE host, so its direct effect stays inside that host's zone. "
            "Zones interact only through lateral pressure and the shared honeypot limit.", fontsize=12.2, color=MUTE, va="center")
    # zone diagram
    zw, gap = 40, 14.5
    base = X0 + (BW - (3 * zw + 2 * gap)) / 2
    top, bot = 142, 117
    for zi, (zn, agent, hosts, fill, col) in enumerate(ZONES):
        x = base + zi * (zw + gap)
        ax.add_patch(FancyBboxPatch((x, bot), zw, top - bot, boxstyle="round,pad=0,rounding_size=1.5", fc=fill, ec=col, lw=2.2))
        ax.text(x + 2, top - 2.7, zn, fontsize=15, fontweight="bold", color=INK, va="center")
        ax.text(x + zw - 2, top - 2.7, agent, fontsize=10.5, color=MUTE, ha="right", va="center")
        for hi, (hn, ty) in enumerate(hosts):
            hx = x + 2 + hi * 12.6
            ax.add_patch(FancyBboxPatch((hx, top - 14.6), 11.6, 8.2, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=col, lw=1.4))
            ax.text(hx + 5.8, top - 9.2, hn, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
            ax.text(hx + 5.8, top - 12.2, ty, ha="center", va="center", fontsize=8.8, color=MUTE)
        notes = {0: "Entry zone: no lateral pressure comes in.", 1: "Receives pressure from the DMZ.",
                 2: "Receives pressure from Internal. Nothing downstream."}[zi]
        ax.text(x + zw / 2, bot + 3.1, notes, ha="center", va="center", fontsize=9.8, color=INK)
        if zi < 2:
            ax.annotate("", xy=(x + zw + gap - 1, 130), xytext=(x + zw + 1, 130),
                        arrowprops=dict(arrowstyle="-|>", color=STAGE[2], lw=3, mutation_scale=22))
            ax.text(x + zw + gap / 2, 135.2, "lateral\npressure", ha="center", va="center", fontsize=9.5, color=STAGE[2], fontweight="bold")
            ax.text(x + zw + gap / 2, 124.6, "×(1+0.6k)", ha="center", va="center", fontsize=9.2, color=STAGE[2])
    ax.text(X0, 110.6, "k = number of ESCALATED-or-worse hosts in the previous zone that are not isolated (counted up to 2), so landing odds in the next zone "
            "can reach ×2.2.  Zone agents see only their own zone plus a small shared summary.", fontsize=9.9, color=MUTE, va="center")
    # table
    y = 106.2
    x = X0
    for name, w in BCOLS:
        ax.add_patch(Rectangle((x, y - 4.4), w, 4.4, fc=INK, ec="white", lw=1))
        ax.text(x + 1, y - 2.2, name, color="white", fontsize=11.5, fontweight="bold", va="center")
        x += w
    y -= 4.4
    RH = 8.8
    for ri, (an, direct, other, who, rew) in enumerate(ROWS_B):
        t = y - ri * RH
        ax.add_patch(Rectangle((X0, t - RH), BW, RH, fc="white" if ri % 2 == 0 else "#f0f2f6", ec=LINE, lw=1))
        x = X0
        col = dict(ACT)[an]
        for ci, ((cn, w), txt) in enumerate(zip(BCOLS, [an, direct, other, who, rew])):
            if ci == 0:
                chip(ax, x + 0.7, t - RH + 1.5, w - 1.4, RH - 3, an, col, size=(10.4 if len(an) > 14 else 11.6))
            else:
                chars = {1: 47, 2: 59, 3: 31, 4: 37}[ci]
                bold = ci == 2 and (txt.startswith("Strongest") or txt.startswith("SHARED"))
                ax.text(x + 1, t - RH / 2, wrap(txt, chars), ha="left", va="center", fontsize=9.6,
                        color=(RED if bold else INK), fontweight=("bold" if bold else "normal"), linespacing=1.25)
            x += w
    yb = y - len(ROWS_B) * RH - 2.6
    # bottom: where it matters
    pw = (BW - 2 * 2) / 3
    ax.text(X0, yb, "Where each action tends to matter, inferred from the simulator's parameters (not tested zone by zone)",
            fontsize=12.5, fontweight="bold", color=INK, va="center")
    items = [
        ("DMZ", BLUE, ["DDoS targets servers and cloud, so redirect_traffic (DDoS landing ×0.1, 60% removal) fits here.",
                       "Cloud hosts breach faster (data theft ×1.5). Stopping an intruder here also stops lateral pressure into Internal."]),
        ("Internal", AMBER, ["Phishing targets workstations (weight 3.0 vs 0.1 elsewhere).",
                             "IoT (base vulnerability 0.75) and workstations (0.60) are the weakest hosts, so patch_system pays off most here."]),
        ("Data tier", PURPLE, ["Databases have the lowest base vulnerability (0.35) but breach faster once ESCALATED (data theft ×1.5).",
                              "Time-critical: restore_backup or isolate_machine early. It receives pressure from Internal and passes none on."]),
    ]
    for k, (zn, col, lines) in enumerate(items):
        x = X0 + k * (pw + 2)
        ax.add_patch(FancyBboxPatch((x, yb - 25), pw, 21.8, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=col, lw=1.8))
        ax.text(x + 1.6, yb - 7.2, zn, fontsize=12, fontweight="bold", color=col, va="center")
        for li, ln in enumerate(lines):
            ax.text(x + 1.6, yb - 13.0 - li * 7.4, "•  " + textwrap.fill(ln, 60, subsequent_indent="    "), fontsize=9.6, color=INK, va="center", linespacing=1.28)
    ax.text(X0 + BW, 8.6, "Source: env/core.py, env/multi_env.py. Base vulnerabilities vary ±0.10 per episode.", fontsize=9.3, color=MUTE, ha="right", va="center")
    fig.savefig("results/cheatsheet_zones.png", dpi=130, facecolor=BG)
    plt.close(fig)
    print("saved results/cheatsheet_zones.png")


if __name__ == "__main__":
    sheet_outcomes()
    sheet_zones()
