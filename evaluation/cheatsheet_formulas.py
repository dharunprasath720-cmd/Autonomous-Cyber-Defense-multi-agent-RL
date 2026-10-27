"""results/cheatsheet_formulas.png: every formula behind the demo (rewards, attacks, observations, learning).
Constants mirror env/core.py and training/*.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTE, LINE, BG = "#15181d", "#5f6672", "#dfe2e6", "#f5f5f6"
RED, BLUE, GREEN, AMBER, PURPLE, TEAL = "#c4405a", "#2f6db3", "#2c8559", "#b9820a", "#6f4fb0", "#0b7285"
plt.rcParams["mathtext.fontset"] = "dejavusans"

W, H = 200, 173
fig = plt.figure(figsize=(20, 17.3), dpi=130, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

ax.text(4, H - 5, "Formula Cheat Sheet: everything behind the demo", fontsize=25, fontweight="bold", color=INK, va="center")
ax.text(4, H - 10.3, "Symbols: v = host vulnerability, θ = threat level, k = escalated hosts in the previous zone, "
        "G = running score (return), r = reward at one step.", fontsize=12, color=MUTE, va="center")

ROW = 7.1


def card(x, y_top, w, title, color, rows):
    h = 8.5 + len(rows) * ROW + 1.5
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc="white", ec=LINE, lw=1.4))
    ax.add_patch(FancyBboxPatch((x, y_top - 6.2), w, 6.2, boxstyle="round,pad=0,rounding_size=1.4", fc=color, ec="none"))
    ax.add_patch(plt.Rectangle((x, y_top - 6.2), w, 3, fc=color, ec="none"))
    ax.text(x + 2.2, y_top - 3.1, title, fontsize=13.5, fontweight="bold", color="white", va="center")
    y = y_top - 11.2
    for math, note in rows:
        ax.text(x + 2.4, y, math, fontsize=13.2, color=INK, va="center")
        ax.text(x + 2.6, y - 3.3, note, fontsize=9.6, color=MUTE, va="center")
        y -= ROW
    return y_top - h


LX, RX, CW = 4, 103, 93

# ---------------------------------------------------------------- left column
y = H - 15
y = card(LX, y, CW, "1  Reward and running score (the ledger)", RED, [
    (r"$r_t=\mathrm{(sum\ of\ the\ amounts\ of\ all\ rules\ that\ fired\ this\ step)}$", "Nothing else is added: the ledger lists every line."),
    (r"$r_t=100\,n_{rev}+50\,n_{det}+50\,n_{trap}-1\,n_{act}-200\,n_{iso}-50\,n_{res}-500\,k_t$",
     "n = times each rule fired this step; k_t = hosts breached at the end of the step."),
    (r"$G_t=G_{t-1}+r_t$", "Running score shown as before + this step = after."),
    (r"$r_t=r_t^{\mathrm{DMZ}}+r_t^{\mathrm{Internal}}+r_t^{\mathrm{Data}}$", "Zone rewards sum to the team reward (MAPPO trains on this sum)."),
    (r"$G_t=\sum_{l\geq 0}\gamma^{\,l}\,r_{t+l},\quad \gamma=0.99$", "Discounted return the learner tries to maximise (training only)."),
    (r"$\mathrm{restore\ on\ early\ intruder:}\ \ +50+100-1-50=+99$", "Demo step 5: G goes from −6 to −6 + 99 = +93."),
]) - 3.2
y = card(LX, y, CW, "3  Escalation: how an intruder progresses", AMBER, [
    (r"$p_{E\rightarrow S}=0.30\,(0.4+0.6\,v)\,\pi\,m_{vec}\,m_{hp}$", "Early → escalated, rolled every step the host is not isolated."),
    (r"$p_{S\rightarrow B}=0.25\,(0.4+0.6\,v)\,\pi\,m_{vec}\,m_{type}\,m_{hp}$", "Escalated → breached."),
    (r"$\pi=0.6\ \mathrm{if\ patched},\quad m_{hp}=0.8\ \mathrm{if\ honeypot}$", "m_vec: phishing 1.2 or DDoS 0.6 (early); DDoS 1.4 (escalated)."),
    (r"$m_{type}=1.5\ \mathrm{for\ database\ and\ cloud\ hosts}$", "Isolated host: p = 0 for the 5 steps."),
    (r"$v_{base}$: server .45, workstation .60, database .35, cloud .40, IoT .75", "Each episode adds random ±0.10 to every host."),
    (r"$\mathrm{patch:}\ v\leftarrow 0.6\,v;\quad \mathrm{unpatch\ chance}=2\%/\mathrm{step}$", "A new CVE also adds +0.15 to a host's v with 1% chance per step."),
]) - 3.2
y = card(LX, y, CW, "5  What each action does (probabilities)", BLUE, [
    (r"$\mathrm{block\_ip:}\ \ P(\mathrm{early}\rightarrow\mathrm{clean})=0.70,\ \ P(\mathrm{esc}\rightarrow\mathrm{early})=0.30$", "Also landing chance on that host ×0.25 for 5 steps."),
    (r"$\mathrm{redirect:}\ P(\mathrm{early\rightarrow clean})=0.60\ (\mathrm{DDoS}),\ 0.15\ (\mathrm{other});\ \ 0.20\ \mathrm{esc\rightarrow early}\ (\mathrm{DDoS})$", "Landing ×0.6 (×0.1 for DDoS) for 4 steps."),
    (r"$\mathrm{honeypot:}\ P(\mathrm{trap})=0.80,\ P(\mathrm{expose\ intruder})=0.50$", "Lasts 20 steps; at most 3 decoys network-wide."),
    (r"$E[\mathrm{block\_ip\ on\ early\ intruder}]=0.7\,(+149)+0.3\,(+49)=+119$", "+149 = +50 detect +100 reversed −1; +49 = +50 −1."),
])
left_bottom = y

# ---------------------------------------------------------------- right column
y = H - 15
y = card(RX, y, CW, "2  Attack arrival and landing", RED, [
    (r"$N\sim\mathrm{Poisson}(1.5\,(0.3+1.4\,\theta))$", "Attack attempts per step. θ = 0.25 (calm) or 0.85 (campaign)."),
    (r"$P(\mathrm{calm}\rightarrow\mathrm{campaign})=0.05,\quad P(\mathrm{campaign}\rightarrow\mathrm{calm})=0.10$", "Gives bursts of activity."),
    (r"$P(\mathrm{land})=\min\{0.95,\ b_{vec}\cdot v_{eff}\cdot Z\cdot m_{block}\cdot m_{redir}\}$", "b: malware 0.55, phishing 0.65, DDoS 0.50."),
    (r"$v_{eff}=v\cdot(0.5\ \mathrm{if\ patched}),\quad m_{block}=0.25,\quad m_{redir}=0.6\ (0.1\ \mathrm{DDoS})$", "Defences multiply the landing chance down."),
    (r"$Z=1+0.6\,\min(2,\,k)$", "Lateral pressure: up to ×2.2 in the next zone."),
    (r"$P(\mathrm{vector})=\{\mathrm{malware}\ .45,\ \mathrm{phishing}\ .35,\ \mathrm{DDoS}\ .20\}$", "Target host is drawn by weight per vector (phishing → workstations)."),
]) - 3.2
y = card(RX, y, CW, "4  What the defender sees (noisy, smoothed)", TEAL, [
    (r"$x_t=b_{stage}+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\,0.22^2)$", "b (traffic): .10 / .30 / .50 / .70;  b (user): .10 / .35 / .55 / .75 for clean → breached."),
    (r"$s_t=0.6\,x_t+0.4\,s_{t-1}$", "Sensor smoothing (moving average); the bar beside each host."),
    (r"$a=\frac{1}{2}\,(s^{traffic}+s^{user})$", "Alert score used by the heuristic and shown as the anomaly bar."),
    (r"$+0.4\ \mathrm{spike\ with\ 4\%\ false\ alarm}$", "DDoS: traffic +0.25; phishing: user +0.20; decoy: noise ×0.5."),
    (r"$\mathrm{heuristic:}\ a\geq0.30\rightarrow\mathrm{block},\ a\geq0.54\rightarrow\mathrm{isolate},\ a\geq0.58\rightarrow\mathrm{restore}$", "Thresholds tuned on separate seeds; patch if v ≥ 0.45."),
]) - 3.2
y = card(RX, y, CW, "6  How the agent learns (PPO / MAPPO)", PURPLE, [
    (r"$\tilde r_t=\mathrm{clip}\left(r_t/\sigma_G,\ \pm10\right)$", "σ_G = running std of the discounted return: −500 becomes about −1."),
    (r"$\delta_t=\tilde r_t+\gamma V(s_{t+1})-V(s_t)$", "How much better the step went than the critic expected."),
    (r"$A_t=\sum_{l\geq0}(\gamma\lambda)^{l}\,\delta_{t+l},\quad \gamma=0.99,\ \lambda=0.95$", "Advantage (GAE)."),
    (r"$L=\mathrm{E}\left[\min\left(\rho_t A_t,\ \mathrm{clip}(\rho_t,\,0.8,\,1.2)\,A_t\right)\right]$", "ρ = new policy probability / old policy probability of the same action."),
    (r"$\mathrm{MAPPO:}\ V(s)\ \mathrm{sees\ the\ global\ state};\ \ \mathrm{IPPO:}\ V\ \mathrm{sees\ only\ its\ own\ zone}$", "Actors always act on their own zone's observation."),
])

ax.text(W - 4, 2.2, "Source: env/core.py, training/train_ppo.py, training/train_marl.py. Values are the simulator's defaults.",
        fontsize=9.5, color=MUTE, ha="right", va="center")
fig.savefig("results/cheatsheet_formulas.png", dpi=130, facecolor=BG)
print("saved results/cheatsheet_formulas.png; bottoms:", round(left_bottom, 1), round(y, 1))
