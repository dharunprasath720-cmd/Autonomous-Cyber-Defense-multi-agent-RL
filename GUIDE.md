# Complete Guide: Autonomous Cyber Defense with Multi-Agent Reinforcement Learning

Read this first. It explains what the project is, what was found, how to open and run everything, and what every
technical term means. The formal write-up is `report/report.md`; this guide is the plain-language companion.

---

## 1. The project in one paragraph
We built a **simulated company network** (8 computers in 3 zones). A simulated **attacker (Red)** tries to break in,
spread, and steal data. A **defender (Blue)** is an AI that must notice and stop it, using only noisy warning signals.
We trained AI defenders with **reinforcement learning** (they learn by trial and error from rewards) and compared them
with a hand-written rule-based defender, and with doing nothing. We also tested them against a smarter attacker they had
never seen, and replayed real network traffic (CICIDS2017) through them to see how well simulator skills transfer.

## 2. The headline findings (honest version)
| Question | Answer |
|---|---|
| Do learned defenders beat doing nothing / random? | Yes, massively (random −114,000 reward vs learned about +2,300 to +2,600). |
| Do they beat the hand-written rule-based defender? | Yes in this simulator (heuristic −237). This does **not** match the published CAGE-4 result where heuristics won. |
| Does "coordinated" MARL (MAPPO) beat independent learners? | No. MAPPO ≈ independent PPO (not statistically different). |
| Does seeing the whole network (instead of one zone) help? | No, slightly worse (within noise). |
| Does it still work against a smarter attacker? | It degrades for everyone. Breach rate rises (e.g. PPO 0.49 → 0.89). |
| Does it work on real traffic? | Poorly. Policy F1 0.43 vs 0.68 for the rule-based defender (a "sim-to-real gap"). |
| Did reward-scale matter? | Hugely. Without normalization the learning completely stalls. |

**How to read this:** a true negative or null result (MAPPO not better, no transfer to real data) is a valid scientific
finding, as long as it is reported honestly and the limits are stated. See §9 for the limits.

## 3. What is in the folder
```
GUIDE.md                  <- this file
README.md                 <- setup + exact commands + headline table
report/report.md          <- the formal report (abstract ... conclusion)
showcase/demo.html        <- INTERACTIVE Red-vs-Blue replay (open this!)
env/                      <- the simulator (core.py) and the Gymnasium / PettingZoo wrappers
agents/                   <- rule-based heuristic + neural-network inference code
training/                 <- training scripts (PPO, IPPO, MAPPO)
evaluation/               <- evaluation, ablations, statistics, real-data replay, showcase builder
data/                     <- CICIDS2017 loader + 24,720-flow labeled subset
results/                  <- trained models, learning logs, tables (.csv), figures (.png)
sanity_check.py           <- 1-minute check that the simulator behaves sensibly
run_all.sh                <- the commands that regenerate everything (hours of compute)
```

## 4. How to open and use the interactive demo (`showcase/demo.html`)
1. Double-click `showcase/demo.html` (or right-click → Open with → Chrome/Edge). Use a wide window.
   Do **not** open `template.html`: it is the empty template and shows blank dropdowns.
2. Top bar:
   - **Blue (defender):** No defense / Rule-based heuristic / PPO / MAPPO.
   - **Episode seed:** two different pre-recorded games.
   - **▶ Play / ◀ ▶ / slider / Speed:** replay controls (Normal = 1.8 s per step; Slow = 3 s).
3. **Left, the network:** three zone boxes (DMZ, Internal, Data tier). Each circle is a host; its **colour is the true
   state** (green clean → yellow early intrusion → orange escalated → red breached). Dashed blue ring = isolated; ✓ =
   patched; ◆ = honeypot decoy; ⛔ = IP blocked; ⇄ = traffic redirected. The thin bar beside a host is the **anomaly score
   Blue actually sees** (it does not see the colour!). Lines: from RED (to a host) = an attack attempt (thick red =
   landed, dashed = failed/wasted, blue = trapped by a decoy); from BLUE = a defensive action.
4. **Right, two cards:** RED's attempts and BLUE's actions this step, each with a written explanation (what happened, the
   odds, what the defender could and could not see, and the reward effect).
   **Reward ledger (two cards side by side, under the map):** left = every reward rule that fired this step with its host,
   calculation and amount, summing to the step reward r_t (plus a by-zone split); right = the update G_after = G_before + r_t,
   and the running total per rule (detections, preventions, action costs, downtime, breach penalties).
5. **Bottom left:** reward timeline + clickable event log (click a line to jump there). **Bottom right:** all four
   defenders on the same episode side by side, plus the 50-episode numbers against the adaptive attacker.
6. **Good things to try:** (a) "No defense" with either attacker, to see how quickly everything is breached; (b) watch the ledger
   as MAPPO or PPO restores a host (+100 +50 −50 −1) and as 'No defense' accumulates −500 per breached host; (c) the heuristic vs PPO,
   noting how the heuristic isolates then restores while PPO often restores straight away.

## 5. How to run things yourself (Windows, PowerShell, from the project folder)
```bash
cd cyberdefense-marl   # the project folder

# 1-minute simulator check (no-op, random, heuristic)
.venv\Scripts\python sanity_check.py --episodes 20 --seed0 20000

# Re-print the full 5-method comparison from the saved models (a few minutes)
.venv\Scripts\python -m evaluation.evaluate --episodes 50

# Re-make learning curves + both ablations
.venv\Scripts\python -m evaluation.plots_and_ablations

# Rebuild the interactive demo
.venv\Scripts\python -m evaluation.showcase_traces
.venv\Scripts\python showcase\build_demo.py
```
Training from scratch is in `README.md` / `run_all.sh` (hours on a normal PC; results vary slightly between machines).

## 6. What each result file is
| File | Meaning |
|---|---|
| `results/comparison_table.csv` | Main table: 5 methods + 2 reference rows × reward, breach rate, TTD, TTC, downtime. |
| `results/generalization_table.csv` | Reward and breach rate under the two distribution shifts. |
| `results/paired_tests.csv` | Statistical tests between methods on the same seeds. |
| `results/ablation_*.csv` | Reward-scale ablation and observation-design ablation. |
| `results/fig_*.png` | Bar charts, learning curves, ablation figures. |
| `results/real_data_*` | Offline CICIDS2017 replay: confusion-style table, per-label rates, summary. |
| `results/adaptive_red_eval.txt` | 50-episode results against the adaptive attacker. |
| `results/logs/*.csv` | Per-update training logs (learning curves). |

## 7. How the simulation works (plain language)
* **Hosts and zones.** 8 hosts: DMZ (server, cloud, server), Internal (workstation, workstation, IoT), Data tier (database,
  database). An escalated host in one zone makes the *next* zone easier to attack ("lateral movement").
* **Compromise stages.** A host goes clean → *early intrusion* (attacker has a foothold) → *escalated* (attacker gained
  higher privileges) → *breached* (data stolen / service impacted). Each step is probabilistic and depends on how
  vulnerable the host is and whether it is patched.
* **What Blue sees.** Per host: type, noisy traffic-anomaly and user-anomaly scores (smoothed, with occasional false
  alarms), vulnerability, and its own flags; plus one noisy global threat level. It never sees the true stage.
* **Blue's actions** (one host + one action per agent per step): no_op, block_ip, isolate_machine, patch_system,
  redirect_traffic, deploy_honeypot, restore_backup. Each really changes the simulation (see the playbook in the demo).
* **Reward (the score Blue is trained to maximize):** +100 for reversing an attack before breach; +50 for early
  detection; −500 per breached host per step; −200 for isolating (downtime); −50 for restoring (design choice of ours);
  −1 per non-trivial action.
* **Episode.** 100 steps. Everything is repeatable from a *seed*.

## 8. Why the two kinds of agents differ
* **Single-agent PPO / heuristic:** one decision-maker sees the whole network and acts once per step.
* **Multi-agent (IPPO, MAPPO):** three agents, one per zone, each seeing only its own zone plus a tiny shared summary,
  each acting once per step: so **3 actions per step** instead of 1. That is why the table has extra reference rows
  (random and heuristic with 3 actions per step) to make the comparison fair.

## 9. Limitations you should state out loud
1. The simulator is our own and fairly simple; results are not comparable to CybORG/CAGE scores.
2. The default attacker is a fixed random process, so there is no real adversarial back-and-forth in the main results.
3. The "adaptive" attacker is hand-written and was designed after seeing the policies, so it is a stress test, not a neutral one.
4. Only 3 training seeds per method; several MARL runs stopped at the 400,000-step minimum.
5. Reward is partly "gameable": more attacks can mean more reward. Prefer breach rate as the security metric.
6. The real-data replay uses weak proxy features and an assumed label-to-stage mapping; it measures a gap, not
   real-world performance. No real systems were touched.
7. The CybORG/CAGE citations in the report are written from memory and must be checked before submission.

## 10. Glossary (every technical term used)
**Core ideas**
* **Reinforcement learning (RL):** a program (the *agent*) learns to act by trying things and receiving numeric *rewards*.
* **Agent:** the decision-maker (here: a defender, or the attacker).
* **Environment:** the world the agent acts in (here: the simulated network).
* **Policy:** the agent's rule for choosing an action from what it observes. Here, a small neural network or hand-written rules.
* **Episode:** one complete game (100 steps). **Step:** one decision round.
* **Reward / return:** the score for a step / the total score over an episode.
* **Observation vs state:** what the agent sees vs the full hidden truth.
* **POMDP (partially observable Markov decision process):** a decision problem where the agent only gets noisy,
  incomplete observations. Our defender is in one.
* **Seed:** a number that fixes the random numbers, so an episode can be repeated exactly.
* **Stochastic:** involving randomness. **Deterministic:** no randomness (same input, same output).
* **Held-out seeds:** episodes never used during training or tuning; used only for the final tests.

**Algorithms and training**
* **PPO (Proximal Policy Optimization):** a popular RL algorithm that improves a policy in small, safe steps.
* **Stable-Baselines3 (SB3):** a Python library with ready-made PPO. **Gymnasium:** the standard interface for single-agent
  environments. **PettingZoo:** the standard interface for multi-agent environments (`ParallelEnv` = all agents act at once).
* **MARL (multi-agent RL):** several learning agents in one environment.
* **IPPO (independent PPO):** each agent learns on its own, with its own network and critic, no sharing.
* **MAPPO (multi-agent PPO):** each agent keeps its own *actor*, but they share one *critic* that sees the whole state during training.
* **Actor / critic:** the actor chooses actions; the critic estimates how good a situation is, which guides learning.
* **CTDE (centralized training, decentralized execution):** a central critic helps in training, but at run time each agent acts alone on its own observation.
* **Parameter sharing:** using the same network weights for several agents. We do **not** do this.
* **Neural network / MLP:** the function approximator (here two hidden layers of 128 units) that maps observations to actions.
* **Hyper-parameters:** settings such as learning rate (3e-4), discount γ (0.99), GAE λ (0.95), clip range (0.2).
* **GAE (generalized advantage estimation):** a way to estimate how much better an action was than average.
* **Clip fraction / approx KL:** training diagnostics showing how much the policy changed per update. **Value loss / explained
  variance:** how well the critic predicts returns. Near zero change = learning is stuck.
* **Entropy bonus:** a small reward for staying exploratory so the policy doesn't collapse too early.
* **Plateau stopping:** we stop training when the smoothed learning curve stops improving, not at a fixed step count.
* **VecNormalize / reward normalization:** rescales rewards and observations so values are of order 1. Needed because
  raw rewards of −500 per step overwhelmed learning.
* **Greedy / deterministic policy:** at test time the agent always picks its most likely action.
* **Overfitting (heuristic tuning):** fitting the thresholds so well to 20 tuning seeds that held-out performance is worse.

**Cyber terms**
* **Blue / Red team:** defenders / attackers. **Host:** a computer or device on the network. **Zone / segment:** a part of the network (DMZ, internal, data tier).
* **DMZ:** the exposed, internet-facing zone. **Data tier:** where databases live.
* **Malware:** malicious software. **Phishing:** tricking a user into giving access. **DDoS:** flooding a service with
  traffic to knock it out. **Privilege escalation:** an attacker gaining higher access rights. **Data theft / exfiltration:**
  stealing data. **Lateral movement:** spreading from one machine/zone to another.
* **Foothold:** an attacker's first working access on a host. **Breach:** the attacker achieves its goal on a host.
* **Vulnerability / patch / CVE:** a weakness / a fix for it / a publicly catalogued weakness.
* **Block IP:** refuse traffic from an attacker's address. **Isolate:** disconnect a host from the network. **Redirect /
  scrubbing:** send traffic through a filter. **Honeypot:** a decoy system built to attract and expose attackers.
  **Restore from backup:** wipe a host back to a known-good state.
* **Anomaly score:** how unusual a host's traffic/user behaviour looks (0 normal, 1 very unusual). Here simulated as noisy signals.
* **False alarm / false positive:** raising an alert when nothing is wrong. **Downtime:** service unavailable because of a defensive action.

**Metrics**
* **Breach rate:** fraction of episodes with at least one breached host-step.
* **Breach-steps per episode:** total (host, step) pairs spent breached.
* **TTD (time to detect):** steps from attack landing to the defender's first relevant action on that host.
* **TTC (time to contain):** steps from detection to the host being clean again.
* **Detect rate / contain rate:** fraction of intrusions detected / contained.
* **Downtime cost:** total penalty from isolating and restoring.
* **Mean ± std:** average and spread. **Across-seed std:** spread between separately trained models.
* **Paired test (t-test / Wilcoxon):** compares two methods on the same episodes. **p-value:** chance of seeing a
  difference this large by luck if there were no real difference; below 0.05 is usually called significant.
* **Confusion matrix:** a table of actual class vs predicted class. **TP/FP/FN/TN:** true/false positive/negative.
  **Precision:** of the alarms raised, how many were real. **Recall:** of the real attacks, how many were caught.
  **F1:** a single score combining precision and recall. **FAR (false-alarm rate):** FP / (FP + TN).

**Experiment design**
* **Baseline:** a reference method to compare against (random, no-op, heuristic).
* **Heuristic:** hand-written rules (here: thresholds on the anomaly score).
* **Ablation:** removing or changing one component to measure its effect.
* **Generalization / distribution shift:** testing on conditions different from training (higher vulnerability;
  attacks concentrated on zone gateways; an adaptive attacker).
* **Sim-to-real gap:** the drop in performance when moving from a simulator to real data.
* **Offline / replay inference:** feeding recorded data to a frozen model and logging what it would do, without acting or retraining.
* **Capacity-matched baseline:** a baseline given the same number of actions per step as the method it is compared with.
* **CICIDS2017:** a public labeled dataset of real network flows with benign traffic and attacks (DoS, DDoS, port scan,
  brute force, web attacks, infiltration, botnet, Heartbleed). **Flow:** a summary of one network connection.
* **CICFlowMeter:** the tool that produced the flow features in CICIDS2017.
* **ECDF:** empirical cumulative distribution function, used to measure how unusual a value is compared with benign traffic.
* **CybORG / CAGE Challenge:** a cyber-operations gym and its public competitions; CAGE-4 was multi-agent. Used here as
  related work, not reproduced.
* **Reproducibility:** others can rerun the same commands and get the same (or statistically similar) numbers.

## 11. If your professor asks...
* *"Why does RL beat the heuristic here but not in CAGE-4?"* Our attacker is a fixed random process and our rewards give
  clear signals; CAGE-4 is far richer and its heuristics were expert-engineered. We make no general claim.
* *"Why is MAPPO not better?"* The simulator has weak cross-zone coupling, and the ablation shows full visibility doesn't
  help, so there is little for coordination to exploit.
* *"Is the 3-actions-per-step comparison fair?"* We included capacity-matched random/heuristic rows for exactly that reason.
* *"Is the real-data result a deployment claim?"* No. It is an offline gap analysis using weak proxy features.
* *"What would you do next?"* A learning (self-play) attacker, recurrent policies, stronger cross-zone coupling,
  realistic temporal attack chains, more seeds, and re-running in CybORG.
