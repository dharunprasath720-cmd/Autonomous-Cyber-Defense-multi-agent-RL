# Autonomous Cyber Defense Using Multi-Agent Reinforcement Learning

**Live demo:** https://dharunprasath720-cmd.github.io/Autonomous-Cyber-Defense-multi-agent-RL/ (interactive Red-vs-Blue replay with a live reward ledger; source: `index.html`, built by `showcase/build_demo.py`).

A from-scratch, reproducible study: a stochastic enterprise-network defense simulator (single- and
multi-agent), a tuned heuristic, SB3 PPO, independent PPO, MAPPO, a 5-method evaluation suite with
generalization tests, two ablations, and an **offline** replay of real CICIDS2017 flows.
Full write-up: [`report/report.md`](report/report.md).

## Interactive showcase (Red vs Blue)
Open `index.html` by double-clicking it (self-contained, works offline). Pick the defender (none / heuristic / PPO /
MAPPO) and an episode, then press Play. Each step shows the real attack attempts, the defender's actions, why they
succeeded or failed, and a side-by-side reward ledger (calculation of this step's reward, and the running-score update). Rebuild with:
```bash
python -m evaluation.showcase_traces   # records traces from the trained policies -> showcase/traces.json
python showcase/build_demo.py          # -> index.html  (showcase/template.html is only the unfilled template)
```

## Layout
```
env/         core.py (shared simulator), cyber_env.py (Gymnasium), multi_env.py (PettingZoo ParallelEnv)
agents/      heuristic.py (+ tuned thresholds json), marl.py (actor nets / inference wrapper)
training/    train_ppo.py (SB3), train_marl.py (custom IPPO + MAPPO), utils.py (plateau stopping, seeds)
evaluation/  evaluate.py, policies.py, stats.py, plots_and_ablations.py, tune_heuristic.py, real_data_replay.py
data/        cicids_loader.py (+ cicids_subset.csv, the 24.7k-flow labeled subset; raw download is git-ignored)
results/     models, logs (learning curves), tables (*.csv), figures (*.png), replay reports
sanity_check.py   run_all.sh   requirements.txt   requirements-lock.txt (pip freeze)
```

## Setup (Python 3.14 used; any recent 3.10+ should work)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # Windows;  use .venv/bin/python elsewhere
# exact versions used: pip install -r requirements-lock.txt
```
Key versions: torch 2.14.1, gymnasium 1.4.0, stable-baselines3 2.9.0, pettingzoo 1.27.0, numpy 2.5.3,
pandas 3.0.6, matplotlib 3.11.2, scikit-learn 1.9.1, scipy 1.18.1 (sb3-contrib 2.9.0 installed but unused:
it has no MAPPO, so MAPPO is a minimal custom implementation in `training/train_marl.py`).

## Reproduce (exact commands; `run_all.sh` runs all of them)
```bash
python sanity_check.py --episodes 20 --seed0 20000             # step 1/2: no-op, random, heuristic
python -m evaluation.tune_heuristic                              # step 2: threshold grid search (tuning seeds)
python -m training.train_ppo  --reward norm --seed 0 --name ppo_norm_s0    # step 3 (seeds 0,1,2)
python -m training.train_ppo  --reward raw  --seed 0 --name ppo_raw_s0     # ablation A
python -m training.train_marl --algo ippo  --seed 0 --name ippo_s0         # step 5 (seeds 0,1,2)
python -m training.train_marl --algo mappo --seed 0 --name mappo_s0        # step 5 (seeds 0,1,2)
python -m training.train_marl --algo ippo  --seed 0 --full-obs --name fullobs_ippo_s0   # ablation B (also mappo; seeds 0,1,2)
python -m evaluation.evaluate --episodes 50                      # step 6 -> results/comparison_table.csv, figures
python -m evaluation.stats                                       # paired tests
python -m evaluation.plots_and_ablations                         # learning curves + both ablations
# step 7 (real data) -- download first (235 MB, third-party mirror of the UNB release):
mkdir -p data/raw && curl -L -o data/raw/MachineLearningCSV.zip \
  https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/MachineLearningCSV.zip
python -I -c "import zipfile; zipfile.ZipFile('data/raw/MachineLearningCSV.zip').extractall('data/raw/extracted')"
python -m data.cicids_loader
python -m evaluation.real_data_replay --policy mappo_s2
```
Training stops on a plateau (no >1 % improvement of the 10-update moving average for 40 updates, after
a 400k-step minimum), not a fixed step count. Training is not bit-reproducible across machines
(torch/BLAS threading); simulator runs are deterministic given a seed. Expect numbers within seed noise.

## Seeds
| Purpose | Seeds |
|---|---|
| Training episodes | `100000 + 1_000_000*run_seed + k`, run_seed ∈ {0,1,2} (torch/numpy seed = run_seed) |
| Heuristic tuning | `5_000_000 + k`, k=0..19 |
| Held-out evaluation (all methods, all tables) | `10_000_000 + k`, k=0..49 |
| Sanity check | 20000..20019 |
| Replay stream construction | numpy seed 0 |

## Headline results (50 held-out seeds; trained methods pool 3 training seeds)
| method | actions/step | reward mean ± std | breach rate | breach-steps/ep | TTD | TTC | downtime cost |
|---|---|---|---|---|---|---|---|
| random | 1 | -114,204 ± 40,933 | 1.00 | 222.4 | 12.1 | 13.9 | 3,603 |
| heuristic | 1 | -237 ± 2,067 | 0.70 | 3.2 | 2.3 | 1.8 | 769 |
| PPO (SB3 + VecNormalize) | 1 | 2,336 ± 826 | 0.49 | 0.81 | 2.0 | 0.07 | 1,617 |
| independent PPO | 3 | 2,617 ± 802 | 0.43 | 0.57 | 1.6 | 0.28 | 1,321 |
| MAPPO | 3 | 2,522 ± 875 | 0.47 | 0.72 | 2.0 | 0.20 | 1,194 |
| *random (3 actions/step)* | 3 | -34,882 ± 19,337 | 0.96 | 51.9 | 4.7 | 8.9 | 9,569 |
| *heuristic (per zone, 3 actions/step)* | 3 | 202 ± 2,032 | 0.72 | 2.6 | 2.0 | 1.5 | 623 |

Everything else (generalization, ablations, real-data table) is in the report and `results/`.
