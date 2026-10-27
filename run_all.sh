#!/usr/bin/env bash
# Reproduces every number/figure. Run from the repo root inside the venv. ~3-4 h on 8 logical cores
# (training dominates). Training steps run in parallel; set JOBS to match your core count.
set -e
PY=${PY:-python}
$PY sanity_check.py --episodes 20 --seed0 20000
$PY -m evaluation.tune_heuristic                       # writes agents/heuristic_tuned.json (tuning seeds only)
for s in 0 1 2; do
  $PY -m training.train_ppo   --reward norm --seed $s --name ppo_norm_s$s &
  $PY -m training.train_marl  --algo ippo   --seed $s --name ippo_s$s &
  $PY -m training.train_marl  --algo mappo  --seed $s --name mappo_s$s &
  $PY -m training.train_marl  --algo ippo   --seed $s --full-obs --name fullobs_ippo_s$s &
  $PY -m training.train_marl  --algo mappo  --seed $s --full-obs --name fullobs_mappo_s$s &
  wait
done
$PY -m training.train_ppo --reward raw --seed 0 --name ppo_raw_s0   # ablation A
$PY -m evaluation.evaluate --episodes 50
$PY -m evaluation.stats
$PY -m evaluation.plots_and_ablations
# Real data (download CICIDS2017 MachineLearningCSV.zip first, see README)
$PY -m data.cicids_loader
$PY -m evaluation.real_data_replay --policy mappo_s2 | tee results/real_data_mappo_s2_report.txt
$PY -m evaluation.real_data_replay --policy ppo_norm_s0 | tee results/real_data_ppo_norm_s0_report.txt
