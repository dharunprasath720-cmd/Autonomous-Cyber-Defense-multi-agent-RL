"""Paired comparisons between methods on the shared held-out seeds (in-distribution and shifted settings)."""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/eval_episodes.csv")
rows = []
for setting in df.setting.unique():
    d = df[df.setting == setting]
    piv = d.groupby(["method", "seed"]).agg(r=("reward", "mean"), b=("breached", "mean")).reset_index()
    for a, b in [("heuristic", "ppo"), ("heuristic_ma", "ippo"), ("heuristic_ma", "mappo"), ("ppo", "ippo"),
                 ("ppo", "mappo"), ("ippo", "mappo")]:
        x = piv[piv.method == a].set_index("seed").r; y = piv[piv.method == b].set_index("seed").r
        diff = (y - x).dropna(); t, p = stats.ttest_rel(y.loc[diff.index], x.loc[diff.index])
        w = stats.wilcoxon(diff).pvalue
        rows.append(dict(setting=setting, comparison=f"{b} - {a}", mean_diff=diff.mean(),
                         ci95=1.96 * diff.std(ddof=1) / np.sqrt(len(diff)), p_ttest=p, p_wilcoxon=w, n=len(diff)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.float_format", "{:.4g}".format)
print(out.to_string(index=False)); out.to_csv("results/paired_tests.csv", index=False)
