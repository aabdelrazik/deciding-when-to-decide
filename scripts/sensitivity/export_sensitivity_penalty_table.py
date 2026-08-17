"""Sensitivity of the winning-model margin to a partial penalty for fixed
temporal-regulation parameters.

Standard BIC charges nothing for a parameter that was fixed. That is fair when
the fixed value is structural (gamma=1 for no forgetting), but weaker for the
regulation parameters fixed to a value averaged from a related model's own fits,
since some information from the sample was used. This recomputes every
candidate's BIC charging each such parameter its exact cost and reports whether
the winner or its margin changes.

The cost is log(N_total), N_total being the pooled observation count, because
each fixed regulation value is a SINGLE number shared by all 105 subjects: one
population-level parameter estimated once. The earlier version of this analysis
used k_adj = k + 0.5 * n_fixed, which adds the parameter to every subject's own k
and so charges 0.5 * sum_i log(n_obs_i) -- the price of 105 distinct per-subject
parameters. That overstates the cost by a factor of about 60 and was enough to
flip the short-horizon winner on its own. No heuristic fraction is needed here:
the shared-parameter cost is exact.

Winners, margins and n_fixed all come from BIC/best_models.json, the comparison
CSV and the configs. The superseded hand-made version of this table still named
SBEXT-RPh---- and CB--TGRPhCL-- as the winners.

Usage (from scripts/):
    python3 export_sensitivity_penalty_table.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

REG = ("c_max", "urgency_coefficient", "urgency_slope")
HKEY = {"short": "S", "long": "L", "combined": "C"}
CFG_DIR = os.path.join(project_root, "data/simulation_configs")
OUT = os.path.join(project_root, "BIC", "tables", "sensitivity_analysis_table.tex")


def overrides(task):
    f = os.path.join(CFG_DIR, f"simulation_params_{task}.py")
    ns = {"__file__": f}
    exec(open(f).read(), ns)
    return ns["OVERRIDES"]


def n_fixed(task):
    """How many regulation parameters this model fixes rather than fits.

    A model with the regulation function switched off entirely (all three set to
    zero) is not charged: nothing about the sample informed those zeros.
    """
    o = overrides(task)
    pr = o.get("PARAM_RANGES", {})
    if all(o.get(k.upper(), None) == 0 for k in REG):
        return 0
    return sum(1 for k in REG if k not in pr)


def pooled_nobs(horizon):
    """N_total, the pooled observation count; identical for every model here."""
    with open(os.path.join(project_root, "BIC", "best_models.json")) as fh:
        task = json.load(fh)[horizon]
    sub = [] if horizon == "combined" else [horizon]
    df = pd.read_pickle(os.path.join(project_root, "data/POMDP", task, "de",
                                     *sub, "results.pkl"))
    tot = 0
    for _, row in df.iterrows():
        dd = row["data_dict_of_lists"]
        hs = dd.values() if isinstance(dd, dict) else [dd]
        tot += sum(len(s) for h in hs for s in h["draw_yellow_blue_action_outcome"].values)
    return tot


def main():
    with open(os.path.join(project_root, "BIC", "best_models.json")) as fh:
        best = json.load(fh)
    cmp = pd.read_csv(os.path.join(project_root, "BIC", "model_comparison_viper.csv"))
    cmp = cmp[cmp.fam == "full"]

    rows = []
    for horizon in ("short", "long", "combined"):
        d = cmp[cmp.horizon == HKEY[horizon]].copy()
        cost = np.log(pooled_nobs(horizon))   # BIC cost of one shared parameter
        d["nfix"] = [n_fixed(t) for t in d.task]
        d["BIC_adj"] = d.BIC + d.nfix * cost

        win = best[horizon]
        w = d[d.task == win].iloc[0]
        others = d[d.task != win]

        pri = others.sort_values("BIC").iloc[0]
        sen = others.sort_values("BIC_adj").iloc[0]
        wa = d[d.task == win].iloc[0]
        adj_winner = d.sort_values("BIC_adj").iloc[0].task

        rows.append(dict(horizon=horizon.capitalize(), winner=win, nfix=int(w.nfix),
                         cost=cost, pri_run=pri.task, d_pri=pri.BIC - w.BIC,
                         sen_run=sen.task, d_sen=sen.BIC_adj - wa.BIC_adj,
                         unchanged=(adj_winner == win)))
        print(f"{horizon:9} winner={win} (fixes {int(w.nfix)} @ {cost:.1f} BIC each)  "
              f"primary runner-up {pri.task} dBIC={pri.BIC - w.BIC:8.1f}  |  "
              f"adjusted runner-up {sen.task} dBIC={sen.BIC_adj - wa.BIC_adj:8.1f}  "
              f"{'winner unchanged' if adj_winner == win else 'WINNER CHANGES to ' + adj_winner}")

    tex = [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Horizon & Winner & Fixed regulation & Cost each & Runner-up & "
           r"$\Delta$BIC primary & $\Delta$BIC adjusted \\",
           r"\midrule"]
    for r in rows:
        tex.append(f"{r['horizon']} & $\\texttt{{{r['winner']}}}$ & {r['nfix']} & "
                   f"{r['cost']:.1f} & $\\texttt{{{r['pri_run']}}}$ & "
                   f"{r['d_pri']:.1f} & {r['d_sen']:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", "}", ""]
    open(OUT, "w").write("\n".join(tex))
    print(f"\nwrote {OUT}")
    json.dump(rows, open(OUT.replace(".tex", ".json"), "w"), indent=1, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r["d_pri"] for r in rows], w, label="Primary (no charge)",
           color="tab:blue", alpha=.85)
    ax.bar(x + w / 2, [r["d_sen"] for r in rows], w, label="Charged for fixed regulation",
           color="tab:orange", alpha=.85)
    ax.axhline(10, ls="--", lw=1.2, color="0.35")
    ax.text(len(rows) - 0.45, 11, r"$\Delta$BIC = 10 (very strong)", fontsize=8,
            color="0.35", ha="right")
    for i, r in enumerate(rows):
        ax.annotate(f"{r['d_pri']:.1f}", (i - w / 2, r["d_pri"]), ha="center",
                    va="bottom", fontsize=8)
        ax.annotate(f"{r['d_sen']:.1f}", (i + w / 2, r["d_sen"]), ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['horizon']}\n{r['winner']}\n({r['nfix']} fixed)"
                        for r in rows], fontsize=8)
    ax.set_ylabel(r"$\Delta$BIC over closest competitor")
    ax.set_title("Winning-model margin, with and without charging for fixed\n"
                 "temporal-regulation parameters", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    figdir = os.path.join(project_root, "BIC", "figures")
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(figdir, f"sensitivity_analysis_margins.{ext}"),
                    bbox_inches="tight", dpi=300)
    print(f"wrote {figdir}/sensitivity_analysis_margins.[pdf|png|svg]")


if __name__ == "__main__":
    main()
