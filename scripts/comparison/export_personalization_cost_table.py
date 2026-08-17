"""Does personalizing model structure pay for its own selection cost?

Personalized selection picks, for each subject independently, whichever of the
candidate structures fits that subject best. That choice is itself estimated
from the subject's data, so it is an extra per-subject parameter: a discrete
index over the candidate set. The main comparison does not charge for it, which
flatters personalization.

Two accountings are reported, because charging a discrete choice as though it
were one continuous parameter is a convention, not a derivation:

  parameter  each subject is charged one extra parameter for their own choice,
             costing log(n_obs_i), so sum_i log(n_obs_i) in total
  selection  the choice is charged its information cost under a uniform prior
             over the M candidates for that horizon, 2*log(M) per subject

Personalization is worth doing only if the BIC it saves exceeds what the choice
costs. The two give similar magnitudes, so where they agree the conclusion does
not depend on which is adopted.

Note this is charged per subject, unlike the fixed regulation values in
export_sensitivity_penalty_table.py, which are single numbers shared by the whole
sample and so cost log(N_total) once. The difference is real: every subject makes
their own model choice, but they all share one phi_max.

Usage (from scripts/):
    python3 export_personalization_cost_table.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

HORIZONS = ("short", "long", "combined")


def per_subject_nobs(task, commit, horizon):
    """n_obs for each subject, under the same convention the fits used."""
    base = "data/POMDP_commit" if commit else "data/POMDP"
    sub = [] if horizon == "combined" else [horizon]
    df = pd.read_pickle(os.path.join(project_root, base, task, "de", *sub, "results.pkl"))
    out = []
    for _, row in df.iterrows():
        dd = row["data_dict_of_lists"]
        hs = dd.values() if isinstance(dd, dict) else [dd]
        n = sum(len(s) for h in hs for s in h["draw_yellow_blue_action_outcome"].values)
        if n:
            out.append(n)
    return np.array(out)


def n_candidates(commit, horizon):
    """How many structures the personalized selection could choose between."""
    key = {"short": "S", "long": "L", "combined": "C"}[horizon]
    d = pd.read_csv(os.path.join(project_root, "BIC", "model_comparison_viper.csv"))
    fam = "commit" if commit else "full"
    return int(d[(d.fam == fam) & (d.horizon == key)].task.nunique())


def parse_table(path):
    """Read the numbers back out of the generated personalized-selection table."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(Short|Long|Combined)\s*&\s*(\d+)\s*&\s*\$\\texttt\{([^}]+)\}\$"
                     r"\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(\d+)", line)
        if m:
            rows[m.group(1).lower()] = dict(
                n=int(m.group(2)), fixed_model=m.group(3), pers=float(m.group(4)),
                fixed=float(m.group(5)), gain=float(m.group(6)), improved=int(m.group(7)))
    return rows


def build(commit):
    tag = "_commit" if commit else ""
    tbl = os.path.join(project_root, "BIC_commit" if commit else "BIC", "tables",
                       f"personalized_model_selection{tag}.tex")
    rows = parse_table(tbl)
    if not rows:
        raise SystemExit(f"could not parse {tbl}")

    out = []
    for h in HORIZONS:
        r = rows[h]
        n_obs = per_subject_nobs(r["fixed_model"], commit, h)
        cost = float(np.log(n_obs).sum())
        M = n_candidates(commit, h)
        cost_sel = float(2 * np.log(M) * len(n_obs))
        net, net_sel = r["gain"] - cost, r["gain"] - cost_sel
        out.append(dict(horizon=h.capitalize(), fixed_model=r["fixed_model"],
                        pers=r["pers"], fixed=r["fixed"], gain=r["gain"],
                        cost=cost, net=net, M=M, cost_sel=cost_sel, net_sel=net_sel,
                        worth=net > 0, worth_sel=net_sel > 0))
        print(f"  {h:9} gain={r['gain']:7.1f} | parameter cost={cost:7.1f} "
              f"net={net:+7.1f} | selection cost={cost_sel:7.1f} (M={M}) "
              f"net={net_sel:+7.1f} | "
              f"{'pays' if net > 0 and net_sel > 0 else 'does not pay' if net <= 0 and net_sel <= 0 else 'BORDERLINE'}")
    return out


def table(rows):
    t = [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Horizon & Fixed model & $\Delta$BIC gained & \multicolumn{2}{c}{Parameter charge} & "
         r"\multicolumn{2}{c}{Selection charge} & Worth it? \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r" & & & Cost & Net & Cost ($M$) & Net & \\", r"\midrule"]
    for r in rows:
        both = r["worth"] and r["worth_sel"]
        nei = (not r["worth"]) and (not r["worth_sel"])
        verd = "Yes" if both else ("No" if nei else "Borderline")
        t.append(f"{r['horizon']} & $\\texttt{{{r['fixed_model']}}}$ & {r['gain']:.1f} & "
                 f"{r['cost']:.1f} & {r['net']:+.1f} & "
                 f"{r['cost_sel']:.1f} ({r['M']}) & {r['net_sel']:+.1f} & {verd} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", "}", ""]
    return "\n".join(t)


def figure(full, commit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, rows, title in ((axes[0], full, "Full likelihood"),
                            (axes[1], commit, "Commit likelihood")):
        x = np.arange(len(rows))
        w = 0.36
        ax.bar(x - w / 2, [r["gain"] for r in rows], w, label="BIC saved by personalizing",
               color="tab:green", alpha=.85)
        ax.bar(x + w / 2, [r["cost"] for r in rows], w, label="Cost: one parameter per subject",
               color="tab:red", alpha=.85)
        ax.plot(x + w / 2, [r["cost_sel"] for r in rows], "k_", markersize=22,
                markeredgewidth=2, label=r"Cost: $2\log M$ per subject")
        for i, r in enumerate(rows):
            ax.annotate(f"{r['gain']:.0f}", (i - w / 2, r["gain"]), ha="center",
                        va="bottom", fontsize=8)
            ax.annotate(f"{r['cost']:.0f}", (i + w / 2, r["cost"]), ha="center",
                        va="bottom", fontsize=8)
            ax.annotate(f"net {r['net']:+.0f}",
                        (i, max(r["gain"], r["cost"], r["cost_sel"]) * 1.10),
                        ha="center", fontsize=8,
                        color="tab:green" if r["worth"] else "tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels([r["horizon"] for r in rows])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.25)
        ax.margins(y=.26)
    axes[0].set_ylabel(r"$\Delta$BIC")
    # legend below the panels: inside the axes it sits on top of the net labels
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=3, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Does personalizing model structure pay for its own selection cost?",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=300)
    print(f"wrote {path}.[pdf|png|svg]")


def main():
    print("full likelihood:")
    full = build(commit=False)
    print("commit likelihood:")
    commit = build(commit=True)

    d = os.path.join(project_root, "BIC", "tables")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "personalization_cost.tex"), "w").write(table(full))
    open(os.path.join(d, "personalization_cost_commit.tex"), "w").write(table(commit))
    json.dump(dict(full=full, commit=commit),
              open(os.path.join(d, "personalization_cost.json"), "w"), indent=1)
    figure(full, commit, os.path.join(project_root, "BIC", "figures",
                                      "personalization_cost"))
    print(f"\nwrote {d}/personalization_cost[_commit].tex")


if __name__ == "__main__":
    main()
