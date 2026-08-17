"""Protected exceedance probabilities over the individualized models.

Random-effects model comparison (Stephan et al. 2009; Rigoux et al. 2014) run on
the per-subject BIC matrix that per_subject_model_selection.py writes. Log
evidence is approximated by -BIC/2.

Two levels, because they answer different questions:

  model   every candidate in that horizon. With 105 subjects spread over 22 to
          31 models this is usually diffuse, and the Bayesian omnibus risk says
          so rather than letting an exceedance probability near 1 stand.
  family  the candidates collapsed to the mechanism contrast the manuscript
          argues about: exaggeration, forgetting, or neither. Summing evidence
          within a family is the standard remedy for a large model space and is
          far better powered.

This asks a different question from the personalization-cost analysis. That one
asks whether personalizing pays for the cost of selecting a model per subject;
this asks whether the population is heterogeneous in which model it uses. They
can disagree, because PXP carries no penalty for the selection itself.

Usage (from scripts/):
    SIM_ALGORITHM=de python3 export_pxp_table.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(R)

from src.stats.rfx_bms import protected_exceedance_probability, validate_rfx_bms

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

COMMIT = os.environ.get("POMDP_SEEDS_COMMIT", "") == "1"
SUBDIR = "POMDP_commit" if COMMIT else "POMDP"
OUTDIR = os.path.join(R, "BIC_commit" if COMMIT else "BIC")
HORIZON = {"S": "Short", "L": "Long", "C": "Combined"}
TOP = int(os.environ.get("PXP_TOP", "5"))


def family_of(task: str) -> str:
    """Which recency mechanism the model carries, from its name.

    Slot order after the horizon letter is B E X T G R P H C L U K, so slot 2 is
    the exaggeration factor and slot 5 the forgetting factor. The two are never
    free together.
    """
    body = task[1:]
    if body[4] == "G":
        return "Forgetting"
    if body[1] == "E":
        return "Exaggeration"
    return "Neither"


def load(prefix: str) -> pd.DataFrame:
    f = os.path.join(R, "data", SUBDIR, f"per_subject_bic_matrix_{prefix}.csv")
    if not os.path.exists(f):
        raise SystemExit(f"{f} missing; run per_subject_model_selection.py first")
    return pd.read_csv(f, index_col="subject_ID")


def main():
    print("validating the RFX implementation before using it:")
    if not validate_rfx_bms():
        raise SystemExit("RFX validation failed; not reporting numbers")
    print()

    rows, fam_rows, summary = [], [], []
    for prefix, hname in HORIZON.items():
        bic = load(prefix).dropna(axis=0, how="any")
        L = -bic.values / 2.0                       # BIC -> log evidence
        tasks = list(bic.columns)

        nparams = {}
        for t_ in tasks:
            f_ = os.path.join(R, "data/simulation_configs",
                              f"simulation_params_{t_}{'_commit' if COMMIT else ''}.py")
            ns_ = {"__file__": f_}
            exec(open(f_).read(), ns_)
            nparams[t_] = len(ns_["OVERRIDES"]["PARAM_RANGES"])

        res = protected_exceedance_probability(L)
        order = np.argsort(-res["pxp"])
        print(f"{hname}: {bic.shape[0]} subjects x {bic.shape[1]} models   "
              f"BOR={res['bor']:.3f}")
        for j in order[:TOP]:
            print(f"    {tasks[j]:16} r={res['r'][j]:.3f}  xp={res['xp'][j]:.3f}  "
                  f"pxp={res['pxp'][j]:.3f}")
        for j in order:
            rows.append(dict(horizon=hname, task=tasks[j], k=int(nparams.get(tasks[j], 0)),
                             r=res["r"][j], xp=res["xp"][j], pxp=res["pxp"][j],
                             bor=res["bor"]))

        # family level: sum evidence within each mechanism family
        fam = pd.Series([family_of(t) for t in tasks], index=tasks)
        names = ["Exaggeration", "Forgetting", "Neither"]
        names = [f for f in names if (fam == f).any()]
        Lf = np.column_stack([
            # log-sum-exp over the family's members, per subject
            (lambda M: M.max(axis=1) + np.log(np.exp(M - M.max(axis=1, keepdims=True)).sum(axis=1)))
            (L[:, [i for i, t in enumerate(tasks) if fam[t] == f]])
            for f in names])
        # label each family by its best-fitting member, so the row names a model
        # the reader already knows rather than an abstract mechanism class
        best_in_fam = {}
        for f in names:
            members = [t_ for t_ in tasks if fam[t_] == f]
            best_in_fam[f] = min(members, key=lambda t_: bic[t_].sum())

        rf = protected_exceedance_probability(Lf)
        print(f"    family level ({', '.join(names)})   BOR={rf['bor']:.3f}")
        for j, f in enumerate(names):
            print(f"      {best_in_fam[f]:16} ({f:12}) r={rf['r'][j]:.3f}  "
                  f"xp={rf['xp'][j]:.3f}  pxp={rf['pxp'][j]:.3f}")
            fam_rows.append(dict(horizon=hname, family=f, best_model=best_in_fam[f],
                                 n_models=int((fam == f).sum()),
                                 r=rf["r"][j], xp=rf["xp"][j], pxp=rf["pxp"][j],
                                 bor=rf["bor"]))
        summary.append(dict(horizon=hname, n_subjects=int(bic.shape[0]),
                            n_models=int(bic.shape[1]), bor_model=res["bor"],
                            bor_family=rf["bor"]))
        print()

    d = os.path.join(OUTDIR, "tables")
    os.makedirs(d, exist_ok=True)
    df, fdf = pd.DataFrame(rows), pd.DataFrame(fam_rows)
    tag = "_commit" if COMMIT else ""
    json.dump(dict(model=rows, family=fam_rows, summary=summary),
              open(os.path.join(d, f"pxp{tag}.json"), "w"), indent=1, default=float)

    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Horizon & Recency mechanism & Frequency $r$ & XP & PXP \\", r"\midrule"]
    for h in ("Short", "Long", "Combined"):
        s = fdf[fdf.horizon == h]
        for _, x in s.iterrows():
            tex.append(f"{h} & {x.family} ({x.n_models} models) & {x.r:.3f} & "
                       f"{x.xp:.3f} & {x.pxp:.3f} \\\\")
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\quad\\footnotesize BOR "
                   f"$= {s.bor.iloc[0]:.3f}$}} \\\\")
        tex.append(r"\midrule")
    tex = tex[:-1] + [r"\bottomrule", r"\end{tabular}", ""]
    open(os.path.join(d, f"pxp_family{tag}.tex"), "w").write("\n".join(tex))

    # compact table: the five leading candidates per horizon, which is what the
    # manuscript shows
    tex = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"Horizon & Model & $N_{\text{params}}$ & Frequency $r$ & XP & PXP \\",
           r"\midrule"]
    for h in ("Short", "Long", "Combined"):
        s = df[df.horizon == h].sort_values("pxp", ascending=False).head(TOP)
        for _, x in s.iterrows():
            tex.append(f"{h} & \\texttt{{{x.task}}} & {int(x.k)} & {x.r:.3f} & "
                       f"{x.xp:.3f} & {x.pxp:.3f} \\\\")
        tex.append(f"\\multicolumn{{6}}{{l}}{{\\quad\\footnotesize BOR "
                   f"$= {s.bor.iloc[0]:.3f}$}} \\\\")
        tex.append(r"\midrule")
    tex = tex[:-1] + [r"\bottomrule", r"\end{tabular}", ""]
    open(os.path.join(d, f"pxp_model{tag}.tex"), "w").write("\n".join(tex))

    # and the complete set, one table per horizon, kept for reference
    for h, pref in (("Short", "short"), ("Long", "long"), ("Combined", "combined")):
        s = df[df.horizon == h].sort_values("pxp", ascending=False)
        tt = [r"\begin{tabular}{lcccc}", r"\toprule",
              r"Model & $N_{\text{params}}$ & Frequency $r$ & XP & PXP \\", r"\midrule"]
        for _, x in s.iterrows():
            tt.append(f"\\texttt{{{x.task}}} & {int(x.k)} & {x.r:.3f} & {x.xp:.3f} & "
                      f"{x.pxp:.3f} \\\\")
        tt += [r"\bottomrule", r"\end{tabular}", ""]
        open(os.path.join(d, f"pxp_model_{pref}{tag}.tex"), "w").write("\n".join(tt))
    print(f"  wrote {d}/pxp_family{tag}.tex, pxp_model_{{short,long,combined}}{tag}.tex, "
          f"pxp{tag}.json")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, h in zip(axes, ("Short", "Long", "Combined")):
        s = fdf[fdf.horizon == h]
        x = np.arange(len(s))
        ax.bar(x, s.pxp, color=["tab:blue", "tab:orange", "0.55"][:len(s)], alpha=.9)
        ax.axhline(1 / len(s), ls="--", lw=1, color="0.4")
        for i, v in enumerate(s.pxp):
            ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(s.family, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{h}\nBOR = {s.bor.iloc[0]:.3f}", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Protected exceedance probability")
    fig.suptitle("Protected exceedance probability by recency mechanism", fontsize=11)
    fig.tight_layout()
    fdir = os.path.join(OUTDIR, "figures")
    os.makedirs(fdir, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(fdir, f"pxp_family{tag}.{ext}"),
                    bbox_inches="tight", dpi=300)
    print(f"  wrote {fdir}/pxp_family{tag}.[pdf|png|svg]")


if __name__ == "__main__":
    main()
