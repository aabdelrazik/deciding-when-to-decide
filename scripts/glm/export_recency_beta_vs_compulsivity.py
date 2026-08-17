"""Recency-weighting GLM coefficient vs. obsessive-compulsive symptom measures.

Reproduces del Rio et al.'s model-agnostic result -- the GLM weight on the most
recent evidence update decreases with symptom severity -- from choices simulated
by the commit-fit POMDP, and checks it against the same correlation computed on
the real human choices.

The human rows depend only on the questionnaire scores and the human GLM fit, so
they must not move when the POMDP winner changes; the simulated rows are read
from the ensemble-averaged per-subject betas, which do move. Both the model pair
and the beta column are resolved rather than hardcoded: beta2 is the Delta ES_t
regressor per BETA_LABEL_MAP_GLM in plot_glm_array_commit_cached.py.

Writes the table and the figure that the appendix recency paragraph cites.

Usage:
    SIM_ALGORITHM=de python3 export_recency_beta_vs_compulsivity.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ALGO = os.environ.get("SIM_ALGORITHM", "de")
MEG = os.path.join(R, "data/TrHu_NHB_light/data_MEG")

# beta2 is the Delta ES_t (most recent evidence update) regressor.
BETA = "beta2"
MEASURES = [("FA2", "FA2"), ("OCIR_total", "OCIR total")]


def stars(p):
    return "**" if p < .01 else "*" if p < .05 else ""


def winners():
    with open(os.path.join(R, "BIC_commit", "best_models.json")) as fh:
        b = json.load(fh)
    return b["short"], b["long"]


def symptoms() -> pd.DataFrame:
    fa = pd.read_csv(os.path.join(MEG, "fa_scores.csv")).rename(columns={"ML2": "FA2"})
    if "OCIR_total" not in fa.columns:
        ocir = [c for c in fa.columns if c.startswith("OCIR_") and c != "OCIR_total"]
        fa["OCIR_total"] = fa[ocir].sum(axis=1)
    return fa[["userID", "FA2", "OCIR_total"]]


def betas():
    """Per-subject Delta ES_t coefficient, human and commit-POMDP-simulated."""
    short, long = winners()
    sim_p = os.path.join(R, "data/POMDP_commit", long, ALGO, f"glm_vs_{short}",
                         "glm_betas_simulated_averaged.pkl")
    if not os.path.exists(sim_p):
        raise SystemExit(f"no cached simulated betas at {sim_p}; run the commit GLM array first")
    sim = pd.read_pickle(sim_p)[["userID", BETA]].rename(columns={BETA: "sim"})
    hum = pd.read_csv(os.path.join(MEG, "glm_betas_human.csv"))[["userID", BETA]] \
        .rename(columns={BETA: "human"})
    print(f"  commit winners: short={short}  long={long}  ({ALGO})")
    print(f"  simulated betas: {os.path.relpath(sim_p, R)} ({len(sim)} subjects)")
    return short, long, hum.merge(sim, on="userID", how="inner")


def main():
    short, long, b = betas()
    df = b.merge(symptoms(), on="userID", how="inner")

    rows = []
    for src, col in (("GLM on commit-POMDP-simulated", "sim"), ("Human GLM", "human")):
        for key, label in MEASURES:
            d = df[[col, key]].dropna()
            rho, p = stats.spearmanr(d[col], d[key])
            rows.append(dict(source=src, measure=label, key=key, col=col,
                             n=len(d), rho=rho, p=p))
    res = pd.DataFrame(rows)
    for _, r in res.iterrows():
        print(f"    {r.source:32} {r.measure:12} n={r.n:<4} rho={r.rho:+.3f} p={r.p:.4f}")

    # ordered measure-major so the two sources sit next to each other per measure
    out = os.path.join(R, "BIC_commit", "tables")
    os.makedirs(out, exist_ok=True)
    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Source & Questionnaire measure & Spearman $\rho$ & $p$ \\", r"\midrule"]
    for _, label in MEASURES:
        for src in ("GLM on commit-POMDP-simulated", "Human GLM"):
            r = res[(res.source == src) & (res.measure == label)].iloc[0]
            lines.append(f"{src} & {label} & ${r.rho:.3f}{stars(r.p)}$ & ${r.p:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex = os.path.join(out, "recency_beta_vs_compulsivity_table.tex")
    open(tex, "w").write("\n".join(lines))
    print(f"  wrote {os.path.relpath(tex, R)}")

    figs = os.path.join(R, "figures/POMDP_commit", long, ALGO, f"glm_vs_{short}")
    os.makedirs(figs, exist_ok=True)
    # One row per questionnaire measure, model on the left and human on the
    # right, so each panel sits beside the comparison it is meant to be read
    # against rather than a row apart.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6))
    for i, (key, label) in enumerate(MEASURES):
        for j, (src, col, colour) in enumerate(
                (("GLM on commit-POMDP-simulated", "sim", "tab:blue"),
                 ("Human GLM", "human", "black"))):
            ax = axes[i][j]
            d = df[[col, key]].dropna()
            x, y = d[key].values, d[col].values
            r = res[(res.source == src) & (res.measure == label)].iloc[0]
            ax.scatter(x, y, s=30, alpha=.7, color=colour, edgecolor="none")
            m, c = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + c, "--", color=colour, lw=1.6)
            ax.set_xlabel(label)
            ax.set_ylabel(r"$\beta_{\Delta ES_t}$")
            ax.set_title(f"{src}\n" + r"$\rho$" +
                         f" = {r.rho:.2f}{stars(r.p)}, p = {r.p:.3f} (N = {r.n})", fontsize=10)
            ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(figs, f"recency_beta_vs_compulsivity_human_vs_simulated.{ext}"),
                    bbox_inches="tight", dpi=300)
    print(f"  wrote {os.path.relpath(figs, R)}/recency_beta_vs_compulsivity_human_vs_simulated.[pdf|png|svg]")


if __name__ == "__main__":
    main()
