"""Figure for the optimality-gap analysis.

Three panels, drawn at the final printed width so LaTeX includes it at scale 1
and the font sizes below are the sizes that appear on the page.

Two files are written. The main-text figure carries the comparison against the
normative ceiling:

  A  total points earned per policy, with every subject shown
  B  the draws dissociation: subjects match the normative agent in the short
     horizon and fall well short of it in the long one, drawn as paired
     per-subject lines because the group means alone hide that the short-horizon
     agreement holds subject by subject

The second file is the mechanism lesion, which is an appendix analysis: it asks
what a mechanism contributes inside a subject's own fitted policy, not how far
that policy sits from the optimum.

Usage (from scripts/):
    SIM_ALGORITHM=de python3 export_optimality_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(R)

SRC = os.environ.get("OPTIMALITY_SRC") or os.path.join(
    R, "BIC", "optimality", "optimality_cost.csv")
OUT = os.path.join(R, "BIC", "figures")
WIDTH_IN = float(os.environ.get("OPT_WIDTH_IN", "6.5"))

FITS = ["short", "long", "combined"]
FITLAB = {"short": "Short", "long": "Long", "combined": "Combined"}
# Panel A contrasts the subjects with the normative ceiling only. The fitted
# model earns its own comparison against the subjects in the model-accuracy
# analysis, and carrying it here made the panel a three-way comparison of which
# only the human-versus-optimum bracket is the point of the figure.
POLICIES = [("human", "Human", "0.45"),
            ("optimal_full", "Normative optimum", "tab:green")]
LESIONS = [("no_subjective_cost", r"$R_{\mathrm{risk}}$"),
           ("no_regulation", r"$\Phi(t)$"),
           ("no_hazard_lapse", "$L$"),
           ("no_belief_bias", "$\\beta$"),
           ("no_exaggeration", "$E$")]
TS, LS = 8, 7          # title and label/tick sizes, in printed points


def star(p):
    """Significance marker. n.s. is spelled out: for the human-vs-model
    comparison a null is the desired result, so it should not read as a
    missing annotation."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def bracket(ax, x0, x1, y, txt, h, fs, colour="0.15"):
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=.6, color="0.25",
            clip_on=False, zorder=5)
    ax.annotate(txt, ((x0 + x1) / 2, y + h), ha="center", va="bottom",
                fontsize=fs, color=colour, clip_on=False, zorder=5)


def main():
    d = pd.read_csv(SRC)
    for c in d.columns:
        if c not in ("subject_ID", "fit", "task"):
            d[c] = pd.to_numeric(d[c], errors="coerce")

    fig = plt.figure(figsize=(WIDTH_IN, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.55,
                          left=0.09, right=0.985, top=0.92, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    figC, axC = plt.subplots(figsize=(WIDTH_IN * 0.62, 2.6))
    rng = np.random.default_rng(0)

    # --- A: points earned -------------------------------------------------
    w = 0.26
    for pi, (key, lab, col) in enumerate(POLICIES):
        xs, ms, es = [], [], []
        for fi, f in enumerate(FITS):
            s = d[d.fit == f]
            pts = s[key] * s.n_games
            x = fi + (pi - (len(POLICIES) - 1) / 2) * w
            xs.append(x); ms.append(pts.mean()); es.append(pts.std())
            axA.scatter(x + rng.uniform(-.07, .07, len(pts)), pts, s=1.6,
                        color=col, alpha=.28, linewidths=0, rasterized=True,
                        zorder=1)
        axA.bar(xs, ms, w * .86, yerr=es, color=col, alpha=.85, zorder=2,
                error_kw=dict(lw=.7, capsize=2), label=lab)
    # Bracket heights are spaced in axis units, not per-group units: the three
    # conditions differ several-fold in scale, so a per-group step made the
    # short-horizon brackets collide with each other and with the annotation.
    hi = max(float((d[d.fit == f][k] * d[d.fit == f].n_games).max())
             for f in FITS for k, _, _ in POLICIES)
    ytop = hi * 1.24          # headroom for the single shortfall bracket
    step = ytop * 0.062
    for fi, f in enumerate(FITS):
        s = d[d.fit == f]
        n = s.n_games.values
        H, O = s.human.values * n, s.optimal_full.values * n
        base = max(v.mean() + v.std() for v in (H, O)) + step * 0.55
        # the bracket carries the shortfall itself, rather than repeating the
        # same comparison as a separate label that collides with it
        lab = f"shortfall {(O - H).mean():.1f} pts {star(wilcoxon(H, O)[1])}"
        bracket(axA, fi - w / 2, fi + w / 2, base, lab, step * 0.30, LS,
                "firebrick")
    axA.set_ylim(0, ytop)
    axA.set_xticks(range(len(FITS)))
    axA.set_xticklabels([f"{FITLAB[f]}\n({int(d[d.fit==f].n_games.iloc[0])} games)"
                         for f in FITS], fontsize=LS)
    axA.set_ylabel("Total points over the session", fontsize=LS)
    axA.set_title("Points earned, against the normative ceiling", fontsize=TS)
    axA.legend(fontsize=LS, frameon=False, ncol=len(POLICIES), loc="upper left",
               bbox_to_anchor=(0.0, 1.0), handlelength=1.2, columnspacing=1.2)
    axA.tick_params(labelsize=LS)
    axA.grid(axis="y", alpha=.25, lw=.5)
    axA.set_axisbelow(True)

    # --- B: draws, paired per subject ------------------------------------
    for fi, f in enumerate(("short", "long")):
        s = d[d.fit == f]
        x0, x1 = fi * 1.4, fi * 1.4 + 0.55
        for h, o in zip(s.human_draws, s.optimal_full_draws):
            axB.plot([x0, x1], [h, o], color="0.6", lw=.35, alpha=.5,
                     zorder=1, rasterized=True)
        for x, v, col in ((x0, s.human_draws, "0.25"),
                          (x1, s.optimal_full_draws, "tab:green")):
            axB.scatter([x] * len(v), v, s=4, color=col, zorder=2,
                        linewidths=0, rasterized=True)
            axB.plot([x - .18, x + .18], [v.mean()] * 2, color="firebrick",
                     lw=1.4, zorder=3)
        diff = (s.human_draws - s.optimal_full_draws).mean()
        pw = wilcoxon(s.human_draws, s.optimal_full_draws)[1]
        axB.annotate(f"{FITLAB[f]}   {diff:+.2f} draws {star(pw)}",
                     ((x0 + x1) / 2, 1.02), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=LS, color="firebrick")
    axB.set_xticks([0, .55, 1.4, 1.95])
    axB.set_xticklabels(["Human", "Optimum", "Human", "Optimum"], fontsize=LS)
    axB.set_ylabel("Mean draws per game", fontsize=LS)
    axB.set_title("Draws taken vs. the optimum", fontsize=TS, pad=14)
    axB.tick_params(labelsize=LS)
    axB.grid(axis="y", alpha=.25, lw=.5)
    axB.set_axisbelow(True)
    axB.margins(y=0.12)

    # --- C: what each mechanism is worth ---------------------------------
    keys = [k for k, _ in LESIONS]
    ypos = np.arange(len(keys))[::-1]
    for fi, (f, col) in enumerate(zip(("short", "long"),
                                      ("tab:blue", "tab:orange"))):
        s = d[d.fit == f]
        vals = []
        for k in keys:
            if k in s and s[k].notna().any():
                vals.append(((s[k] - s.fitted) * s.n_games).mean())
            else:
                vals.append(np.nan)
        axC.barh(ypos + (0.2 if fi == 0 else -0.2), vals, 0.36, color=col,
                 alpha=.85, label=FITLAB[f])
    axC.axvline(0, color="0.3", lw=.7)
    axC.set_yticks(ypos)
    axC.set_yticklabels([lab for _, lab in LESIONS], fontsize=LS + 1)
    axC.set_xlabel("Points lost when the mechanism is removed", fontsize=LS)
    axC.set_title("What each mechanism is worth", fontsize=TS, pad=14)
    axC.legend(fontsize=LS, frameon=False, loc="lower left")
    axC.tick_params(labelsize=LS)
    axC.grid(axis="x", alpha=.25, lw=.5)
    axC.set_axisbelow(True)

    for ax, lab in ((axA, "A"), (axB, "B")):
        ax.annotate(lab, (-0.055, 1.06), xycoords="axes fraction",
                    fontsize=9, fontweight="bold", va="bottom")

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"optimality_gap.{ext}"),
                    bbox_inches="tight", dpi=400)
        figC.savefig(os.path.join(OUT, f"optimality_lesion.{ext}"),
                     bbox_inches="tight", dpi=400)
    print(f"wrote {OUT}/optimality_gap.[pdf|png|svg] and "
          f"{OUT}/optimality_lesion.[pdf|png|svg]")


if __name__ == "__main__":
    main()
