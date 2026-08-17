"""All three winners' parameter recovery as one figure, sized for PLOS.

The three horizons were three separate full-width figures, one per page. This
packs them into a single figure with labelled blocks, which is the layout the
three-panel result wants anyway and saves two pages.

Drawn at its final printed size (6.5 in wide, the template's \\textwidth) so it is
included at width=\\textwidth with no scaling. That is what keeps the type legible
in a grid this dense: a figure drawn large and then shrunk by LaTeX takes its
fonts down with it, which is how the earlier five-column version became
unreadable. Font sizes below are therefore points on the printed page. PLOS asks
for 8 to 12 pt text in figures and allows smaller tick labels.

Usage (from scripts/):
    SIM_ALGORITHM=de python3 export_parameter_recovery_panel.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ALGO = os.environ.get("SIM_ALGORITHM", "de")
OUT = os.path.join(R, "BIC", "figures")
NCOL = int(os.environ.get("RECOVERY_NCOL", "5"))
WIDTH_IN = float(os.environ.get("RECOVERY_WIDTH_IN", "6.5"))

LABEL = {"tau": r"$\tau$", "subjective_cost": r"$R_{\mathrm{risk}}$", "patience": r"$t_p$",
         "belief_bias": r"$\beta$", "exaggeration_factor": r"$E$", "xi": r"$\xi$",
         "c_max": r"$\phi_{\max}$", "hazard_lapse": r"$L$", "gamma": r"$\gamma$",
         "is_hazardous": r"$H$", "urgency_coefficient": r"$\phi_{\min}$",
         "urgency_slope": r"$k$"}
HORIZON_NAME = {"short": "short horizon", "long": "long horizon",
                "combined": "combined horizons"}


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def load(horizon):
    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        task = json.load(fh)[horizon]
    cfg = os.path.join(R, "data/simulation_configs", f"simulation_params_{task}.py")
    ns = {"__file__": cfg}
    exec(open(cfg).read(), ns)
    names = list(ns["OVERRIDES"]["PARAM_RANGES"])

    sub = "" if horizon == "combined" else horizon
    d = os.path.join(R, "data/POMDP", task, ALGO, sub) if sub else \
        os.path.join(R, "data/POMDP", task, ALGO)
    rec = os.path.join(d, "results_recovered.pkl")
    if not os.path.exists(rec):
        print(f"  {horizon}: no results_recovered.pkl for {task}; skipped")
        return None
    a = pd.read_pickle(os.path.join(d, "results.pkl")).set_index("subject_ID")
    b = pd.read_pickle(rec).set_index("subject_ID")
    idx = a.index.intersection(b.index)
    A = np.array([r for r in a.loc[idx, "fit_params_ga"]], float)
    B = np.array([r for r in b.loc[idx, "fit_params_ga"]], float)
    return dict(task=task, names=names, A=A, B=B, n=len(idx))


def main():
    blocks = [(h, load(h)) for h in ("short", "long", "combined")]
    blocks = [(h, b) for h, b in blocks if b]
    if not blocks:
        raise SystemExit("no recovery results found")

    rows = [int(np.ceil(len(b["names"]) / NCOL)) for _, b in blocks]
    panel_h = WIDTH_IN / NCOL * 0.92          # panel aspect, slightly wide
    height = sum(rows) * panel_h + 0.52 * len(blocks) + 0.25
    fig = plt.figure(figsize=(WIDTH_IN, height))
    outer = fig.add_gridspec(len(blocks), 1, hspace=0.75,
                             height_ratios=[r for r in rows])

    for bi, ((horizon, blk), nrow) in enumerate(zip(blocks, rows)):
        gs = outer[bi].subgridspec(nrow, NCOL, hspace=0.72, wspace=0.52)
        names, A, B = blk["names"], blk["A"], blk["B"]
        print(f"  {horizon:9} {blk['task']:14} {len(names)} params, {blk['n']} subjects, "
              f"{nrow} row(s)")
        for k, p in enumerate(names):
            ax = fig.add_subplot(gs[k // NCOL, k % NCOL])
            x, y = A[:, k], B[:, k]
            r, pv = stats.pearsonr(x, y)
            ax.scatter(x, y, s=4.5, alpha=.45, color="tab:blue", edgecolor="none",
                       rasterized=True)
            lim = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lim, lim, "--", color="0.45", lw=.7)
            if np.std(x) > 0:
                m, c = np.polyfit(x, y, 1)
                xs = np.linspace(*lim, 50)
                ax.plot(xs, m * xs + c, color="tab:blue", lw=1.0)
            ax.set_title(f"{LABEL.get(p, p)}  $r$={r:.2f}{stars(pv)}", fontsize=8, pad=2.5)
            ax.tick_params(labelsize=6, length=2, pad=1.2)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            for s in ax.spines.values():
                s.set_linewidth(.6)
        # block label, and the axis meaning once per block instead of per panel
        top = fig.add_subplot(gs[0, 0])
        pos = top.get_position()
        top.remove()
        # clear the first panel's own title, which sits directly above pos.y1
        y = pos.y1 + 0.030
        fig.text(0.005, y, f"({chr(97 + bi)})", fontsize=9,
                 fontweight="bold", va="bottom")
        fig.text(0.055, y,
                 f"\\texttt{{{blk['task']}}}, {HORIZON_NAME[horizon]}"
                 if plt.rcParams["text.usetex"] else
                 f"{blk['task']}, {HORIZON_NAME[horizon]}",
                 fontsize=8, va="bottom")

    fig.supxlabel("True value", fontsize=8, y=0.004)
    fig.supylabel("Recovered value", fontsize=8, x=0.004)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"parameter_recovery_panel.{ext}"),
                    bbox_inches="tight", dpi=400)
    print(f"\n  wrote {OUT}/parameter_recovery_panel.[pdf|png|svg] "
          f"({WIDTH_IN:.1f} x {height:.1f} in, {NCOL} cols)")


if __name__ == "__main__":
    main()
