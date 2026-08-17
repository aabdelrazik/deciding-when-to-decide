"""Table and figure for the fixed temporal-regulation value sweep.

Reads the .npz files written by sensitivity_value_sweep.py and reports, per
swept parameter, the chosen value, the value that maximises the population-summed
log-likelihood, and the BIC cost of the former. Since a sweep changes only a
value and not the number of free parameters, that cost is 2 * dlogL.

Horizons whose winner fits every regulation parameter per subject have nothing
fixed to be sensitive about and are omitted, which is why the combined horizon
does not appear: C-EXT-RPHC-UK fits phi_max, phi_min and k.

Usage (from scripts/):
    python3 export_sensitivity_sweep_table.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

LABEL = {"c_max": r"$\phi_{\max}$", "urgency_coefficient": r"$\phi_{\min}$",
         "urgency_slope": r"$k$"}
RANGE = {"c_max": "[0, 80]", "urgency_coefficient": "[-30, 0]",
         "urgency_slope": "[-20, 0]"}
ORDER = ("c_max", "urgency_coefficient", "urgency_slope")
OUT = os.path.join(project_root, "BIC", "tables", "sensitivity_value_sweep_table.tex")


def load(horizon):
    p = os.path.join(project_root, "BIC", "tables", f"sensitivity_{horizon}_sweep.npz")
    if not os.path.exists(p):
        return None, []
    z = np.load(p, allow_pickle=True)
    if "nothing_fixed" in z:
        return str(z["task"]), []
    out = []
    for k in ORDER:
        if f"{k}_grid" not in z:
            continue
        grid, lls = z[f"{k}_grid"], z[f"{k}_lls"]
        cur, chosen = float(z[f"{k}_cur_ll"]), float(z[f"{k}_chosen"])
        b = int(np.argmax(lls))
        out.append(dict(param=k, chosen=chosen, opt=float(grid[b]),
                        dbic=2.0 * (float(lls[b]) - cur), grid=grid, lls=lls, cur=cur))
    return str(z["task"]), out


def main():
    with open(os.path.join(project_root, "BIC", "best_models.json")) as fh:
        best = json.load(fh)

    blocks, missing = [], []
    for h in ("short", "long", "combined"):
        task, rows = load(h)
        if task is None:
            missing.append(h)
            continue
        if not rows:
            print(f"  {h:9} {best[h]}: nothing fixed, omitted from the table")
            continue
        blocks.append((h.capitalize(), task, rows))
        for r in rows:
            print(f"  {h:9} {r['param']:20} chosen={r['chosen']:<7g} "
                  f"optimum={r['opt']:<7g} dBIC={r['dbic']:.1f}")
    if missing:
        raise SystemExit(f"no sweep output for {missing}; run sensitivity_value_sweep.py")

    tex = [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
           r"Horizon & Parameter & Chosen value & Optimum & "
           r"$\Delta$BIC (chosen vs.\ optimum) & Sweep range \\", r"\midrule"]
    for h, task, rows in blocks:
        for r in rows:
            fmt = lambda v: (f"${v:g}$" if v >= 0 else f"$-{abs(v):g}$")
            tex.append(f"{h} & {LABEL[r['param']]} & {fmt(r['chosen'])} & "
                       f"{fmt(r['opt'])} & {r['dbic']:.1f} & ${RANGE[r['param']]}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", "}", ""]
    open(OUT, "w").write("\n".join(tex))
    print(f"\nwrote {OUT}")

    n = sum(len(r) for _, _, r in blocks)
    # wrap at 3 columns: five panels in a single row get scaled down so far by
    # \textwidth that the tick labels stop being readable
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.5 * nrow), squeeze=False)
    axes = np.atleast_1d(axes).ravel().reshape(1, -1)
    for extra in axes[0][n:]:
        extra.axis("off")
    i = 0
    for h, task, rows in blocks:
        for r in rows:
            ax = axes[0][i]; i += 1
            dbic = 2.0 * (np.max(r["lls"]) - r["lls"])
            ax.plot(r["grid"], dbic, color="tab:blue", lw=1.4)
            ax.axvline(r["chosen"], ls="--", color="tab:blue", lw=1.4,
                       label="chosen" if i == 1 else None)
            ax.axvline(r["opt"], ls=":", color="tab:red", lw=1.6,
                       label="optimum" if i == 1 else None)
            ax.set_xlabel(f"{LABEL[r['param']]}  ({h.lower()} horizon)", fontsize=9)
            ax.set_ylabel(r"$\Delta$BIC from optimum", fontsize=9)
            ax.set_title(f"cost of chosen value: {r['dbic']:.1f}", fontsize=9)
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    figdir = os.path.join(project_root, "BIC", "figures")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(figdir, f"sensitivity_value_sweep.{ext}"),
                    bbox_inches="tight", dpi=300)
    print(f"wrote {figdir}/sensitivity_value_sweep.[pdf|png|svg]")


if __name__ == "__main__":
    main()
