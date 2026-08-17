"""Full-fit vs commit-fit parameter values, per subject, for the same structure.

The parameter set is taken from the config rather than hardcoded: the two
objectives can select different winning structures, and a fixed panel list
silently plots a parameter the model does not fit.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

sys.path.append(os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PAIRS = [("short", os.environ.get("FVC_SHORT", "SB-XT-RPh----")),
         ("long",  os.environ.get("FVC_LONG",  "LBE-T-RPhCL--"))]
LABEL = {"tau": r"$\tau$", "subjective_cost": r"$R_{\mathrm{risk}}$", "patience": r"$t_p$",
         "belief_bias": r"$\beta$", "exaggeration_factor": r"$E$", "xi": r"$\xi$",
         "c_max": r"$\phi_{\max}$", "hazard_lapse": r"$L$", "is_hazardous": r"$H$",
         "urgency_coefficient": r"$\phi_{\min}$", "urgency_slope": r"$k$"}

def load(task, horizon, commit):
    sfx = "_commit" if commit else ""
    f = f"{R}/data/simulation_configs/simulation_params_{task}{sfx}.py"
    ns = {"__file__": f}; exec(open(f).read(), ns)
    order = list(ns["OVERRIDES"]["PARAM_RANGES"])
    sub = "POMDP_commit" if commit else "POMDP"
    df = pd.read_pickle(f"{R}/data/{sub}/{task}/de/{horizon}/results.pkl")
    v = {p: np.array([r[i] for r in df["fit_params_ga"]], float) for i, p in enumerate(order)}
    return order, pd.DataFrame(v, index=df["subject_ID"].astype(str).values)

def stars(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

rows_out = []
for horizon, task in PAIRS:
    of, F = load(task, horizon, False)
    oc, C = load(task, horizon, True)
    params = [p for p in of if p in oc]
    idx = F.index.intersection(C.index)
    ncol = 4; nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for a in axes[len(params):]: a.axis("off")
    print(f"{horizon} {task}: {len(params)} shared params -> {params}")
    for ax, p in zip(axes, params):
        x, y = F.loc[idx, p].values, C.loc[idx, p].values
        r, pv = stats.pearsonr(x, y)
        rows_out.append((horizon, task, p, r, pv))
        ax.scatter(x, y, s=24, alpha=.6, edgecolor="none")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, "--", color="0.5", lw=1)
        m, b = np.polyfit(x, y, 1); xs = np.linspace(*lim, 50)
        ax.plot(xs, m * xs + b, color="tab:red", lw=1.2)
        ax.set_xlabel(f"Full-fit {LABEL.get(p,p)}"); ax.set_ylabel(f"Commit-fit {LABEL.get(p,p)}")
        ax.set_title(f"{LABEL.get(p,p)} ($r={r:.2f}${stars(pv)})", fontsize=10)
    fig.suptitle(f"{horizon}: {task}", y=1.00, fontsize=11)
    fig.tight_layout()
    out = f"{R}/BIC/figures/fullfit_vs_commit_params_{horizon}"
    for ext in ("pdf", "png", "svg"): fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=300)
    print(f"  wrote {out}.[pdf|png|svg]")

# Use the LaTeX label for each parameter: raw names such as subjective_cost
# contain an underscore, which is a subscript operator in text mode and makes
# the table fail to compile.
rows_out = [(h, f"\\texttt{{{m}}}", LABEL.get(p, p.replace("_", r"\_")), r, pv)
            for h, m, p, r, pv in rows_out]
t = pd.DataFrame(rows_out, columns=["Horizon", "Model", "Parameter", "$r$", "$p$"])
tex = t.to_latex(index=False, float_format="%.3f", escape=False)
open(f"{R}/BIC/tables/fullfit_vs_commit_params_table.tex", "w").write(tex)
print(f"\nwrote BIC/tables/fullfit_vs_commit_params_table.tex ({len(t)} rows)")
