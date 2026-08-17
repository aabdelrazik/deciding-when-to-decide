"""Per-subject correlation of the parameters shared by the short and long winners.

The shared set is derived from the two configs rather than hardcoded: when a
winner changes, the set of parameters the two models have in common changes with
it, and a hardcoded list silently reports a parameter one of the models does not
fit.
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
SHORT = os.environ.get("SHARED_SHORT", "SB-XT-RPh----")
LONG  = os.environ.get("SHARED_LONG",  "LBE-T-RPhCL--")
OUT   = os.path.join(R, "BIC", "figures")
LABEL = {"tau": r"$\tau$", "subjective_cost": r"$R_{\mathrm{risk}}$", "patience": r"$t_p$",
         "belief_bias": r"$\beta$", "exaggeration_factor": r"$E$", "xi": r"$\xi$",
         "c_max": r"$\phi_{\max}$", "hazard_lapse": r"$L$"}

def load(task, horizon):
    f = os.path.join(R, "data/simulation_configs", f"simulation_params_{task}.py")
    ns = {"__file__": f}; exec(open(f).read(), ns)
    order = list(ns["OVERRIDES"]["PARAM_RANGES"])
    df = pd.read_pickle(os.path.join(R, "data/POMDP", task, "de", horizon, "results.pkl"))
    vals = {p: np.array([r[i] for r in df["fit_params_ga"]], dtype=float) for i, p in enumerate(order)}
    return order, pd.DataFrame(vals, index=df["subject_ID"].astype(str).values)

so, S = load(SHORT, "short")
lo, L = load(LONG, "long")
shared = [p for p in so if p in lo]
idx = S.index.intersection(L.index)
print(f"{SHORT} ({len(so)} params) vs {LONG} ({len(lo)} params)")
print(f"shared: {shared}   subjects: {len(idx)}")

def stars(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

# A single row of panels gives an extreme aspect ratio that has to be squashed
# to fit \textwidth; a near-square grid keeps the panels legible on the page.
ncol = 2 if len(shared) <= 4 else 3
nrow = int(np.ceil(len(shared) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.7 * nrow))
axes = np.atleast_1d(axes).ravel()
for extra in axes[len(shared):]:
    extra.axis("off")
rows = []
for ax, p in zip(axes, shared):
    x, y = S.loc[idx, p].values, L.loc[idx, p].values
    r, pr = stats.pearsonr(x, y); rho, prho = stats.spearmanr(x, y)
    rows.append((p, r, pr, rho, prho))
    ax.scatter(x, y, s=26, alpha=.65, edgecolor="none")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "--", color="0.4", lw=1, label="$y=x$")
    ax.set_xlabel(f"short: {LABEL.get(p,p)}"); ax.set_ylabel(f"long: {LABEL.get(p,p)}")
    ax.set_title(f"{LABEL.get(p,p)}\n$r$={r:.3f}{stars(pr)}, "+r"$\rho$"+f"={rho:.3f}{stars(prho)}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
fig.suptitle(f"Shared parameters: {SHORT} (short) vs {LONG} (long)", y=1.00, fontsize=11)
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(OUT, f"shared_parameter_correlation.{ext}"), bbox_inches="tight", dpi=300)
print(f"\nwrote {OUT}/shared_parameter_correlation.[pdf|png|svg]\n")
print(f"{'parameter':22}{'Pearson r':>12}{'p':>12}{'Spearman rho':>14}{'p':>12}")
for p, r, pr, rho, prho in rows:
    print(f"{p:22}{r:>12.3f}{pr:>12.2e}{rho:>14.3f}{prho:>12.2e}")
