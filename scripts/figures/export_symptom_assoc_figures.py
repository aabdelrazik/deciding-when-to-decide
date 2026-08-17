"""Fitted POMDP parameters vs obsessive-compulsive symptom measures.

Produces the symptom-association figures for the long and combined horizons.

The winning model and its parameter list come from BIC/best_models.json and the
config, never from a hardcoded list: the winners change, and a hardcoded list
plots a parameter the current model does not fit (the previous version of this
figure showed hazard lapse for the combined horizon, which C-EXT-RPHC-UK does
not have).

Usage:
    SIM_ALGORITHM=de python3 export_symptom_assoc_figures.py
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
OUT = os.path.join(R, "figures", "group_level_exploration")

LABEL = {"tau": r"Temperature $\tau$", "subjective_cost": r"Subjective cost $R_{\mathrm{risk}}$",
         "patience": r"Patience $t_p$", "belief_bias": r"Belief bias $\beta$",
         "exaggeration_factor": r"Exaggeration factor $E$", "xi": r"Lapse rate $\xi$",
         "c_max": r"Max regulation $\phi_{\max}$", "hazard_lapse": r"Hazard lapse $L$",
         "gamma": r"Forgetting factor $\gamma$", "is_hazardous": r"Hazard $H$",
         "urgency_coefficient": r"Min regulation $\phi_{\min}$",
         "urgency_slope": r"Regulation slope $k$"}

# Y-BOCS was administered to the OCD patients only, so those panels are N = 29.
YBOCS = ("YBOCS_total_score", "YBOCS_insight", "YBOCS_Indecisiveness",
         "YBOCS_obsess_subtotal", "YBOCS_compulsions_subtotal")

# The obsessive-compulsive measures the manuscript states this scan covers. The
# questionnaire file also carries IQ, BIS, BDI, STAI, FMPS and the other two
# factor scores; sweeping those too would inflate the test count and put a
# non-symptom measure such as IQ into a symptom-association figure.
MEASURES = ("ML2", "OCIR_total", "OCIR_Washing", "OCIR_Obsessing", "OCIR_Hoarding",
            "OCIR_Ordering", "OCIR_Checking", "OCIR_Neutralizing", "compulsion_score",
            "PIWSUR_total", "PIWSUR_contamination_and_washing",
            "PIWSUR_dressing_and_grooming", "PIWSUR_checking",
            "PIWSUR_obsessional_thoughts_of_harm", "PIWSUR_obsessional_impulses_to_harm"
            ) + YBOCS

# Axis labels for the measures. Without these the raw CSV column names end up on
# the figure axes, underscores and all.
MEASURE_LABEL = {
    "ML2": "OC factor (FA2)",
    "OCIR_total": "OCI-R total",
    "OCIR_Washing": "OCI-R washing",
    "OCIR_Obsessing": "OCI-R obsessing",
    "OCIR_Hoarding": "OCI-R hoarding",
    "OCIR_Ordering": "OCI-R ordering",
    "OCIR_Checking": "OCI-R checking",
    "OCIR_Neutralizing": "OCI-R neutralizing",
    "compulsion_score": "Composite compulsion score",
    "PIWSUR_total": "PI-WSUR total",
    "PIWSUR_contamination_and_washing": "PI-WSUR contamination and washing",
    "PIWSUR_dressing_and_grooming": "PI-WSUR dressing and grooming",
    "PIWSUR_checking": "PI-WSUR checking",
    "PIWSUR_obsessional_thoughts_of_harm": "PI-WSUR obsessional thoughts of harm",
    "PIWSUR_obsessional_impulses_to_harm": "PI-WSUR obsessional impulses to harm",
    "YBOCS_total_score": "Y-BOCS total",
    "YBOCS_insight": "Y-BOCS insight",
    "YBOCS_Indecisiveness": "Y-BOCS indecisiveness",
    "YBOCS_obsess_subtotal": "Y-BOCS obsessions subtotal",
    "YBOCS_compulsions_subtotal": "Y-BOCS compulsions subtotal",
}


# Which comparison each figure shows, as (parameter, measure, subgroup). Chosen
# editorially, not by p-value: the long-horizon pair is the recency argument the
# text rests on, the combined pair the two strongest of the associations reported
# in the Results. Values are still read from the scan, so they cannot go stale.
PANELS = {
    "long": [("exaggeration_factor", "OCIR_Checking", "all"),
             ("exaggeration_factor", "YBOCS_Indecisiveness", "all")],
    "combined": [("xi", "OCIR_total", "all"),
                 ("c_max", "OCIR_Checking", "top20 compulsive")],
}


def symptoms() -> pd.DataFrame:
    meg = os.path.join(R, "data/TrHu_NHB_light/data_MEG")
    fa = pd.read_csv(os.path.join(meg, "fa_scores.csv")).rename(columns={"ML2": "FA2"})
    fa["compulsion_score"] = fa[
        ["OCIR_Washing", "OCIR_Checking", "OCIR_Ordering", "OCIR_Neutralizing"]
    ].sum(axis=1)
    yb = pd.read_csv(os.path.join(meg, "ybocs_scores.csv"))
    return fa.merge(yb, on="userID", how="outer")


def winner(horizon: str):
    """Return (task, parameter names, fitted values indexed by subject)."""
    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        task = json.load(fh)[horizon]
    cfg = os.path.join(R, "data/simulation_configs", f"simulation_params_{task}.py")
    ns = {"__file__": cfg}
    exec(open(cfg).read(), ns)
    names = list(ns["OVERRIDES"]["PARAM_RANGES"])
    sub = "" if horizon == "combined" else horizon
    path = os.path.join(R, "data/POMDP", task, ALGO, sub, "results.pkl") if sub else \
           os.path.join(R, "data/POMDP", task, ALGO, "results.pkl")
    df = pd.read_pickle(path)
    vals = pd.DataFrame(
        {p: np.array([row[i] for row in df["fit_params_ga"]], float) for i, p in enumerate(names)},
        index=df["subject_ID"].astype(str).values)
    return task, names, vals


def scan(vals, names, sym, measures, top_n=None):
    """Spearman rho for every parameter x measure, optionally within a subgroup."""
    out = []
    sym = sym.copy()
    sym.index = sym["userID"].astype(str)
    for m in measures:
        if m not in sym.columns:
            continue
        s = sym[m].dropna()
        idx = vals.index.intersection(s.index)
        if len(idx) < 10:
            continue
        sub = idx
        if top_n:  # restrict to the most compulsive subjects
            comp = sym.loc[idx, "compulsion_score"].dropna()
            sub = comp.sort_values(ascending=False).head(top_n).index
        for p in names:
            x, y = vals.loc[sub, p].values, sym.loc[sub, m].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10 or np.std(x[ok]) == 0:
                continue
            rho, pv = stats.spearmanr(x[ok], y[ok])
            out.append(dict(parameter=p, measure=m, n=int(ok.sum()), rho=rho, p=pv,
                            subgroup="top20 compulsive" if top_n else "all"))
    return pd.DataFrame(out)


def panel(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, s=34, alpha=.75, color="tab:orange", edgecolor="none")
    if np.std(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, "--", color="tab:orange", lw=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25)


def build(horizon: str):
    task, names, vals = winner(horizon)
    sym = symptoms()
    measures = [c for c in MEASURES if c in sym.columns]
    res = pd.concat([scan(vals, names, sym, measures),
                     scan(vals, names, sym, measures, top_n=20)], ignore_index=True)
    res = res.sort_values("p")
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, f"symptom_assoc_{horizon}.csv"), index=False)

    # Panels are pinned, not chosen by smallest p-value. Ranking by p makes the
    # figure show whichever comparison happens to survive the noise, which drifts
    # on every refit and need not be a parameter the text discusses; these are the
    # comparisons the recency argument rests on.
    pinned = PANELS.get(horizon, [])
    sig = pd.concat([res[(res.parameter == p) & (res.measure == m) & (res.subgroup == s)]
                     for p, m, s in pinned]) if pinned else \
        res[res.p < .05].drop_duplicates(subset=["parameter", "measure"]).head(2)
    print(f"\n{horizon}: {task} ({len(names)} parameters), "
          f"{len(res)} tests, {int((res.p < .05).sum())} at p<0.05 uncorrected")
    for _, r in res.head(5).iterrows():
        print(f"    {r.parameter:22} {r.measure:34} n={r.n:<4} rho={r.rho:+.3f} p={r.p:.4f} [{r.subgroup}]")
    if sig.empty:
        print("    nothing significant; no figure written")
        return

    si = sym.copy(); si.index = si["userID"].astype(str)
    fig, axes = plt.subplots(1, len(sig), figsize=(5.6 * len(sig), 4.2), squeeze=False)
    for ax, (_, r) in zip(axes[0], sig.iterrows()):
        s = si[r.measure].dropna()
        idx = vals.index.intersection(s.index)
        if r.subgroup.startswith("top20"):
            idx = si.loc[idx, "compulsion_score"].dropna().sort_values(ascending=False).head(20).index
        x, y = vals.loc[idx, r.parameter].values, si.loc[idx, r.measure].values.astype(float)
        star = "*" if r.p < .05 else ""
        sub = "all subjects" if r.subgroup == "all" else "20 most compulsive subjects"
        panel(ax, x, y, LABEL.get(r.parameter, r.parameter),
              MEASURE_LABEL.get(r.measure, r.measure),
              f"{sub}\n" + r"$\rho$" + f" = {r.rho:.2f}{star}, p = {r.p:.3f} (N = {r.n})")
    fig.suptitle(f"{horizon.capitalize()}-horizon fitted parameters vs obsessive-compulsive symptoms "
                 f"({task})", y=1.02, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"symptom_assoc_{horizon}.{ext}"), bbox_inches="tight", dpi=300)
    print(f"    wrote {OUT}/symptom_assoc_{horizon}.[pdf|png|svg]")


if __name__ == "__main__":
    for h in ("long", "combined"):
        build(h)
