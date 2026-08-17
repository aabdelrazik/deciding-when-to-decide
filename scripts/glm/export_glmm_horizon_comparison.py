"""Per-horizon GLMM: human vs model-simulated, one horizon at a time.

The GLMM reported in the appendix is fitted to short and long trials together,
with termination (the horizon condition) as a regressor. This fits it separately
within each horizon, comparing that horizon's own winning model against the
human data restricted to the same horizon.

Within one horizon termination is constant, so it is collinear with the
intercept and carries no information. It and its interactions are dropped rather
than fitted at zero, leaving:

    decide ~ totevminus + deltaev + trial
             + FA2 + FA2:totevminus + FA2:deltaev + FA2:trial
             + subject fixed effects

Human trials are split on the termination column that del Rio's data already
carries (1 = long, 2 = short). Model trials come from that horizon's own fit, so
the short panel uses the short winner simulated on short games only, and the
long panel the long winner on long games only.

Usage (from scripts/):
    SIM_ALGORITHM=de python3 export_glmm_horizon_comparison.py
    POMDP_SEEDS_COMMIT=1 ...   to use the commit-likelihood winners
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

from src.config.loader import load_config
from src.glm import fit_glm, fit_glm_separate_for_human_data, fit_glmm
from src.utils import (assemble_glm_outputs, combine_sequences,
                       load_questionnaire_data)
from src.utils.plotting import plot_glmm_betas_comparison

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

COMMIT = os.environ.get("POMDP_SEEDS_COMMIT", "") == "1"
BEST = "BIC_commit" if COMMIT else "BIC"
SUBDIR = "POMDP_commit" if COMMIT else "POMDP"
N_INSTANCES = int(os.environ.get("GLMM_HORIZON_INSTANCES", "20"))
# One randomly chosen simulation instead of the pooled ensemble. Pooling many
# instances inflates the model's trial count by that factor, which shrinks its
# confidence intervals and inflates its significance stars relative to the human
# fit for no reason other than simulated sample size. A single instance has the
# same number of games per subject as the human data, so the two sides are on
# the same footing.
SINGLE = os.environ.get("GLMM_SINGLE_INSTANCE", "") == "1"
SEED = int(os.environ.get("GLMM_INSTANCE_SEED", "0"))
TAG = ("_single" if SINGLE else "") + ("_commit" if COMMIT else "")

# termination coding in the source data
CODE = {"long": 1, "short": 2}
# the single-horizon predictor set: no termination, no termination interactions
PREDICTORS = ["totevminus", "deltaev", "trial",
              "FA2", "FA2:totevminus", "FA2:deltaev", "FA2:trial"]
LABEL = {"totevminus": r"$ES_{t-1}$", "deltaev": r"$\Delta ES_t$", "trial": "trial",
         "termination": "termination", "FA2:termination": r"FA2 $\times$ termination",
         "FA2:totevminus": r"FA2 $\times$ $ES_{t-1}$",
         "FA2:deltaev": r"FA2 $\times$ $\Delta ES_t$",
         "FA2:trial": r"FA2 $\times$ trial"}
SHOW = ["totevminus", "deltaev", "trial", "FA2:totevminus", "FA2:deltaev", "FA2:trial"]
OUT = os.path.join(R, "BIC_commit" if COMMIT else "BIC", "figures")


def human_pmat(ocir, horizon):
    """Human regressors for one horizon only.

    The raw trials are filtered before the GLM pipeline runs, not afterwards, so
    that the predictors are z-scored within the horizon being analysed. Doing it
    the other way round would z-score against the pooled mean and leave the
    single-horizon subset off-centre.
    """
    hd = pd.read_pickle(os.path.join(R, "data/TrHu_NHB_light/data_MEG",
                                     "behdat_preprocessed.pkl")).copy()
    hd["data"] = hd["data"].apply(
        lambda df: df[df["termination"] == CODE[horizon]].copy())
    betas, ids = fit_glm_separate_for_human_data(hd, ocir_all=ocir)
    return assemble_glm_outputs(betas, ids)["pmat_z_all_df"]


def model_pmat(task, horizon, n_instances):
    """Trial-level regressors from this model's own simulations of this horizon."""
    cfg = load_config(os.path.join(R, "data/simulation_configs",
                                   f"simulation_params_{task}"
                                   f"{'_commit' if COMMIT else ''}.py"))
    raw = os.path.join(cfg.DATA_PATH, "raw_simulations")
    if SINGLE:
        avail = sorted(int(f.split("_")[2]) for f in os.listdir(raw)
                       if f.startswith("sim_run_") and f.endswith(f"_{horizon}.pkl"))
        pick = int(np.random.default_rng(SEED).choice(avail))
        print(f"      single instance {pick} drawn from {len(avail)} available "
              f"(seed {SEED})")
        which = [pick]
    else:
        which = range(n_instances)
    frames = []
    for i in which:
        f = os.path.join(raw, f"sim_run_{i}_{horizon}.pkl")
        if not os.path.exists(f):
            continue
        betas, ids = fit_glm(combine_sequences(pd.read_pickle(f)), source="df")
        for uid, ba in zip(ids, betas):
            if ba is None:
                continue
            # pmat_z, not pmat: the human side comes from pmat_z_all_df, which is
            # z-scored, so using the raw model matrix puts the two sets of
            # coefficients on different scales and makes every model effect look
            # uniformly shrunk by that predictor's standard deviation
            pmat = np.asarray(ba["pmat_z"])
            d = pd.DataFrame(pmat[:, :6], columns=["totevminus", "deltaev", "trial",
                                                   "termination", "totevminusxterm",
                                                   "trialxterm"])
            d["decide"] = np.asarray(ba["decide"]).reshape(-1)
            d["userID"] = str(uid)
            d["_instance"] = i
            frames.append(d)
    if not frames:
        raise SystemExit(f"no simulations for {task} / {horizon}; run the ensembles first")
    return pd.concat(frames, ignore_index=True)


FULL_PREDICTORS = ["totevminus", "deltaev", "trial", "termination",
                   "totevminus:termination", "deltaev:termination",
                   "FA2", "FA2:totevminus", "FA2:deltaev", "FA2:trial",
                   "FA2:termination"]


def model_pmat_combined(short_task, long_task, n_instances):
    """Both horizons from one simulated dataset, z-scored across the pair.

    The single-horizon files must be merged BEFORE fit_glm runs. pmat_z is
    z-scored within whatever is passed to it, so z-scoring each horizon on its
    own leaves termination constant and therefore exactly zero, and concatenating
    those columns afterwards yields a dead regressor whose coefficient is 0 by
    construction rather than by fit.
    """
    frames = []
    for task, horizon in ((short_task, "short"), (long_task, "long")):
        cfg = load_config(os.path.join(R, "data/simulation_configs",
                                       f"simulation_params_{task}"
                                       f"{'_commit' if COMMIT else ''}.py"))
        raw = os.path.join(cfg.DATA_PATH, "raw_simulations")
        if SINGLE:
            avail = sorted(int(f.split("_")[2]) for f in os.listdir(raw)
                           if f.startswith("sim_run_") and f.endswith(f"_{horizon}.pkl"))
            which = [int(np.random.default_rng(SEED).choice(avail))]
        else:
            which = range(n_instances)
        for i in which:
            f = os.path.join(raw, f"sim_run_{i}_{horizon}.pkl")
            if os.path.exists(f):
                frames.append(pd.read_pickle(f))
    if not frames:
        raise SystemExit("no simulations for the combined fit")
    betas, ids = fit_glm(combine_sequences(pd.concat(frames, ignore_index=True)),
                         source="df")
    rows = []
    for uid, ba in zip(ids, betas):
        if ba is None:
            continue
        pm = np.asarray(ba["pmat_z"])
        d = pd.DataFrame(pm[:, :6], columns=["totevminus", "deltaev", "trial",
                                             "termination", "totevminusxterm",
                                             "trialxterm"])
        d["decide"] = np.asarray(ba["decide"]).reshape(-1)
        d["userID"] = str(uid)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    print(f"      termination sd in the merged matrix: {out.termination.std():.3f} "
          f"(zero would mean the horizons were z-scored separately)")
    return out


def fit(pmat, ocir, tag, predictors=None):
    """Fit and return the estimates in the Mean_Estimate/Mean_SE frame that
    plot_glmm_betas_comparison expects, so the per-horizon figures use exactly
    the same style as the combined one already in the manuscript."""
    _, est, _ = fit_glmm(pmat, ocir, predictors=predictors or PREDICTORS)
    e = pd.concat([est["group1_estimates"], est["group2_estimates"]])
    s = pd.concat([est["group1_se"], est["group2_se"]])
    print(f"      {tag:28} pseudo-R2={est['pseudo_r2']:.3f} acc={est['accuracy']:.3f}")
    return pd.DataFrame({"Mean_Estimate": e, "Mean_SE": s})


def main():
    ocir, _ = load_questionnaire_data(R)
    with open(os.path.join(R, BEST, "best_models.json")) as fh:
        best = json.load(fh)

    res = {}
    for horizon in ("short", "long"):
        task = best[horizon]
        print(f"  {horizon}: {task}")
        h = human_pmat(ocir, horizon)
        print(f"      human trials in this horizon: {len(h)}")
        hd_est = fit(h, ocir, "human")

        m = model_pmat(task, horizon, N_INSTANCES)
        print(f"      model trials: {len(m)} over "
              f"{m._instance.nunique()} simulated instances")
        md = fit(m.drop(columns=["_instance"]), ocir, f"model ({task})")
        res[horizon] = dict(task=task, human=hd_est, model=md)

    # combined: both horizons together, so termination varies and is kept
    print("  combined: both horizons")
    hd_all = pd.read_pickle(os.path.join(R, "data/TrHu_NHB_light/data_MEG",
                                         "behdat_preprocessed.pkl"))
    hb, hi = fit_glm_separate_for_human_data(hd_all, ocir_all=ocir)
    h_all = assemble_glm_outputs(hb, hi)["pmat_z_all_df"]
    print(f"      human trials: {len(h_all)}")
    hc = fit(h_all, ocir, "human", FULL_PREDICTORS)
    mc = model_pmat_combined(best["short"], best["long"], N_INSTANCES)
    print(f"      model trials: {len(mc)}")
    mcf = fit(mc, ocir, f"model ({best['short']} + {best['long']})", FULL_PREDICTORS)
    res["combined"] = dict(task=f"{best['short']} + {best['long']}",
                           human=hc, model=mcf)

    os.makedirs(OUT, exist_ok=True)
    for horizon in ("short", "long", "combined"):
        r = res[horizon]
        comb = horizon == "combined"
        plot_glmm_betas_comparison(
            r["human"], r["model"],
            model_label=r["task"],
            main_effects=(["totevminus", "deltaev", "trial", "termination"] if comb
                          else ["totevminus", "deltaev", "trial"]),
            interaction_effects=(["FA2:totevminus", "FA2:deltaev", "FA2:trial",
                                  "FA2:termination"] if comb
                                 else ["FA2:totevminus", "FA2:deltaev", "FA2:trial"]),
            horizon=horizon,
            # the panel letter and the shared caption say which horizon, so the
            # title carries only the model name
            condition_label="",
            path=OUT,
            fname=f"glmm_by_horizon_{horizon}{TAG}",
        )
        print(f"  wrote {OUT}/glmm_by_horizon_{horizon}{TAG}.[pdf|png|svg]")

    tex = [r"\begin{tabular}{llccc}", r"\toprule",
           r"Horizon & Predictor & Human & Model & Difference \\", r"\midrule"]
    for horizon in ("short", "long", "combined"):
        r = res[horizon]
        for p in [q for q in (SHOW + ["termination", "FA2:termination"]) if q in r["human"].index and q in r["model"].index]:
            h_, m_ = r["human"].Mean_Estimate[p], r["model"].Mean_Estimate[p]
            tex.append(f"{horizon.capitalize()} & {LABEL.get(p, p)} & "
                       f"{h_:.3f} & {m_:.3f} & {m_ - h_:.3f} \\\\")
        tex.append(r"\midrule")
    tex = tex[:-1] + [r"\bottomrule", r"\end{tabular}", ""]
    d = os.path.join(R, "BIC_commit" if COMMIT else "BIC", "tables")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"glmm_by_horizon{TAG}.tex")
    open(f, "w").write("\n".join(tex))
    print(f"  wrote {f}")


if __name__ == "__main__":
    main()
