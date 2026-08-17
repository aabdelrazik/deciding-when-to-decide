"""Commit-vs-wait log-likelihood for the best-fitting POMDP models.

log_likelihood() scores the full 3-way action (yellow / blue / wait) at every
decision point. Magda's GLM only ever predicts "decide" (commit to Y or B,
collapsing yellow/blue together) vs. "wait" -- so a BIC built on
log_likelihood() is not on equal footing with the GLM's BIC.

POMDP.log_likelihood_commit() scores the same action sequences but collapses
yellow/blue into a single "decide" probability (p_y + p_b), exactly matching
what the GLM predicts. This script re-evaluates each best model's
already-fitted per-subject parameters (no re-fitting) under
log_likelihood_commit, using the same data each subject's GA fit originally
used (data_dict_of_lists), and saves the resulting per-subject values so the
model_comparison notebook can build a GLM vs. POMDP(commit) BIC comparison.

Usage:
    python scripts/comparison/compute_bic_commit.py
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.pomdp import POMDPFactory
from src.utils.plotting import compute_metrics_per_subject_summed

HORIZON_SUBDIR = {"short": "short", "long": "long", "combined": None}
MAX_CARDS_PER_DRAW = 5
OUTPUT_PATH = os.path.join(project_root, "BIC_commit", "bic_commit.pkl")


def load_param_order(task):
    """PARAM_RANGES key order (dict insertion order) from the model's saved config."""
    cfg_path = os.path.join(
        project_root, "data/simulation_configs", f"simulation_params_{task}.py"
    )
    spec = importlib.util.spec_from_file_location("sim_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERRIDES["PARAM_RANGES"].keys())


def compute_commit_lls_for_model(task, horizon_key):
    """Re-evaluate every subject's already-fitted params under
    log_likelihood_commit, on the same data their fit was scored on.

    The optimizer is part of the output path, so this has to follow
    SIM_ALGORITHM rather than assume one: with the default 'de' it would
    otherwise read a 'ga' directory that a reviewer never produced.
    """
    algorithm = os.environ.get("SIM_ALGORITHM", "de")
    base = os.path.join(project_root, "data/POMDP", task, algorithm)
    subdir = HORIZON_SUBDIR[horizon_key]
    data_path = os.path.join(base, subdir) if subdir else base

    results_df = pd.read_pickle(os.path.join(data_path, "results.pkl"))
    param_order = load_param_order(task)

    horizon_list = sorted(
        {h for d in results_df["data_dict_of_lists"].tolist() for h in d.keys()}
    )

    after_lls_commit = []
    for _, row in results_df.iterrows():
        base_params = dict(zip(param_order, row["fit_params_ga"]))
        subject_ll = 0.0
        for horizon in horizon_list:
            params = {
                **base_params,
                "verbose": False,
                "max_cards_per_draw": MAX_CARDS_PER_DRAW,
                "horizon_condition": horizon,
            }
            if "is_hazardous" in params:
                params["is_hazardous"] = bool(round(params["is_hazardous"]))

            pomdp = POMDPFactory("exaggerate")
            pomdp.__init__(**params)
            pomdp.value_iteration()

            df = row["data_dict_of_lists"][horizon]
            subject_ll += pomdp.log_likelihood_commit(df)

        after_lls_commit.append(subject_ll)

    out_df = results_df[["subject_ID", "fit_params_ga"]].copy()
    out_df["after_lls_commit"] = after_lls_commit

    # Per-subject-summed convention (see compute_metrics_per_subject_summed):
    # each subject's own n_obs, not one pooled total, matching the fix
    # applied to the full-fit aggregate comparison (Tables 6-9).
    metrics_df = results_df[["data_dict_of_lists", "fit_params_ga"]].copy()
    metrics_df["after_lls_commit"] = after_lls_commit
    metrics = compute_metrics_per_subject_summed(metrics_df, ll_col="after_lls_commit")

    return out_df, metrics


def main():
    with open(os.path.join(project_root, "BIC_commit/best_models.json")) as f:
        best_models = json.load(f)

    bic_commit = {}
    summary_rows = []
    missing = []
    for horizon_key, task in best_models.items():
        print(f"Evaluating commit log-likelihood for POMDP {task} ({horizon_key})...")
        try:
            out_df, metrics = compute_commit_lls_for_model(task, horizon_key)
        except FileNotFoundError as exc:
            # A partial model set is a normal state: someone reproducing one
            # horizon should still get that horizon rather than a traceback.
            missing.append(f"{task} ({horizon_key})")
            print(f"  skipped, no fit at {os.path.relpath(exc.filename, project_root)}")
            continue
        bic_commit[horizon_key] = {"task": task, "per_subject": out_df, **metrics}
        summary_rows.append({"horizon": horizon_key, "task": task, **metrics})
        print(f"  sum logL={metrics['sum logL']:.2f}  BIC={metrics['BIC']:.2f}")

    if not summary_rows:
        raise SystemExit("no fitted model was found for any horizon; fit at least "
                         "one commit likelihood model first")
    if missing:
        print(f"\nnot evaluated, no fit present: {', '.join(missing)}")

    with open(OUTPUT_PATH, "wb") as f:
        pd.to_pickle(bic_commit, f)
    print(f"\nSaved per-subject commit log-likelihoods to {OUTPUT_PATH}")

    summary = pd.DataFrame(summary_rows)
    print()
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
