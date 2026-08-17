"""Per-subject model selection, commit-likelihood counterpart to
per_subject_model_selection.py.

Same question (does the best-fitting model *structure* vary across people?),
but run over the commit-fit ladder (data/POMDP_commit/, configs matching
simulation_params_*_commit.py, POMDP_COMMIT=True -- these were fit directly
against log_likelihood_commit, i.e. GA already optimized decide-vs-wait, not
the full-fit (yellow/blue/wait) action likelihood). Not mixed with the non-commit sweep: the
two likelihoods live on different numeric scales, so BIC only compares
fairly within one convention at a time -- see per_subject_model_selection.py's
own skip-commit-configs guard for the mirror image of this restriction.

For the combined ("C") horizon group only, GLM is added as one more
per-subject candidate alongside every POMDP(commit) config. GLM is only ever
fit jointly across both horizons (no separate-horizon GLM exists -- see
model_comparison_commit.ipynb's cell 10), so it cannot enter the S/L groups.
Reuses the exact per-subject GLM log-likelihood/BIC convention already
established in model_comparison_commit.ipynb (cells 11 and 21):
fit_glm_separate_for_human_data + compute_full_per_draw_probabilities,
scored on ALL raw draws (not just the valid_mask-filtered subset llf uses),
N_PARAMS_GLM=7, BIC = 7*log(n_obs_raw) - 2*ll_raw.

Also records, per subject, the BIC of that horizon's single *aggregate*-best
commit model (BIC_commit/best_models.json), computed with the identical
per-subject-n_obs formula as winning_BIC, so cumulative sums are directly
comparable (see per_subject_model_selection.py's docstring for the exact
rationale -- same convention, commit-likelihood counterpart).

Usage:
    python scripts/comparison/per_subject_model_selection_commit.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.config.loader import load_config
from src.glm import fit_glm_separate_for_human_data
from src.glm.glm import compute_full_per_draw_probabilities
from src.utils.data_handling import load_fitted_results
from src.utils.plotting import compute_metrics, get_best_model

CONFIG_DIR = os.path.join(project_root, "data/simulation_configs")
OUTPUT_DIR = os.path.join(project_root, "data/POMDP_commit")
BEST_MODELS_PATH = os.path.join(project_root, "BIC_commit", "best_models.json")
HORIZON_KEY = {"S": "short", "L": "long", "C": "combined"}
N_PARAMS_GLM = 7  # 6 regressors + intercept, matching model_comparison_commit.ipynb


def per_subject_n_obs(data_dict_of_lists):
    # Forgetting-model rows store this as a pandas Series (Series.values is
    # a property, not a method, unlike dict.values()); every model type
    # (short/long/combined, forgetting or not) is otherwise horizon-keyed
    # the same way, so this is the only branch needed -- see plotting.py's
    # _per_subject_n_obs for the identical fix.
    horizon_dfs = data_dict_of_lists.values() if isinstance(data_dict_of_lists, dict) else data_dict_of_lists
    return sum(
        len(seq)
        for horizon_df in horizon_dfs
        for seq in horizon_df["draw_yellow_blue_action_outcome"].values
    )


def build_full_decisions(row_data, games_lengths):
    """Binary decision vector aligned to games_lengths, covering ALL raw draws
    (not valid_mask-filtered). y=0 on every draw before the decision, y=1 at
    the decision draw, or all zeros if the subject never decided in that
    game. Verbatim from model_comparison_commit.ipynb's cell 11."""
    decisions = []
    for (_, game_data), game_len in zip(row_data.groupby(["block", "game"]), games_lengths):
        game_dec = np.zeros(game_len, dtype=float)
        decision_index = game_data["choiceTrial"].first_valid_index()
        if decision_index is not None:
            decision_pos = game_data.index.get_loc(decision_index)
            if decision_pos < game_len:
                game_dec[decision_pos] = 1.0
        decisions.append(game_dec)
    return np.concatenate(decisions)


def compute_glm_per_subject_bic():
    """Fit the combined GLM once and return {subject_ID: {"BIC": ...}},
    matching model_comparison_commit.ipynb's cells 11/21 exactly."""
    human_data_path = os.path.join(
        project_root, "data/TrHu_NHB_light/data_MEG/behdat_preprocessed.pkl"
    )
    glm_human_data = pd.read_pickle(human_data_path)

    print("Fitting GLM (combined, jointly across both horizons)...")
    betas_glm_combined, _ = fit_glm_separate_for_human_data(glm_human_data, ocir_all=None)

    glm_by_id = {
        b["id"]: b for b in betas_glm_combined
        if b is not None and not np.isnan(b["pdecide_beta"]).any() and not np.isnan(b["llf"])
    }
    print(f"  {len(glm_by_id)}/{len(betas_glm_combined)} subjects with a valid GLM fit.")

    glm_metrics = {}
    for uid, b in glm_by_id.items():
        row_data = glm_human_data.loc[glm_human_data["userID"] == uid, "data"].iloc[0]
        p = compute_full_per_draw_probabilities(
            row_data, b["games_lengths"], b["mu"], b["sigma"], b["pdecide_beta"]
        )
        y = build_full_decisions(row_data, b["games_lengths"])
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        ll_raw = float(np.sum(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))
        n_obs_raw = int(sum(b["games_lengths"]))
        glm_metrics[uid] = compute_metrics(ll_raw, N_PARAMS_GLM, n_obs_raw)

    return glm_metrics


def main():
    with open(BEST_MODELS_PATH) as f:
        best_models = json.load(f)

    results_df_dict, _ = load_fitted_results(CONFIG_DIR, commit=True)

    # {horizon_prefix: [(task, n_params, results_df), ...]}
    by_horizon = {"S": [], "L": [], "C": []}
    skipped = []
    for task, results_df in results_df_dict.items():
        cfg_path = os.path.join(CONFIG_DIR, f"simulation_params_{task}_commit.py")
        cfg = load_config(cfg_path)
        if "after_lls_ga" not in results_df.columns or "data_dict_of_lists" not in results_df.columns:
            skipped.append((task, "missing after_lls_ga/data_dict_of_lists"))
            continue
        n_params = len(results_df["fit_params_ga"].iloc[0])
        by_horizon[task[0]].append((task, n_params, results_df))

    print(f"Loaded {sum(len(v) for v in by_horizon.values())} commit configs "
          f"(S={len(by_horizon['S'])}, L={len(by_horizon['L'])}, C={len(by_horizon['C'])}); "
          f"skipped {len(skipped)}: {skipped}")

    glm_metrics_by_subject = compute_glm_per_subject_bic()

    for horizon_prefix, configs in by_horizon.items():
        if len(configs) < 2:
            print(f"\n{horizon_prefix}: fewer than 2 usable configs, skipping model selection.")
            continue

        # GLM is reported as a reference point for the combined horizon (row["GLM_BIC"]
        # below), but is deliberately NOT added to subject_metrics -- personalized
        # model selection here only ever chooses among POMDP(commit) structures, never
        # GLM itself, so "personalized" cannot trivially collapse to "GLM won for most
        # subjects" and can be validly compared against GLM as an independent model.
        include_glm = horizon_prefix == "C"
        print(f"\n{horizon_prefix}: comparing {len(configs)} commit configs"
              f"{' (GLM reported as reference only, not a selectable candidate)' if include_glm else ''} "
              f"({', '.join(t for t, _, _ in configs)})")

        # subject_ID -> {task -> {"BIC": ..., "n_params": ..., "ll": ...}}
        subject_metrics = {}
        for task, n_params, results_df in configs:
            for _, row in results_df.iterrows():
                sid = row["subject_ID"]
                n_obs = per_subject_n_obs(row["data_dict_of_lists"])
                if n_obs == 0:
                    continue
                m = compute_metrics(row["after_lls_ga"], n_params, n_obs)
                subject_metrics.setdefault(sid, {})[task] = m

        fixed_task = best_models.get(HORIZON_KEY[horizon_prefix])
        has_fixed = fixed_task is not None and any(t == fixed_task for t, _, _ in configs)
        if not has_fixed:
            print(f"  Note: aggregate-best commit model {fixed_task!r} for {horizon_prefix} not in "
                  f"this config set -- fixed_model_BIC will be left blank.")

        rows = []
        for sid, task_metrics in subject_metrics.items():
            if len(task_metrics) < 2:
                continue  # subject missing from most configs -- not a fair comparison
            winner = get_best_model(task_metrics, criterion="BIC")
            sorted_tasks = sorted(task_metrics.items(), key=lambda kv: kv[1]["BIC"])
            runner_up_task, runner_up_metrics = sorted_tasks[1] if len(sorted_tasks) > 1 else (None, None)
            fixed_metrics = task_metrics.get(fixed_task) if has_fixed else None
            row = {
                "subject_ID": sid,
                "n_configs_compared": len(task_metrics),
                "winning_task": winner,
                "winning_BIC": task_metrics[winner]["BIC"],
                "runner_up_task": runner_up_task,
                "runner_up_BIC": runner_up_metrics["BIC"] if runner_up_metrics else np.nan,
                "BIC_margin": (runner_up_metrics["BIC"] - task_metrics[winner]["BIC"]) if runner_up_metrics else np.nan,
                "fixed_model_task": fixed_task if has_fixed else None,
                "fixed_model_BIC": fixed_metrics["BIC"] if fixed_metrics else np.nan,
            }
            if include_glm:
                row["GLM_BIC"] = glm_metrics_by_subject.get(sid, {}).get("BIC", np.nan)
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUTPUT_DIR, f"per_subject_model_selection_commit_{horizon_prefix}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(out_df)} subjects)")

        counts = out_df["winning_task"].value_counts()
        print(f"  Winning-model counts ({horizon_prefix}, POMDP(commit)-only selection):")
        for task, count in counts.items():
            n_params = next(np for t, np, _ in configs if t == task)
            print(f"    {task} ({n_params} params): {count} subjects")

        if include_glm:
            n_glm_better = int((out_df["GLM_BIC"] < out_df["winning_BIC"]).sum())
            print(f"  GLM (reference only) has a lower BIC than the personalized POMDP(commit) "
                  f"winner for {n_glm_better}/{len(out_df)} subjects.")

        if has_fixed:
            subset_cols = ["fixed_model_BIC"] + (["GLM_BIC"] if include_glm else [])
            both = out_df.dropna(subset=subset_cols)
            cum_personalized = both["winning_BIC"].sum()
            cum_fixed = both["fixed_model_BIC"].sum()
            print(f"  Cumulative BIC ({horizon_prefix}, n={len(both)} subjects, "
                  f"per-subject-summed convention):")
            print(f"    Personalized selection (POMDP(commit) candidates only): "
                  f"{cum_personalized:.2f}")
            print(f"    Fixed aggregate-best commit model ({fixed_task}): {cum_fixed:.2f}")
            print(f"    Delta (fixed - personalized): {cum_fixed - cum_personalized:+.2f} "
                  f"({'personalization wins' if cum_fixed > cum_personalized else 'fixed model wins'})")
            n_personalized_better = int((both["fixed_model_BIC"] - both["winning_BIC"] > 2).sum())
            print(f"    Subjects with >2 BIC improvement from personalization "
                  f"(positive evidence, Kass & Raftery): {n_personalized_better}/{len(both)}")
            if include_glm:
                cum_glm = both["GLM_BIC"].sum()
                print(f"    GLM alone (applied to everyone): {cum_glm:.2f}")


if __name__ == "__main__":
    main()
