"""Per-subject model selection: does the best-fitting model *structure* vary
across people? (A separate question from how much one fixed model's parameter
*values* vary across people.)

Every model variant is fitted per subject independently
(data/POMDP/<TASK>/<algorithm>/<horizon>/results.pkl, one file per config in
data/simulation_configs/) and carries a per-subject summed log-likelihood
cached in its `after_lls_ga` column -- reused directly
here rather than re-running value_iteration() per subject per candidate
model (would be expensive at dozens-of-configs x 105-subjects scale).

Compares configs *within* each horizon category (S/L/C) only: a subject's
short-horizon trials can only be validly compared across short-horizon-fit
configs (different n_obs/likelihood otherwise).

Also records, per subject, the BIC that horizon's single *aggregate*-best
model (BIC/best_models.json -- picked by pooled sum-logL/shared-n_obs BIC,
see model_comparison.ipynb) gets for that same subject, computed with the
identical per-subject-n_obs formula as winning_BIC. Summing winning_BIC
across subjects gives Sum_i(k_i*log(n_obs_i) - 2*ll_i), the mathematically
exact BIC of "the ensemble of 105 independently-selected model structures";
summing fixed_model_BIC the same way gives the equivalent number for "one
model structure applied to everyone" on identical footing (same per-subject
log/n_obs terms, not the aggregate notebook's single shared-n_obs formula),
isolating the effect of *personalizing model structure* from any
aggregation-formula difference. See notebooks/model_comparison.ipynb's new
"Personalized model selection" section for the resulting comparison.

Usage:
    python scripts/comparison/per_subject_model_selection.py
"""

import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.utils.plotting import compute_metrics, get_best_model

CONFIG_DIR = os.path.join(project_root, "data/simulation_configs")
OUTPUT_DIR = os.path.join(project_root, "data/POMDP")
BEST_MODELS_PATH = os.path.join(project_root, "BIC", "best_models.json")
HORIZON_KEY = {"S": "short", "L": "long", "C": "combined"}


def load_config_module(cfg_path):
    spec = importlib.util.spec_from_file_location("sim_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OVERRIDES


def build_task_from_filename(cfg_path):
    """TASK is embedded in the filename: simulation_params_<TASK>.py."""
    base = os.path.basename(cfg_path)
    return base[len("simulation_params_"):-len(".py")]


def results_path_for(task, overrides):
    is_commit = overrides.get("POMDP_COMMIT", False)
    subdir = "POMDP_commit" if is_commit else "POMDP"
    fit_horizon = overrides.get("FIT_HORIZON", ["short"])
    # The optimizer directory is part of the results path, so this must follow
    # SIM_ALGORITHM rather than assume "ga": models fitted only with DE (the
    # ablation cells among them) are otherwise silently skipped as "no
    # results.pkl", which drops them from the personalized comparison and can
    # leave the fixed-model column blank.
    algorithm = os.environ.get("SIM_ALGORITHM", "de")
    if len(fit_horizon) == 1:
        return os.path.join(project_root, "data", subdir, task, algorithm, fit_horizon[0], "results.pkl")
    return os.path.join(project_root, "data", subdir, task, algorithm, "results.pkl")


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


def main():
    with open(BEST_MODELS_PATH) as f:
        best_models = json.load(f)

    config_files = sorted(glob.glob(os.path.join(CONFIG_DIR, "simulation_params_*.py")))
    config_files = [f for f in config_files if not f.endswith("_commit.py")]

    # {horizon_prefix: [(task, n_params, results_df), ...]}
    by_horizon = {"S": [], "L": [], "C": []}
    skipped = []
    for cfg_path in config_files:
        task = build_task_from_filename(cfg_path)
        overrides = load_config_module(cfg_path)
        if overrides.get("POMDP_COMMIT", False):
            continue  # non-commit only, matching the BIC convention used elsewhere
        results_pkl = results_path_for(task, overrides)
        if not os.path.exists(results_pkl):
            skipped.append((task, "no results.pkl"))
            continue
        try:
            results_df = pd.read_pickle(results_pkl)
            if "after_lls_ga" not in results_df.columns or "data_dict_of_lists" not in results_df.columns:
                skipped.append((task, "missing after_lls_ga/data_dict_of_lists"))
                continue
        except Exception as e:
            skipped.append((task, f"load error: {e}"))
            continue
        n_params = len(overrides.get("PARAM_RANGES", {}))
        by_horizon[task[0]].append((task, n_params, results_df))

    print(f"Loaded {sum(len(v) for v in by_horizon.values())} configs "
          f"(S={len(by_horizon['S'])}, L={len(by_horizon['L'])}, C={len(by_horizon['C'])}); "
          f"skipped {len(skipped)}: {skipped}")

    for horizon_prefix, configs in by_horizon.items():
        if len(configs) < 2:
            print(f"\n{horizon_prefix}: fewer than 2 usable configs, skipping model selection.")
            continue

        print(f"\n{horizon_prefix}: comparing {len(configs)} configs "
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
            print(f"  Note: aggregate-best model {fixed_task!r} for {horizon_prefix} not in this "
                  f"config set -- fixed_model_BIC will be left blank.")

        rows = []
        for sid, task_metrics in subject_metrics.items():
            if len(task_metrics) < 2:
                continue  # subject missing from most configs -- not a fair comparison
            winner = get_best_model(task_metrics, criterion="BIC")
            sorted_tasks = sorted(task_metrics.items(), key=lambda kv: kv[1]["BIC"])
            runner_up_task, runner_up_metrics = sorted_tasks[1] if len(sorted_tasks) > 1 else (None, None)
            fixed_metrics = task_metrics.get(fixed_task) if has_fixed else None
            rows.append(
                {
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
            )

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUTPUT_DIR, f"per_subject_model_selection_{horizon_prefix}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(out_df)} subjects)")

        # The full subject x model BIC matrix, which the selection above reduces
        # to one winner per subject. Random-effects model comparison needs the
        # whole matrix, so write it out rather than recomputing it elsewhere.
        mat = pd.DataFrame(
            {task: {sid: m[task]["BIC"] for sid, m in subject_metrics.items()
                    if task in m}
             for task, _, _ in configs}
        )
        mat.index.name = "subject_ID"
        mat_path = os.path.join(OUTPUT_DIR,
                                f"per_subject_bic_matrix_{horizon_prefix}.csv")
        mat.to_csv(mat_path)
        print(f"  Saved: {mat_path} ({mat.shape[0]} subjects x {mat.shape[1]} models, "
              f"{int(mat.isna().sum().sum())} missing)")

        counts = out_df["winning_task"].value_counts()
        print(f"  Winning-model counts ({horizon_prefix}):")
        for task, count in counts.items():
            n_params = next(np for t, np, _ in configs if t == task)
            print(f"    {task} ({n_params} params): {count} subjects")

        if has_fixed:
            both = out_df.dropna(subset=["fixed_model_BIC"])
            cum_personalized = both["winning_BIC"].sum()
            cum_fixed = both["fixed_model_BIC"].sum()
            print(f"  Cumulative BIC ({horizon_prefix}, n={len(both)} subjects, "
                  f"per-subject-summed convention):")
            print(f"    Personalized selection: {cum_personalized:.2f}")
            print(f"    Fixed aggregate-best ({fixed_task}): {cum_fixed:.2f}")
            print(f"    Delta (fixed - personalized): {cum_fixed - cum_personalized:+.2f} "
                  f"({'personalization wins' if cum_fixed > cum_personalized else 'fixed model wins'})")
            n_personalized_better = int((both["fixed_model_BIC"] - both["winning_BIC"] > 2).sum())
            print(f"    Subjects with >2 BIC improvement from personalization "
                  f"(positive evidence, Kass & Raftery): {n_personalized_better}/{len(both)}")


if __name__ == "__main__":
    main()
