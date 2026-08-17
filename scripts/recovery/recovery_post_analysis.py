"""Per-config recovery diagnostics, extracted from recovery_post_analysis.ipynb.

Runs against whichever config SIM_CONFIG_PATH points to (same convention as
fit_data.py / run_array.sh) and saves figures to that config's FIGURE_PATH:
  - human vs. simulated draws/outcomes
  - true vs. recovered fitted parameters
  - recovery + log-likelihood summary
  - parameter recovery correlations
  - fitted-parameter correlation matrix (+ scatter)
  - simulated-vs-fit comparison
  - all-subjects ensemble (draws/outcomes), using the cached
    ensemble_metrics_summary.csv / ensemble_distribution_data.pkl / raw_simulations/
    written by fit_data.py if present, instead of resimulating 300 runs.

Usage:
    SIM_CONFIG_PATH=/abs/path/to/simulation_params_XYZ.py python recovery_post_analysis.py
"""

import glob
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys

sys.path.append(project_root)

from src.config import CONFIG
from src.params_fitting.data_simulation import simulate_data
from src.utils import (
    extract_draws_for_subject,
    extract_hist_data,
    extract_hist_data_for_user,
    plot_all_subjects_ensemble,
    plot_fitted_param_correlations,
    plot_fitted_param_correlations_scatter,
    plot_ensemble_r2_metrics,
    plot_human_vs_simulated_data,
    plot_param_correlations,
    plot_recovery_and_ll,
    plot_sim_vs_fit,
    plot_true_vs_recovered_params,
)


def load_ocd_userids():
    ocir_data = pd.read_csv(
        os.path.join(project_root, "data/TrHu_NHB_light/data_MEG/fa_scores.csv")
    )
    ybocs_data = pd.read_csv(
        os.path.join(project_root, "data/TrHu_NHB_light/data_MEG/ybocs_scores.csv")
    )
    ocd_userID = ybocs_data[ybocs_data.drop(columns="userID").notna().all(axis=1)][
        "userID"
    ]
    ocd_userID = ocd_userID[
        ybocs_data[ybocs_data["userID"].isin(ocd_userID)]["YBOCS_obsess_subtotal"] > 11
    ].tolist()
    return ocd_userID


def compute_subject_metrics(
    run_idx, userId_list, results_df, human_data, bins, fit_horizon, raw_sim_dir
):
    """One ensemble run: simulate, save the raw trial-by-trial data, bin per subject."""
    all_simulated_data, _ = simulate_data(results_df, sim_same_data=True)

    horizon_str = fit_horizon[0]
    sim_file_path = os.path.join(raw_sim_dir, f"sim_run_{run_idx}_{horizon_str}.pkl")
    all_simulated_data.to_pickle(sim_file_path)

    outcome_bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    records = []
    for target_id in userId_list:
        sim_draws = extract_draws_for_subject(all_simulated_data, target_id)
        outcome_sim = all_simulated_data[all_simulated_data["userID"] == target_id][
            "outcome"
        ].values

        short_draws, long_draws, short_rewards, long_rewards = (
            extract_hist_data_for_user(human_data, target_id)
        )
        if fit_horizon == ["short"]:
            human_draws, human_outcome = short_draws, short_rewards
        elif fit_horizon == ["long"]:
            human_draws, human_outcome = long_draws, long_rewards
        else:
            human_draws, human_outcome = (
                short_draws + long_draws,
                short_rewards + long_rewards,
            )

        sim_counts, _ = np.histogram(sim_draws, bins=bins)
        human_counts, _ = np.histogram(human_draws, bins=bins)
        sim_outcome_counts, _ = np.histogram(outcome_sim, bins=outcome_bins)
        human_outcome_counts, _ = np.histogram(human_outcome, bins=outcome_bins)

        r2_draws = r2_score(human_counts, sim_counts)
        r2_outcome = r2_score(human_outcome_counts, sim_outcome_counts)

        records.append(
            {
                "userID": target_id,
                "r2_draws": r2_draws,
                "r2_outcome": r2_outcome,
                "sim_counts": sim_counts,
                "human_counts": human_counts,
                "sim_outcome_counts": sim_outcome_counts,
                "human_outcome_counts": human_outcome_counts,
            }
        )
    return records


def build_ensemble_data(cfg, results_df, human_data, userId_list, num_runs=300):
    """Mirrors fit_data.py's ensemble step exactly, including its cache files,
    so a prior fit_data.py run's raw_simulations/ + summary files are reused
    instead of resimulating."""
    metrics_save_path = os.path.join(cfg.DATA_PATH, "ensemble_metrics_summary.csv")
    ensemble_save_path = os.path.join(cfg.DATA_PATH, "ensemble_distribution_data.pkl")
    bins = np.arange(1, 14, 1)

    raw_sim_dir = os.path.join(cfg.DATA_PATH, "raw_simulations")
    horizon_str = cfg.FIT_HORIZON[0]
    raw_sims_complete = (
        os.path.isdir(raw_sim_dir)
        and len(glob.glob(os.path.join(raw_sim_dir, f"sim_run_*_{horizon_str}.pkl")))
        >= num_runs
    )

    if (
        os.path.exists(metrics_save_path)
        and os.path.exists(ensemble_save_path)
        and raw_sims_complete
    ):
        print(
            "Metrics, ensemble data, and raw simulations already exist. Loading from disk..."
        )
        metrics_df = pd.read_csv(metrics_save_path)
        with open(ensemble_save_path, "rb") as f:
            ensemble_data = pickle.load(f)
        return metrics_df, ensemble_data

    os.makedirs(raw_sim_dir, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Runs", total=num_runs)):
        total_records = Parallel(n_jobs=-1, verbose=1)(
            delayed(compute_subject_metrics)(
                i,
                userId_list,
                results_df,
                human_data,
                bins,
                cfg.FIT_HORIZON,
                raw_sim_dir,
            )
            for i in range(num_runs)
        )

    metrics, ensemble_data = [], {}
    for target_id in userId_list:
        user_records = [
            r[i]
            for r in total_records
            for i in range(len(r))
            if r[i]["userID"] == target_id
        ]
        r2_draws_values = [r["r2_draws"] for r in user_records]
        r2_outcome_values = [r["r2_outcome"] for r in user_records]
        metrics.append(
            {
                "userID": target_id,
                "r2_draws": np.mean(r2_draws_values),
                "r2_draws_var": np.var(r2_draws_values),
                "r2_outcome": np.mean(r2_outcome_values),
                "r2_outcome_var": np.var(r2_outcome_values),
            }
        )

        all_sim_counts = np.vstack([r["sim_counts"] for r in user_records])
        all_sim_outcomes = np.vstack([r["sim_outcome_counts"] for r in user_records])
        ensemble_data[target_id] = {
            "human_counts": user_records[0]["human_counts"],
            "avg_sim_counts": np.mean(all_sim_counts, axis=0),
            "std_sim_counts": np.std(all_sim_counts, axis=0),
            "human_outcomes": user_records[0]["human_outcome_counts"],
            "avg_sim_outcomes": np.mean(all_sim_outcomes, axis=0),
            "std_sim_outcomes": np.std(all_sim_outcomes, axis=0),
        }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_save_path, index=False)
    with open(ensemble_save_path, "wb") as f:
        pickle.dump(ensemble_data, f)
    return metrics_df, ensemble_data


def run(cfg):
    os.makedirs(cfg.FIGURE_PATH, exist_ok=True)
    horizon = "combined" if len(cfg.FIT_HORIZON) == 2 else cfg.FIT_HORIZON[0]
    print(f"TASK={cfg.TASK}  POMDP_TYPE={cfg.POMDP_TYPE}  horizon={horizon}")

    human_data = pd.read_pickle(cfg.HUMAN_DATA_PATH)
    human_data_filtered = human_data[: cfg.N_SUBJECTS]
    results_df = pd.read_pickle(cfg.RESULTS_PATH)
    all_simulated_data = pd.read_pickle(cfg.FULL_SIM_DF_PATH)
    # Forgetting models never get a recovery refit from fit_data.py (replaying
    # the gamma-discounted sequences stored in results.pkl would double-discount
    # them), so the recovery-dependent panels are simply skipped when its output
    # is absent rather than failing the whole run: the fitted-parameter
    # distributions, the human-vs-simulated comparison and the ensemble fit all
    # come from results.pkl alone and are still worth having.
    has_recovery = os.path.exists(cfg.RESULTS_RECOVERED_PATH) and os.path.exists(
        cfg.FULL_SIM_DF_RECOVERED_PATH
    )
    results_df_recovered = (
        pd.read_pickle(cfg.RESULTS_RECOVERED_PATH) if has_recovery else None
    )
    simulated_subject_dfs_recovered = (
        pd.read_pickle(cfg.FULL_SIM_DF_RECOVERED_PATH) if has_recovery else None
    )
    if not has_recovery:
        print("no recovery fit found; plotting the panels that do not need one")

    short_draws_humans, long_draws_humans, short_rewards_humans, long_rewards_humans = (
        extract_hist_data(human_data_filtered)
    )
    num_draws_human = short_draws_humans + long_draws_humans
    outcome_human = long_rewards_humans + short_rewards_humans

    outcome_simulated = all_simulated_data["outcome"].values
    num_draws_simulated = np.array(
        [len(ev) for ev in all_simulated_data["ev"].tolist()], dtype=int
    )

    # 1. Human vs. simulated draws/outcomes
    if horizon == "combined":
        plot_human_vs_simulated_data(
            outcome_human,
            num_draws_human,
            outcome_simulated,
            num_draws_simulated,
            horizon="combined",
            path=cfg.FIGURE_PATH,
        )
    elif cfg.FIT_HORIZON == ["short"]:
        plot_human_vs_simulated_data(
            short_rewards_humans,
            short_draws_humans,
            outcome_simulated,
            num_draws_simulated,
            horizon="short",
            path=cfg.FIGURE_PATH,
        )
    else:
        plot_human_vs_simulated_data(
            long_rewards_humans,
            long_draws_humans,
            outcome_simulated,
            num_draws_simulated,
            horizon="long",
            path=cfg.FIGURE_PATH,
        )

    # 2. True vs. recovered parameters
    plot_true_vs_recovered_params(
        results_df,
        cfg.PARAM_ORDER,
        results_df_recovered,
        horizon=horizon,
        path=cfg.FIGURE_PATH,
    )

    if has_recovery:
        # 3. Recovery + log-likelihood summary
        true_params_all = np.array(list(results_df["fit_params_ga"]))
        fit_params_ga = np.array(results_df_recovered["fit_params_ga"].tolist())
        after_lls_ga = results_df_recovered["after_lls_ga"].tolist()
        ocd_userID = load_ocd_userids()
        plot_recovery_and_ll(
            true_params_all,
            fit_params_ga,
            after_lls_ga,
            results_df,
            results_df_recovered,
            cfg.PARAM_ORDER,
            cfg.PARAM_RANGES,
            ocd_userID=ocd_userID,
            font_size=20,
            plot_regression=True,
            horizon=horizon,
            path=cfg.FIGURE_PATH,
        )

        # 4. Parameter recovery correlations
        plot_param_correlations(
            true_params_all,
            fit_params_ga,
            cfg.PARAM_ORDER,
            method_name="Recovered",
            horizon=horizon,
            path=cfg.FIGURE_PATH,
        )

        # 5. Fitted-parameter correlation matrix (+ scatter)
        # is_latex not passed here -- both default to src.utils.plotting.IS_LATEX,
        # the single global switch, rather than a locally hardcoded value.
        fit_params_dict = {"Recovered": fit_params_ga}
        plot_fitted_param_correlations(
            fit_params_dict,
            cfg.PARAM_ORDER,
            font_size=20,
            horizon=horizon,
            path=cfg.FIGURE_PATH,
        )
        plot_fitted_param_correlations_scatter(
            fit_params_dict,
            cfg.PARAM_ORDER,
            font_size=20,
            horizon=horizon,
            path=cfg.FIGURE_PATH,
        )

        # 6. Simulated (original) vs. simulated-from-recovered-fit
        outcome_simulated_fit = simulated_subject_dfs_recovered["outcome"].values
        num_draws_simulated_fit = np.array(
            [len(ev) for ev in simulated_subject_dfs_recovered["ev"].tolist()]
        )
        plot_sim_vs_fit(
            outcome_simulated_fit,
            num_draws_simulated_fit,
            outcome_simulated,
            num_draws_simulated,
            path=cfg.FIGURE_PATH,
        )

    # 7. All-subjects ensemble (uses fit_data.py's cached ensemble files/raw_simulations if present)
    userId_list = human_data["userID"].unique().tolist()
    metrics_df, ensemble_data = build_ensemble_data(
        cfg, results_df, human_data, userId_list
    )
    plot_all_subjects_ensemble(ensemble_data, horizon=horizon, path=cfg.FIGURE_PATH)

    # 8. Per-subject R^2 (draws & outcome) goodness-of-fit, mean +/- SD across the ensemble
    plot_ensemble_r2_metrics(metrics_df, horizon=horizon, path=cfg.FIGURE_PATH)

    print(f"Figures saved to: {cfg.FIGURE_PATH}")


if __name__ == "__main__":
    run(CONFIG)
