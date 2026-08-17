"""Build the ensemble-simulation cache (raw_simulations/, ensemble_metrics_summary.csv,
ensemble_distribution_data.pkl) for a forgetting-model config, mirroring fit_data.py's
tail section exactly (same compute_subject_metrics logic, same num_runs=300, same
output paths/filenames) but starting from an already-fitted results.pkl instead of
running the GA fit itself, and WITHOUT the parameter-recovery refit step (fit_data.py
doesn't have one either; fit_data_forgetting.py's own recovery-refit is explicitly
skipped for POMDP_TYPE=="forgetting" and is not built here).

Usage:
    SIM_CONFIG_PATH=/abs/path/to/simulation_params_XYZ.py python build_ensemble_forgetting.py
"""
import sys
import os
import glob
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.metrics import r2_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.params_fitting.data_simulation import simulate_data
from src.config import *
from src.utils import extract_draws_for_subject, extract_hist_data_for_user

assert POMDP_TYPE == "forgetting", f"expected a forgetting config, got POMDP_TYPE={POMDP_TYPE!r}"

data_path = DATA_PATH
results_path = RESULTS_PATH
human_data = pd.read_pickle(HUMAN_DATA_PATH)
userId_list = human_data["userID"].unique().tolist()
results_df = pd.read_pickle(results_path)

bins = np.arange(1, 14, 1)
metrics_save_path = os.path.join(data_path, "ensemble_metrics_summary.csv")
ensemble_save_path = os.path.join(data_path, "ensemble_distribution_data.pkl")


def compute_subject_metrics(run_idx, userId_list, results_df, human_data, bins, fit_horizon, raw_sim_dir):
    all_simulated_data, _ = simulate_data(results_df, sim_same_data=True)

    horizon_str = fit_horizon[0]
    sim_file_path = os.path.join(raw_sim_dir, f"sim_run_{run_idx}_{horizon_str}.pkl")
    all_simulated_data.to_pickle(sim_file_path)

    records = []
    outcome_bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    for target_id in userId_list:
        sim_draws = extract_draws_for_subject(all_simulated_data, target_id)
        outcome_sim = all_simulated_data[all_simulated_data["userID"] == target_id]["outcome"].values

        short_draws, long_draws, short_rewards, long_rewards = extract_hist_data_for_user(human_data, target_id)
        if fit_horizon == ["short"]:
            human_draws = short_draws
            human_outcome = short_rewards
        elif fit_horizon == ["long"]:
            human_draws = long_draws
            human_outcome = long_rewards
        else:
            human_draws = short_draws + long_draws
            human_outcome = short_rewards + long_rewards

        sim_counts, _ = np.histogram(sim_draws, bins=bins)
        human_counts, _ = np.histogram(human_draws, bins=bins)
        sim_outcome_counts, _ = np.histogram(outcome_sim, bins=outcome_bins)
        human_outcome_counts, _ = np.histogram(human_outcome, bins=outcome_bins)

        r2_draws = r2_score(human_counts, sim_counts)
        r2_outcome = r2_score(human_outcome_counts, sim_outcome_counts)

        records.append({
            "userID": target_id,
            "r2_draws": r2_draws,
            "r2_outcome": r2_outcome,
            "sim_counts": sim_counts,
            "human_counts": human_counts,
            "sim_outcome_counts": sim_outcome_counts,
            "human_outcome_counts": human_outcome_counts,
        })

    return records


num_runs = 300
raw_sim_dir = os.path.join(data_path, "raw_simulations")
horizon_str = FIT_HORIZON[0]
raw_sims_complete = (
    os.path.isdir(raw_sim_dir)
    and len(glob.glob(os.path.join(raw_sim_dir, f"sim_run_*_{horizon_str}.pkl"))) >= num_runs
)

if os.path.exists(metrics_save_path) and os.path.exists(ensemble_save_path) and raw_sims_complete:
    print("Metrics, ensemble data, and raw simulations already exist. Nothing to do.")
else:
    os.makedirs(raw_sim_dir, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Runs", total=num_runs)):
        total_records = Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(compute_subject_metrics)(i, userId_list, results_df, human_data, bins, FIT_HORIZON, raw_sim_dir)
            for i in range(num_runs)
        )

    metrics = []
    ensemble_data = {}

    for target_id in userId_list:
        user_records = [
            record[i]
            for record in total_records
            for i in range(len(record))
            if record[i]["userID"] == target_id
        ]
        if not user_records:
            continue

        r2_draws_values = [r["r2_draws"] for r in user_records]
        r2_outcome_values = [r["r2_outcome"] for r in user_records]

        metrics.append({
            "userID": target_id,
            "r2_draws": np.mean(r2_draws_values),
            "r2_draws_var": np.var(r2_draws_values),
            "r2_outcome": np.mean(r2_outcome_values),
            "r2_outcome_var": np.var(r2_outcome_values),
        })

        all_sim_counts = np.vstack([r["sim_counts"] for r in user_records])
        avg_sim_counts = np.mean(all_sim_counts, axis=0)
        std_sim_counts = np.std(all_sim_counts, axis=0)
        human_counts = user_records[0]["human_counts"]

        all_sim_outcomes = np.vstack([r["sim_outcome_counts"] for r in user_records])
        avg_sim_outcomes = np.mean(all_sim_outcomes, axis=0)
        std_sim_outcomes = np.std(all_sim_outcomes, axis=0)
        human_outcomes = user_records[0]["human_outcome_counts"]

        ensemble_data[target_id] = {
            "human_counts": human_counts,
            "avg_sim_counts": avg_sim_counts,
            "std_sim_counts": std_sim_counts,
            "human_outcomes": human_outcomes,
            "avg_sim_outcomes": avg_sim_outcomes,
            "std_sim_outcomes": std_sim_outcomes,
        }

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Metrics summary saved to: {metrics_save_path}")

    with open(ensemble_save_path, "wb") as f:
        pickle.dump(ensemble_data, f)
    print(f"Ensemble distribution data saved to: {ensemble_save_path}")
    print(f"Raw trial-by-trial data saved to: {raw_sim_dir}")

print("ALL OK")
