import sys
import os
import glob
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  # if using older versions
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# # save the config file simulation_params to the same data directory.
import os
import shutil

# Add the src directory to the Python path
project_root = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(project_root)
from src.params_fitting.data_simulation import simulate_data
from src.config import *
from itertools import islice

from src.utils import extract_draws_for_subject, extract_hist_data_for_user

algorithm = ALGORITHM
max_cards_per_draw = MAX_CARDS_PER_DRAW
param_order = PARAM_ORDER
param_ranges = PARAM_RANGES
n_subjects = N_SUBJECTS
n_jobs = N_JOBS
figure_path = FIGURE_PATH
data_path = DATA_PATH
file_path = HUMAN_DATA_PATH
all_evidence_path = ALL_EVIDENCE_PATH
full_sim_df_path = FULL_SIM_DF_PATH
results_path = RESULTS_PATH
full_sim_df_recovered_path = FULL_SIM_DF_RECOVERED_PATH
results_recovered_path = RESULTS_RECOVERED_PATH
from src.pomdp import POMDPFactory


def parse_simulation_results(results, data_fullsequence=None):
    # Remove any failed jobs (None)
    results = [r for r in results if r is not None]

    fit_params_ga = []
    after_lls_ga = []
    subject_IDs = []
    data_list = []
    hessian_matrices = []

    for res in results:
        ga, ag, data, subject_ID, hessian = res
        fit_params_ga.append(ga)
        after_lls_ga.append(ag)
        data_list.append(data)
        subject_IDs.append(subject_ID)
        hessian_matrices.append(hessian)

    # Build a list for data_fullsequence that is aligned with subject_IDs
    if data_fullsequence is None:
        data_fullsequence_list = [None] * len(subject_IDs)
    elif isinstance(data_fullsequence, dict):
        # Map per-subject full-sequence dict to the order of subject_IDs
        data_fullsequence_list = [
            data_fullsequence.get(sid, None) for sid in subject_IDs
        ]
    elif isinstance(data_fullsequence, list):
        # Already a list: assume it is aligned
        data_fullsequence_list = data_fullsequence
    else:
        # Fallback: duplicate the provided object for each subject
        data_fullsequence_list = [data_fullsequence] * len(subject_IDs)

    fit_params_ga_list = np.array([[d[k] for k in param_order] for d in fit_params_ga])

    # save a dataframe with the results and the corresponding data
    results_df = pd.DataFrame(
        {
            "fit_params_ga": list(fit_params_ga_list),
            "after_lls_ga": after_lls_ga,
            "data_dict_of_lists": data_list,
            "data_dict_of_lists_fullsequence": data_fullsequence_list,
            "subject_ID": subject_IDs,
            "Hessian_matrix": hessian_matrices,
        }
    )
    return results_df


def fit_one_subject(df_ev_simulated, param_ranges, subject_ID, algorithm):
    pomdp = POMDPFactory(POMDP_TYPE)
    best_params_ga, log_likelihood, df_ev_simulated, subject_ID, hessian_matrix = (
        pomdp.fit_subject(df_ev_simulated, param_ranges, subject_ID, algorithm)
    )
    return (best_params_ga, log_likelihood, df_ev_simulated, subject_ID, hessian_matrix)


if __name__ == "__main__":
    if len(FIT_HORIZON) == 1:

        all_evidence_path = os.path.join(
            project_root,
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_{FIT_HORIZON[0]}.pkl",
        )
        all_evidence_path_fullsequence_path = os.path.join(
            project_root,
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_full_sequence_{FIT_HORIZON[0]}.pkl",
        )
    if len(FIT_HORIZON) == 2:

        all_evidence_path = os.path.join(
            project_root, f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts.pkl"
        )
        all_evidence_path_fullsequence_path = os.path.join(
            project_root,
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_full_sequence.pkl",
        )

    all_subject_evidence_dicts_of_dict = pd.read_pickle(all_evidence_path)
    # as dictionary as before from df
    all_subject_evidence_dicts_of_dict = all_subject_evidence_dicts_of_dict.to_dict(
        orient="index"
    )
    filtered_all_subject_evidence_dicts_of_dict = dict(
        islice(all_subject_evidence_dicts_of_dict.items(), n_subjects)
    )

    all_subject_evidence_dicts_of_dict_fullsequence = pd.read_pickle(
        all_evidence_path_fullsequence_path
    )
    # as dictionary as before from df
    all_subject_evidence_dicts_of_dict_fullsequence = (
        all_subject_evidence_dicts_of_dict_fullsequence.to_dict(orient="index")
    )
    filtered_all_subject_evidence_dicts_of_dict_fullsequence = dict(
        islice(all_subject_evidence_dicts_of_dict_fullsequence.items(), n_subjects)
    )

    def ensure_all_paths_exist():
        """Ensure all data directories exist before saving files."""
        paths_to_create = [
            data_path,  # Main data path
            os.path.dirname(results_path),
            os.path.dirname(full_sim_df_path),
            os.path.dirname(FULL_SIM_DF_PATH_compressed),
            os.path.dirname(results_recovered_path),
            os.path.dirname(full_sim_df_recovered_path),
            os.path.dirname(FULL_SIM_DF_RECOVERED_PATH_compressed),
        ]

        for path in paths_to_create:
            if path and not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")

    ensure_all_paths_exist()

    # I will give only the short horizon, then, I will generate the simulated data, then fit the long, then simulate. The same for the following, but I need to modify the simulate function to look for either short or long
    # # Number of iterations you plan to run
    with tqdm_joblib(tqdm(desc="Processing subjects", total=n_subjects)):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fit_one_subject)(df_ev_simulated, param_ranges, user_id, algorithm)
            for user_id, df_ev_simulated in filtered_all_subject_evidence_dicts_of_dict.items()
        )

    os.makedirs(data_path, exist_ok=True)

    results_df = parse_simulation_results(
        results, filtered_all_subject_evidence_dicts_of_dict_fullsequence
    )
    results_df.to_pickle(results_path)
    results_df = pd.read_pickle(results_path)

    # here I need to concatenate both data.
    simulated_subject_dfs, evidence_to_fit_dict = simulate_data(
        results_df, sim_same_data=True
    )
    simulated_subject_dfs.to_pickle(full_sim_df_path)
    pd.DataFrame(evidence_to_fit_dict).T.to_pickle(FULL_SIM_DF_PATH_compressed)

    if POMDP_TYPE != "forgetting":

        all_subject_evidence_dicts_of_dict_simulated = evidence_to_fit_dict
        with tqdm_joblib(tqdm(desc="Processing subjects", total=n_subjects)):
            results_recovered = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(fit_one_subject)(
                    df_ev_simulated, param_ranges, user_id, algorithm
                )
                for user_id, df_ev_simulated in all_subject_evidence_dicts_of_dict_simulated.items()
            )

        # here I need to concatenate both data.

        results_df_recovered = parse_simulation_results(results_recovered)
        # Save the results DataFrame to a CSV file join the data path with recovery folder
        results_df_recovered.to_pickle(results_recovered_path)
        results_df_recovered = pd.read_pickle(results_recovered_path)

        simulated_subject_dfs_recovered, data_compressed_recovered_simulation = (
            simulate_data(results_df_recovered, sim_same_data=False)
        )
        # save the recovered simulated data to a pickle file
        simulated_subject_dfs_recovered.to_pickle(full_sim_df_recovered_path)
        pd.DataFrame(data_compressed_recovered_simulation).T.to_pickle(
            FULL_SIM_DF_RECOVERED_PATH_compressed
        )

    # the override file that actually drove this run (set per Slurm job/array
    # task via SIM_CONFIG_PATH; falls back to the bundled default)
    from src.config.loader import DEFAULT_OVERRIDE_PATH

    src = os.environ.get("SIM_CONFIG_PATH", DEFAULT_OVERRIDE_PATH)

    # destination directory (uses the notebook variable `data_path`)
    os.makedirs(data_path, exist_ok=True)
    dst = os.path.join(data_path, f"simulation_params_{TASK}.py")

    # safe copy (overwrites existing file)
    try:
        shutil.copy2(src, dst)
        print(f"Copied:\n  {src}\n-> {dst}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Source not found: {src}")
    except Exception as e:
        raise RuntimeError(f"Copy failed: {e}")

    print("Simulation and fitting completed successfully.")

results_df = pd.read_pickle(results_path)
# Setup paths and bins
bins = np.arange(1, 14, 1)
metrics_save_path = os.path.join(data_path, "ensemble_metrics_summary.csv")
ensemble_save_path = os.path.join(data_path, "ensemble_distribution_data.pkl")

human_data = pd.read_pickle(HUMAN_DATA_PATH)
userId_list = human_data["userID"].unique().tolist()


# ==========================================
# 1. Metric Computation Function
# ==========================================
def compute_subject_metrics(
    run_idx, userId_list, results_df, human_data, bins, FIT_HORIZON, raw_sim_dir
):
    """Compute metrics and extract binned data for all subjects in one simulation run."""
    # Assuming simulate_data, extract_draws_for_subject, and extract_hist_data_for_user
    # are defined elsewhere in your script.
    all_simulated_data, _ = simulate_data(results_df, sim_same_data=True)

    # --- ADDED: Save the raw trial-by-trial data for the GLM script ---
    # We save this as a pickle file which cleanly handles lists inside dataframes
    horizon_str = FIT_HORIZON[0]
    sim_file_path = os.path.join(raw_sim_dir, f"sim_run_{run_idx}_{horizon_str}.pkl")
    all_simulated_data.to_pickle(sim_file_path)
    # ------------------------------------------------------------------

    records = []

    # Specific bins to cleanly capture the discrete outcomes: -2, -1, 0, 1, 2
    # The '1' bin will naturally be empty if your data only has -2, -1, 0, 2
    outcome_bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    for target_id in userId_list:
        sim_draws = extract_draws_for_subject(all_simulated_data, target_id)
        outcome_sim = all_simulated_data[all_simulated_data["userID"] == target_id][
            "outcome"
        ].values

        # Extract human data based on FIT_HORIZON
        short_draws, long_draws, short_rewards, long_rewards = (
            extract_hist_data_for_user(human_data, target_id)
        )
        if FIT_HORIZON == ["short"]:
            human_draws = short_draws
            human_outcome = short_rewards
        elif FIT_HORIZON == ["long"]:
            human_draws = long_draws
            human_outcome = long_rewards
        else:  # both
            human_draws = short_draws + long_draws
            human_outcome = short_rewards + long_rewards

        # Compute histograms for draws
        sim_counts, _ = np.histogram(sim_draws, bins=bins)
        human_counts, _ = np.histogram(human_draws, bins=bins)

        # Compute histograms for outcomes using discrete bins
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


# Check first if the simulations ran and the files exist, to avoid rerunning them.
# Checking metrics_save_path/ensemble_save_path alone isn't enough: those predate
# raw_simulations/ being saved at all, so a stale pair from an older run would skip
# regenerating raw_simulations/ even when it's missing or incomplete -- which is
# exactly the data the GLM ensemble scripts need to avoid resimulating from scratch.
num_runs = 300
raw_sim_dir = os.path.join(data_path, "raw_simulations")
horizon_str = FIT_HORIZON[0]
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

else:
    os.makedirs(raw_sim_dir, exist_ok=True)

    # ==========================================
    # 2. Parallel Execution
    # ==========================================

    with tqdm_joblib(tqdm(desc="Runs", total=num_runs)) as progress_bar:
        total_records = Parallel(n_jobs=-1, verbose=1)(
            # --- EDITED: Pass raw_sim_dir to the function ---
            delayed(compute_subject_metrics)(
                i, userId_list, results_df, human_data, bins, FIT_HORIZON, raw_sim_dir
            )
            for i in range(num_runs)
        )

    # ==========================================
    # 3. Data Aggregation (Metrics + Ensemble)
    # ==========================================
    metrics = []
    ensemble_data = {}

    for target_id in userId_list:
        # Fetch all records for this user
        user_records = [
            record[i]
            for record in total_records
            for i in range(len(record))
            if record[i]["userID"] == target_id
        ]

        # --- Standard R2 Metrics ---
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

        # --- Ensemble Data for Plots ---
        # 1. Draws (Averaged Bins)
        all_sim_counts = np.vstack([r["sim_counts"] for r in user_records])
        avg_sim_counts = np.mean(all_sim_counts, axis=0)
        std_sim_counts = np.std(all_sim_counts, axis=0)
        human_counts = user_records[0]["human_counts"]

        # 2. Outcomes (Averaged Bins)
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
    # Assuming data_path is a directory like "outputs/data/"
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Metrics summary saved to: {metrics_save_path}")

    with open(ensemble_save_path, "wb") as f:
        pickle.dump(ensemble_data, f)
    print(f"Ensemble distribution data saved to: {ensemble_save_path}")
    print(f"Raw trial-by-trial data saved to: {raw_sim_dir}")
