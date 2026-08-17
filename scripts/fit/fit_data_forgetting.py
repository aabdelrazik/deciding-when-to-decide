import sys
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  # if using older versions

# # save the config file simulation_params to the same data directory.
import os
import shutil
import numpy as np

# Add the src directory to the Python path
project_root = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(project_root)
from src.params_fitting.data_simulation import simulate_data
from src.config import *
from itertools import islice

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


def apply_discounting(card_sequences, gamma=0.5):
    """Recompute gamma-discounted effective yellow/blue counts for a raw
    (undiscounted) card sequence, matching data_preprocessing_forgetting's
    recursive weighting: discounted[i] = gamma * discounted[i-1] + raw[i] +
    (1 - gamma).
    """
    undiscounted_yellow = [0]
    undiscounted_blue = [0]
    for card_sequence in card_sequences:
        draw, yellow, blue, action, outcome = card_sequence
        undiscounted_yellow.append(yellow)
        undiscounted_blue.append(blue)
    undiscounted_yellow = np.diff(undiscounted_yellow)
    undiscounted_blue = np.diff(undiscounted_blue)
    discounted_yellow = np.zeros((len(undiscounted_yellow), 1))
    discounted_blue = np.zeros((len(undiscounted_blue), 1))
    discounted_yellow[0] = undiscounted_yellow[0]
    discounted_blue[0] = undiscounted_blue[0]
    i = 1
    for yellow in undiscounted_yellow[1:]:
        discounted_yellow[i] = gamma * discounted_yellow[i - 1] + yellow + (1 - gamma)
        i += 1
    i = 1
    for blue in undiscounted_blue[1:]:
        discounted_blue[i] = gamma * discounted_blue[i - 1] + blue + (1 - gamma)
        i += 1
    new_card_sequences = []
    for i, card_sequence in enumerate(card_sequences):
        draw, yellow, blue, action, outcome = card_sequence
        new_card_sequences.append(
            [
                draw,
                np.round(discounted_yellow[i][0], 2),
                np.round(discounted_blue[i][0], 2),
                action,
                outcome,
            ]
        )
    return new_card_sequences


def parse_simulation_results(results, data_fullsequence=None):
    # Remove any failed jobs (None)
    results = [r for r in results if r is not None]

    fit_params_ga = []
    after_lls_ga = []
    subject_IDs = []
    data_list = []
    hessian_matrices = []
    data_fullsequence_list = []

    for res in results:
        ga_parameters, likelihood_after_fit, data, subject_ID, hessian = res
        fit_params_ga.append(ga_parameters)
        after_lls_ga.append(likelihood_after_fit)
        data_list.append(data)
        subject_IDs.append(subject_ID)
        hessian_matrices.append(hessian)
        if data_fullsequence is None:
            data_fullsequence_list.append(None)
        else:
            data_fullsequence_list.append(
                data_fullsequence[subject_ID][ga_parameters["gamma"]]
            )

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
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_{FIT_HORIZON[0]}_combined.pkl",
        )
        all_evidence_path_fullsequence_path = os.path.join(
            project_root,
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_full_sequence_{FIT_HORIZON[0]}_combined.pkl",
        )
    if len(FIT_HORIZON) == 2:

        all_evidence_path = os.path.join(
            project_root,
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_combined.pkl",
        )
        all_evidence_path_fullsequence_path = os.path.join(
            project_root,
            f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_full_sequence_combined.pkl",
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

        # Call this EARLY in your __main__ block, right after defining paths

    ensure_all_paths_exist()

    # RECOVERY_ONLY=1: skip the original GA fit entirely and load the
    # already-published results.pkl instead, so re-running this script to
    # backfill a missing recovery step can never perturb the original
    # fit_params_ga (GA re-optimization is stochastic, so simply re-running
    # the whole script would risk producing slightly different numbers).
    RECOVERY_ONLY = os.environ.get("RECOVERY_ONLY", "0") == "1"

    if RECOVERY_ONLY:
        print(f"RECOVERY_ONLY=1: loading existing fit from {results_path}")
        results_df = pd.read_pickle(results_path)

        if POMDP_TYPE == "forgetting":
            # results.pkl's data_dict_of_lists / data_dict_of_lists_fullsequence
            # store each subject's own gamma-discounted evidence, not raw card
            # counts. Replaying discounted values through the raw-count replay
            # path would truncate them (given_sequence casts to int) and then
            # double-discount downstream, so substitute the genuinely raw
            # sequences before simulating the recovery target.
            print("Substituting raw (undiscounted) evidence for recovery replay...")
            raw = {}
            raw_full = {}
            for hz in FIT_HORIZON:
                raw[hz] = pd.read_pickle(
                    os.path.join(
                        project_root,
                        f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_{hz}.pkl",
                    )
                ).to_dict(orient="index")
                raw_full[hz] = pd.read_pickle(
                    os.path.join(
                        project_root,
                        f"data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_full_sequence_{hz}.pkl",
                    )
                ).to_dict(orient="index")

            def _raw_row(row, source):
                sid = row["subject_ID"]
                return pd.Series({hz: source[hz][sid][hz] for hz in FIT_HORIZON})

            results_df = results_df.copy()
            results_df["data_dict_of_lists"] = pd.Series(
                [_raw_row(r, raw) for _, r in results_df.iterrows()],
                index=results_df.index,
            )
            results_df["data_dict_of_lists_fullsequence"] = pd.Series(
                [_raw_row(r, raw_full) for _, r in results_df.iterrows()],
                index=results_df.index,
            )
    else:
        # I will give only the short horizon, then, I will generate the simulated data, then fit the long, then simulate. The same for the following, but I need to modify the simulate function to look for either short or long
        # Number of iterations you plan to run
        with tqdm_joblib(tqdm(desc="Processing subjects", total=n_subjects)):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(fit_one_subject)(df_ev_simulated, param_ranges, user_id, algorithm)
                for user_id, df_ev_simulated in filtered_all_subject_evidence_dicts_of_dict.items()
            )

        results_df = parse_simulation_results(
            results, filtered_all_subject_evidence_dicts_of_dict_fullsequence
        )
        results_df.to_pickle(results_path)
        results_df = pd.read_pickle(results_path)

    # here I need to concatenate both data.
    simulated_subject_dfs, evidence_to_fit_dict = simulate_data(
        results_df, sim_same_data=True
    )
    if not RECOVERY_ONLY:
        simulated_subject_dfs.to_pickle(full_sim_df_path)
        pd.DataFrame(evidence_to_fit_dict).T.to_pickle(FULL_SIM_DF_PATH_compressed)

    if RECOVERY_ONLY or POMDP_TYPE not in ("forgetting", "exaggerate"):

        fullsequence_for_parse = None
        if RECOVERY_ONLY and POMDP_TYPE == "forgetting":
            # fit_subject expects a gamma value -> {horizon: DataFrame} dict
            # (make_cost_function snaps the candidate gamma to the nearest grid
            # point and indexes into this dict), so rebuild that structure from
            # the raw-replayed simulated sequences by discounting fresh at each
            # gamma_values grid point.
            print("Building gamma-keyed evidence structure for recovery fit...")
            import copy as _copy

            gamma_keyed_evidence_to_fit_dict = {}
            for sid, horizon_dict in evidence_to_fit_dict.items():
                gamma_keyed_evidence_to_fit_dict[sid] = {}
                for gamma in gamma_values:
                    horizon_dict_copy = _copy.deepcopy(horizon_dict)
                    for hz, df in horizon_dict_copy.items():
                        if (
                            isinstance(df, pd.DataFrame)
                            and "draw_yellow_blue_action_outcome" in df.columns
                        ):
                            df["draw_yellow_blue_action_outcome"] = df[
                                "draw_yellow_blue_action_outcome"
                            ].apply(lambda x: apply_discounting(x, gamma=gamma))
                    # stored as a Series, not a plain dict: fit_subject's return
                    # (df_ev_simulated[gamma]) becomes results_df_recovered's
                    # "data_dict_of_lists", and the downstream simulate_data call
                    # indexes it via .keys().tolist(), which a plain dict lacks.
                    gamma_keyed_evidence_to_fit_dict[sid][gamma] = pd.Series(
                        horizon_dict_copy
                    )

            all_subject_evidence_dicts_of_dict_simulated = (
                gamma_keyed_evidence_to_fit_dict
            )
            fullsequence_for_parse = gamma_keyed_evidence_to_fit_dict
        else:
            all_subject_evidence_dicts_of_dict_simulated = evidence_to_fit_dict

        with tqdm_joblib(tqdm(desc="Processing subjects", total=n_subjects)):
            results_recovered = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(fit_one_subject)(
                    df_ev_simulated, param_ranges, user_id, algorithm
                )
                for user_id, df_ev_simulated in all_subject_evidence_dicts_of_dict_simulated.items()
            )
        # here I need to concatenate both data.

        results_df_recovered = parse_simulation_results(
            results_recovered, fullsequence_for_parse
        )
        # Save the results DataFrame to a CSV file join the data path with recovery folder
        results_df_recovered.to_pickle(results_recovered_path)
        results_df_recovered = pd.read_pickle(results_recovered_path)

        simulated_subject_dfs_recovered, data_compressed_recovered_simulation = (
            simulate_data(results_df_recovered, sim_same_data=True)
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
    dst = os.path.join(data_path, "simulation_params.py")

    # safe copy (overwrites existing file)
    try:
        shutil.copy2(src, dst)
        print(f"Copied:\n  {src}\n-> {dst}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Source not found: {src}")
    except Exception as e:
        raise RuntimeError(f"Copy failed: {e}")

    print("Simulation and fitting completed successfully.")
