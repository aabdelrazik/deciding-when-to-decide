from tqdm import tqdm
import pandas as pd
import numpy as np
from src.pomdp import POMDPFactory
from src.pomdp import simulate_cards_pomdp
from src.config import *


def generate_meg_data(
    params: dict,
    data: pd.DataFrame | None = None,
    subject_id=None,
    num_trials: int | None = None,
) -> tuple:
    """Build a POMDP from `params`, run value iteration, then simulate one trial
    per row of `data` (replaying that subject's card sequence) or, if `data` is
    None, `num_trials` freshly-generated trials.

    Args:
        params (dict): Keyword args for POMDPFactory(POMDP_TYPE)'s constructor;
            must include "max_cards_per_draw" and "horizon_condition".
        data (pd.DataFrame, optional): Real per-trial data with a
            "draw_yellow_blue_action_outcome" column (as produced by this same
            function). If given, trials are replayed from these sequences.
            Defaults to None.
        subject_id: Identifier stored in the "userID" column of the returned
            DataFrame. Defaults to None.
        num_trials (int, optional): Number of trials to freshly simulate when
            `data` is None. Required in that case. Defaults to None.

    Returns:
        tuple: (data_simulated, df_ev_simulated, best_actions), where
            data_simulated is a per-trial summary DataFrame, df_ev_simulated
            has one "draw_yellow_blue_action_outcome" column in the same
            format as `data`, and best_actions is the fitted POMDP's
            self.best_actions array.
    """
    # --- POMDP Setup ---
    pomdp = POMDPFactory(POMDP_TYPE)
    pomdp.__init__(**params)
    pomdp.value_iteration()
    best_actions = pomdp.best_actions
    max_cards_per_draw = params["max_cards_per_draw"]
    horizon_condition = params["horizon_condition"]

    (
        evidence_list,
        reward_list,
        num_draws_list,
        termination_list,
        action_lists,
        draw_sequence_list,
        num_yellow,
        num_blue,
        dist2ch_list,
        belief_trajectory,
    ) = ([], [], [], [], [], [], [], [], [], [])

    # --- Main Simulation Loop with Enhanced Retry Logic ---
    num_trials = (
        num_trials
        if data is None and num_trials != None
        else len(data["draw_yellow_blue_action_outcome"])
    )
    for trial in tqdm(range(num_trials), desc="Simulating cards", disable=True):
        if data is not None:
            sequence = data.loc[trial, "draw_yellow_blue_action_outcome"]
            given_sequence = True
        else:
            sequence = None
            given_sequence = False

        results = pomdp.simulate_cards_pomdp(
            given_sequence=given_sequence,
            card_sequence=sequence,
        )

        # --- Process Successful Results ---
        termination = 1 if horizon_condition == "long" else 2
        termination_list.append(termination)
        evidence_list.append(results["num_yellows"])
        # confidence_list.append(results["confidence"])
        reward_list.append(results["reward"])
        num_draws_list.append(results["num_draws"])
        draw_sequence = list(range(1, results["num_draws"] + 1))
        draw_sequence_list.append(draw_sequence)
        action_lists.append(results["decision"])
        num_blue.append(results["num_blues"])
        num_yellow.append(results["num_yellows"])
        # calculate the dist2ch
        # initalize the dist2ch to nan of the same size as the actions
        dist2ch = np.full_like(results["actions"], np.nan, dtype=np.float64)
        if results["decision"] != 2:
            # Since I already retrun the evidence until the decision draw. I don't need to extract when the subject acted I can
            # just reverse the order.
            just_before_actions = len(results["actions"]) - 1
            dist2ch[:just_before_actions] = np.arange(just_before_actions, 0, -1)
        dist2ch_list.append(dist2ch)
        belief_trajectory.append(results["trajectory"])

    # --- DataFrame Creation (only runs if all trials were successful) ---
    data_simulated = pd.DataFrame(
        {
            "ev": evidence_list,
            "outcome": reward_list,
            "termination": termination_list,
            "num_draws": num_draws_list,
            "userID": subject_id,
            "action": action_lists,
            "trial": draw_sequence_list,
            "totEvLeft": num_blue,
            "totEvRight": num_yellow,
            "distance2choice": dist2ch_list,
            "belief_trajectory": belief_trajectory,
        }
    )

    ev = data_simulated["ev"]
    action_list = data_simulated["action"]
    outcome_list = data_simulated["outcome"]
    result = []
    for i, row in enumerate(ev):
        tuples = []
        for k, yellow in enumerate(row):
            blue = (k + 1) * max_cards_per_draw - yellow
            action = 2  # default: waiting
            outcome = 0
            tuples.append([k + 1, int(yellow), int(blue), action, outcome])
        # set the last action to the actual action commited by the agent
        tuples[-1][3] = int(action_list[i])
        tuples[-1][4] = int(outcome_list[i])

        result.append(tuples)
    df_ev_simulated = pd.DataFrame({"draw_yellow_blue_action_outcome": result})
    return data_simulated, df_ev_simulated, best_actions


def get_horizon_keys(results_df: pd.DataFrame) -> list:
    """Collect the distinct horizon keys (e.g. "long"/"short") present across
    all rows of the "data_dict_of_lists" column.

    Args:
        results_df (pd.DataFrame): Must have a "data_dict_of_lists" column of
            dicts keyed by horizon condition.

    Returns:
        list: Sorted list of the distinct keys found.
    """
    keys = set()
    for d in results_df["data_dict_of_lists"].tolist():
        if isinstance(d, dict):
            keys.update(d.keys())
    return sorted(keys)


def simulate_data(
    results_df: pd.DataFrame, sim_same_data: bool = True, param_order=PARAM_ORDER
) -> tuple:
    """For every subject (row) in results_df, simulate trials for each horizon
    condition using that subject's fitted params (row["fit_params_ga"]), via
    generate_meg_data.

    Args:
        results_df (pd.DataFrame): One row per subject; must have columns
            "fit_params_ga", "subject_ID", "data_dict_of_lists", and (when
            sim_same_data is True) "data_dict_of_lists_fullsequence".
        sim_same_data (bool, optional): If True, replay each subject's actual
            card sequences (generate_meg_data's `data` arg); if False,
            generate the same number of fresh random trials instead. Defaults
            to True.
        param_order (list, optional): Order in which row["fit_params_ga"]'s
            values map onto parameter names. Defaults to PARAM_ORDER.

    Returns:
        tuple: (simulated_subject_dfs, evidence_to_fit_dict), where
            simulated_subject_dfs is the per-trial data_simulated DataFrames
            concatenated across all subjects/horizons, and
            evidence_to_fit_dict maps subject_ID -> {horizon: df_ev_simulated}.
    """
    simulated_subject_dfs = []
    evidence_to_fit_dict = {}
    for idx, row in results_df.iterrows():
        params = row[
            "fit_params_ga"
        ]  # or fit_params_pygad, fit_params_anneal, fit_params_ga
        params = {
            **dict(zip(param_order, params)),
            "verbose": VERBOSE,
            "max_cards_per_draw": MAX_CARDS_PER_DRAW,
        }

        sid = row["subject_ID"]  # Get the subject ID from the data
        if POMDP_TYPE == "forgetting" or POMDP_TYPE == "exaggerate_data":
            horizon_list = row["data_dict_of_lists"].keys().tolist()
        else:
            horizon_list = get_horizon_keys(results_df)
        num_trials = 0
        evidence_to_fit = {}
        data_simulated_list = []
        for horizon in horizon_list:
            num_trials = len(row["data_dict_of_lists"][horizon])

            params.update({"horizon_condition": horizon})
            if sim_same_data:

                data = row["data_dict_of_lists_fullsequence"][horizon]
                data_simulated, df_ev_simulated, _ = generate_meg_data(
                    params,
                    data=data,
                    subject_id=sid,
                )
                evidence_to_fit[horizon] = df_ev_simulated
                data_simulated_list.append(data_simulated)

            else:
                data_simulated, df_ev_simulated, _ = generate_meg_data(
                    params,
                    num_trials=num_trials,
                    subject_id=sid,
                )

                evidence_to_fit[horizon] = df_ev_simulated
                data_simulated_list.append(data_simulated)

        sim_df = pd.concat(data_simulated_list, axis=0)
        simulated_subject_dfs.append(sim_df)
        evidence_to_fit_dict[sid] = evidence_to_fit
    # Combine all into one DataFrame
    simulated_subject_dfs = pd.concat(simulated_subject_dfs, ignore_index=True)
    return simulated_subject_dfs, evidence_to_fit_dict


def simulate_data_single_subject(row, sim_same_data: bool = True) -> tuple:
    """Like simulate_data, but for a single-subject `row` where each column
    holds a length-1 Series (values are accessed via `.iloc[0]`) rather than
    a scalar.

    Args:
        row: Single-subject row (e.g. `results_df.loc[[idx]]`) with columns
            "fit_params_ga", "subject_ID", "data_dict_of_lists", and (when
            sim_same_data is True) "data_dict_of_lists_fullsequence".
        sim_same_data (bool, optional): If True, replay the subject's actual
            card sequences; if False, generate fresh random trials of the
            same count instead. Defaults to True.

    Returns:
        tuple: (sim_df, evidence_to_fit_dict), where sim_df is the per-trial
            data_simulated DataFrame concatenated across horizons, and
            evidence_to_fit_dict maps subject_ID -> {horizon: df_ev_simulated}.
    """
    evidence_to_fit_dict = {}
    params = row["fit_params_ga"].iloc[
        0
    ]  # or fit_params_pygad, fit_params_anneal, fit_params_ga
    params = {
        **dict(zip(PARAM_ORDER, params)),
        "verbose": VERBOSE,
        "max_cards_per_draw": MAX_CARDS_PER_DRAW,
    }

    sid = row["subject_ID"]  # Get the subject ID from the data
    if POMDP_TYPE == "forgetting" or POMDP_TYPE == "exaggerate_data":

        horizon_list = row["data_dict_of_lists"].keys().tolist()
    else:
        horizon_list = list(row["data_dict_of_lists"].iloc[0].keys())
    num_trials = 0
    evidence_to_fit = {}
    data_simulated_list = []
    for horizon in horizon_list:
        num_trials = len(row["data_dict_of_lists"].iloc[0][horizon])

        params.update({"horizon_condition": horizon})
        if sim_same_data:

            data = row["data_dict_of_lists_fullsequence"].iloc[0][horizon]
            data_simulated, df_ev_simulated, _ = generate_meg_data(
                params,
                data=data,
                subject_id=sid,
            )
            evidence_to_fit[horizon] = df_ev_simulated
            data_simulated_list.append(data_simulated)

        else:
            data_simulated, df_ev_simulated, _ = generate_meg_data(
                params,
                num_trials=num_trials,
                subject_id=sid,
            )

            evidence_to_fit[horizon] = df_ev_simulated
            data_simulated_list.append(data_simulated)

    sim_df = pd.concat(data_simulated_list, axis=0)
    evidence_to_fit_dict[sid] = evidence_to_fit
    return sim_df, evidence_to_fit_dict
