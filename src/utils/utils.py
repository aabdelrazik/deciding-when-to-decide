from scipy.stats import truncnorm, beta, skellam
from math import ceil
from scipy.special import gamma
import numpy as np
from scipy.stats import beta, binom
import scipy.integrate as integrate
import sys
import os
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler
import glob
import importlib.util
from scipy.stats import spearmanr

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
from src.config import *

param_order = PARAM_ORDER


import importlib.util


def load_config_module(file_path: str):
    """Loads a python file as a module dynamically using importlib.

    Args:
        file_path: Path to the .py file to load.

    Returns:
        The loaded module, registered internally under the name "params".
    """
    spec = importlib.util.spec_from_file_location("params", file_path)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    return params


# for the glm fitting.
def nannormalise(arr: np.ndarray) -> np.ndarray:
    """Z-score columns ignoring rows with NaNs (applies per-column across rows).

    Rows containing any NaN are excluded from fitting the scaler and left
    as NaN in the output; all other rows are column-wise z-scored (mean 0,
    unit variance) using `StandardScaler`.

    Args:
        arr: 2-D array, shape (n_rows, n_cols).

    Returns:
        Array of the same shape, z-scored per column, with NaN-containing
        rows left as NaN.
    """
    scaler = StandardScaler()
    mask = ~np.isnan(arr).any(axis=1)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if mask.sum() > 0:
        out[mask] = scaler.fit_transform(arr[mask])
    return out


def lagmatrix(arr: np.ndarray, lag: int = 1) -> np.ndarray:
    """Shift a 1-D array forward by `lag` positions, filling with NaN.

    Args:
        arr: 1-D array to lag.
        lag: Number of positions to shift forward. If `lag <= 0`, the
            array is returned unshifted (as a float copy).

    Returns:
        Array of the same shape as `arr`, where element `i` holds
        `arr[i - lag]` (NaN for `i < lag`).
    """
    out = np.empty_like(arr, dtype=np.float64)
    out[:] = np.nan
    if lag > 0:
        out[lag:] = arr[:-lag]
    else:
        out[:] = arr
    return out


def safe_spearman(s1: pd.Series, s2: pd.Series) -> tuple[float, float]:
    """Spearman correlation between two Series, robust to empty/NaN-only overlap.

    Args:
        s1: First series.
        s2: Second series.

    Returns:
        (correlation, p-value) from `scipy.stats.spearmanr`, computed only
        on index-aligned, non-NaN pairs. Returns (nan, nan) if either
        series is empty or fewer than 2 valid paired observations remain.
    """
    if s1.empty or s2.empty:
        return (np.nan, np.nan)
    s1a, s2a = s1.align(s2, join="inner")
    mask = (~s1a.isna()) & (~s2a.isna())
    if mask.sum() < 2:
        return (np.nan, np.nan)
    return spearmanr(s1a[mask], s2a[mask])


def calculate_hazard_cum(start_hazard: int, end_hazard: int) -> np.ndarray:
    """Compute the discrete hazard function for a uniform draw-termination distribution.

    Builds a uniform probability mass over draws `[start_hazard, end_hazard]`
    and converts it to a hazard rate (probability of terminating on draw i
    given no termination before draw i).

    Args:
        start_hazard: First draw index with nonzero termination probability.
        end_hazard: Last draw index with nonzero termination probability.

    Returns:
        1-D array of length `end_hazard + 1`, the hazard rate at each draw
        index (rounded to 3 decimals); indices before `start_hazard` are 0.
    """
    p = 1 / (end_hazard - start_hazard + 1)
    start_hazard = int(start_hazard)
    pmf = np.zeros((end_hazard + 1))
    pmf_uniform = [p] * (end_hazard - (start_hazard) + 1)
    pmf[start_hazard : end_hazard + 1] = pmf_uniform
    shifted_pmf = np.zeros_like(pmf)
    shifted_pmf[1:] = pmf[:-1]
    hazard = pmf / (1 - np.cumsum(shifted_pmf))
    return np.round(hazard, 3)


def parse_simulation_results(
    results: list[tuple | None], data_fullsequence: dict | None = None
) -> pd.DataFrame:
    """Collect parallel-fit job results into a single results DataFrame.

    Args:
        results: One entry per job, each either None (failed job, dropped)
            or a tuple (ga_parameters, likelihood_after_fit, data,
            subject_ID, hessian).
        data_fullsequence: Nested mapping `data_fullsequence[subject_ID][gamma]`
            used to look up each subject's full-sequence data for the
            fitted gamma. Must contain an entry for every subject_ID and
            fitted gamma present in `results`.

    Returns:
        DataFrame with one row per successful job and columns
        "fit_params_ga", "after_lls_ga", "data_dict_of_lists",
        "data_dict_of_lists_fullsequence", "subject_ID", "Hessian_matrix".
    """
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


def get_subject_data(human_meg_data: pd.DataFrame, user_id: str) -> pd.DataFrame | None:
    """
    Accesses the nested trial data for a specific subject from the main DataFrame.

    Args:
        human_meg_data (pd.DataFrame): The main DataFrame containing a 'userID' column
                                       and a 'data' column with nested DataFrames.
        user_id (str): The unique identifier for the subject.

    Returns:
        pd.DataFrame or None: The subject's trial data as a DataFrame, or None if not found.
    """
    # Use .loc for explicit label-based indexing, which is slightly safer
    subject_row = human_meg_data.loc[human_meg_data["userID"] == user_id]
    if not subject_row.empty:
        # .values[0] extracts the nested DataFrame from the 'data' column of the first matching row
        return subject_row["data"].values[0]
    else:
        print(f"Warning: No data found for userID: {user_id}")
        return None


def calc_num_draws(results: dict) -> tuple[list[int], list[int], list[int]]:
    """Split per-game draw counts by horizon condition.

    Args:
        results: Mapping of game key -> dict with "num_draws" and
            "horizon_condition" ("short" or "long") keys, e.g. the
            "results" entry returned by `analyze_meg_draws`.

    Returns:
        Tuple (num_draws_both, num_draws_long, num_draws_short): draw
        counts for all games, long-horizon games only, and short-horizon
        games only, respectively.
    """
    num_draws_long = []
    num_draws_short = []
    num_draws_both = []
    for k, v in results.items():
        if v["horizon_condition"] == "short":
            num_draws_short.append(v["num_draws"])
        elif v["horizon_condition"] == "long":
            num_draws_long.append(v["num_draws"])
        num_draws_both.append(v["num_draws"])
    return num_draws_both, num_draws_long, num_draws_short


def analyze_meg_draws(
    human_meg_data: pd.DataFrame, userID_list: list | None = None, verbose: bool = False
) -> dict:
    """Compute per-game draw counts and decision-point pair statistics.

    For each subject/block/game, counts draws taken until the decision
    (or the full game if missed) and tallies how often each
    (total_yellow, total_blue) card count at the decision point led to a
    decision, split by horizon condition (short: <=8 total draws, long:
    >8).

    Args:
        human_meg_data: DataFrame with "userID" and "data" columns (the
            latter holding each subject's nested trial DataFrame), as
            consumed by `get_subject_data`.
        userID_list: Subject IDs to include; defaults to all unique
            userIDs in `human_meg_data`.
        verbose: If True, print the (total_yellow, total_blue) decision
            counts per horizon condition.

    Returns:
        dict with keys:
            results: dict keyed by (userID, block, game) -> per-game
                stats (num_draws, yellow_sequence, blue_sequence, missed,
                total_yellow, total_blue, full_sequence_length,
                horizon_condition).
            pair_count_short / pair_count_long: Counter of
                (total_yellow, total_blue) at the decision point, by
                horizon condition.
            counts_dict_short / counts_dict_long: {(total_yellow,
                total_blue): "decisions/occurrences"} strings.
            num_games: Total number of games processed.
    """
    # Initialize counters
    pair_count_short = Counter()
    pair_count_long = Counter()
    results = dict()
    num_games = 0
    all_occurances_pairs_short = []
    all_occurances_pairs_long = []
    if userID_list is None:
        userID_list = human_meg_data[
            "userID"
        ].unique()  # this happens when I want to loop over the whole data but not specified groups (e.g., ocd or healthy)

    for user_idx in userID_list:
        user = get_subject_data(human_meg_data, user_idx)
        if user is not None:
            for block in user["block"].unique():
                for game in user["game"].unique():
                    num_games += (
                        1  # only counts the total number of games to compare at the end
                    )
                    key = (user_idx, block, game)
                    # access the data directly from the block and game instead of the mask,
                    data_mask = (user["block"] == block) & (user["game"] == game)
                    data_block = user[
                        data_mask
                    ]  # only to show the current block and game data

                    yellow_sequence = data_block["currEvYellow"].tolist()
                    blue_sequence = (
                        5 - data_block["currEvYellow"]
                    ).tolist()  # since the maximum number of cards is 5, blue is the complement of yellow

                    chosen_sequence = data_block[
                        "choiceTrial"
                    ]  # chosen is when the subject chose an action, and the first non Nan value is the trial at which the subject decided

                    full_sequence_length = len(
                        yellow_sequence
                    )  # used later to denote the predetermined number of draws, and whether it's short or long

                    # Find the trial index where the subject first chose an action
                    # if the subject missed, this game is not counted for future analysis
                    if chosen_sequence.isnull().all():
                        total_yellow = sum(yellow_sequence)
                        total_blue = sum(blue_sequence)
                        results[(user_idx, block, game)] = {
                            "num_draws": len(yellow_sequence),
                            "yellow_sequence": yellow_sequence,
                            "blue_sequence": blue_sequence,
                            "missed": True,
                            "total_yellow": total_yellow,
                            "total_blue": total_blue,
                            "full_sequence_length": full_sequence_length,
                        }
                        tuples_yellow_blue = list(
                            zip(np.cumsum(yellow_sequence), np.cumsum(blue_sequence))
                        )
                    else:
                        # get the index of the first non-null value in the chosen sequence
                        first_choice_idx = chosen_sequence.first_valid_index()
                        pos = list(chosen_sequence.index).index(first_choice_idx) + 1
                        yellow_seq_till_choice = yellow_sequence[:pos]
                        blue_seq_till_choice = blue_sequence[:pos]
                        num_draws = pos + 1  # number of draws before choice
                        tuples_yellow_blue = list(
                            zip(
                                np.cumsum(yellow_seq_till_choice),
                                np.cumsum(blue_seq_till_choice),
                            )
                        )  # create a list of tuples for yellow and blue sequences

                        total_yellow = sum(yellow_seq_till_choice)
                        total_blue = sum(blue_seq_till_choice)

                        # Store in results dictionary (optional, can remove if not needed)
                        results[key] = {
                            "num_draws": num_draws,
                            "yellow_sequence": yellow_seq_till_choice,
                            "blue_sequence": blue_seq_till_choice,
                            "total_yellow": total_yellow,
                            "total_blue": total_blue,
                            "missed": False,
                            "full_sequence_length": full_sequence_length,
                        }

                    # Increment the appropriate counter
                    # If I wanted to calculate the fraction of chosen number of blue and yellow, I can do this here.
                    if full_sequence_length <= 8:
                        all_occurances_pairs_short.extend(
                            tuples_yellow_blue
                        )  # collect all pairs of yellow and blue sequences
                        results[key]["horizon_condition"] = "short"
                        if results[key]["missed"] is False:
                            pair_count_short[
                                (total_yellow, total_blue)
                            ] += 1  # for the last pair count (how many blue and how many yellow at the decision point appeared), count how many times it appears.
                    else:
                        all_occurances_pairs_long.extend(
                            tuples_yellow_blue
                        )  # collect all pairs of yellow and blue sequences
                        results[key]["horizon_condition"] = "long"
                        if results[key]["missed"] is False:
                            pair_count_long[(total_yellow, total_blue)] += 1

    all_occurances_pairs_short_counts = Counter(
        all_occurances_pairs_short
    )  # Only for trials where subjects decided, this has all the counts of cards at each draw, so across all subjects how many times these pair of cards appeared. But this needs to also include the trials where subjects missed
    all_occurances_pairs_long_counts = Counter(
        all_occurances_pairs_long
    )  # The same but for the long horizon condition.

    # here this implementation is not correct, I want to loop over the all occurances pairs counts, and make it in the denominator, and then, search for the corresponding pair_count_short index, if it exists I use it, if it doesn't, I write 0 in the numerator
    counts_dict_short = {}
    for tpl, denom_count in all_occurances_pairs_short_counts.items():
        numerator = pair_count_short.get(tpl, 0)
        counts_dict_short[tpl] = f"{numerator}/{denom_count}"

    counts_dict_long = {}
    for tpl, denom_count in all_occurances_pairs_long_counts.items():
        numerator = pair_count_long.get(tpl, 0)
        counts_dict_long[tpl] = f"{numerator}/{denom_count}"
    if verbose:
        # Print summary
        print("Short sequences (length <= 8):")
        for pair, count in pair_count_short.items():
            print(f"{pair}: {count}")
        print("\nLong sequences (length > 8):")
        for pair, count in pair_count_long.items():
            print(f"{pair}: {count}")

        # You may choose to return results for unit testing
    return {
        "results": results,
        "pair_count_short": pair_count_short,
        "pair_count_long": pair_count_long,
        "counts_dict_short": counts_dict_short,
        "counts_dict_long": counts_dict_long,
        "num_games": num_games,
    }


def analyze_meg_draws_color(
    human_meg_data: pd.DataFrame, userID_list: list | None = None, verbose: bool = False
) -> dict:
    """
    Like analyze_meg_draws but tracks fraction-yellow and fraction-blue at each
    (cumulative_yellow, cumulative_blue) state separately.

    Color is read from the 'action_taken' column at the decision trial:
      action_taken == 0  →  chose yellow
      action_taken == 1  →  chose blue

    'choiceTrial' is still used only to locate WHEN the decision happened
    (first non-null row).

    Note: `verbose` is accepted for interface parity with `analyze_meg_draws`
    but is not currently used in this function's body.

    Returns
    -------
    dict with keys:
      counts_dict_yellow_short / long  : {(y,b): "yellow_choices/total_visits"}
      counts_dict_blue_short  / long   : {(y,b): "blue_choices/total_visits"}
      counts_dict_short / long         : same "decide/visits" as analyze_meg_draws
      results, num_games               : same as analyze_meg_draws
    """
    yellow_choice_short = Counter()
    blue_choice_short = Counter()
    yellow_choice_long = Counter()
    blue_choice_long = Counter()
    pair_count_short = Counter()
    pair_count_long = Counter()

    all_occurrences_short = []
    all_occurrences_long = []

    results = {}
    num_games = 0

    if userID_list is None:
        userID_list = human_meg_data["userID"].unique()

    for user_idx in userID_list:
        user = get_subject_data(human_meg_data, user_idx)
        if user is None:
            continue

        for block in user["block"].unique():
            for game in user["game"].unique():
                num_games += 1
                key = (user_idx, block, game)
                data_mask = (user["block"] == block) & (user["game"] == game)
                data_block = user[data_mask]

                yellow_sequence = data_block["currEvLeft"].tolist()
                blue_sequence = (data_block["currEvRight"]).tolist()
                chosen_sequence = data_block["choiceTrial"]
                full_seq_len = len(yellow_sequence)

                if chosen_sequence.isnull().all():
                    # missed trial — no color choice to record
                    tuples_yb = list(
                        zip(np.cumsum(yellow_sequence), np.cumsum(blue_sequence))
                    )
                    results[key] = {
                        "missed": True,
                        "full_sequence_length": full_seq_len,
                        "horizon_condition": "short" if full_seq_len <= 8 else "long",
                    }
                else:
                    first_idx = chosen_sequence.first_valid_index()
                    pos = list(chosen_sequence.index).index(first_idx) + 1
                    y_till = yellow_sequence[:pos]
                    b_till = blue_sequence[:pos]
                    tuples_yb = list(zip(np.cumsum(y_till), np.cumsum(b_till)))
                    total_y = sum(y_till)
                    total_b = sum(b_till)

                    action_val = data_block.loc[first_idx, "action_taken"]
                    results[key] = {
                        "missed": False,
                        "total_yellow": total_y,
                        "total_blue": total_b,
                        "full_sequence_length": full_seq_len,
                        "action_taken": action_val,
                    }

                    if full_seq_len <= 8:
                        pair_count_short[(total_y, total_b)] += 1
                        if action_val == 0:
                            yellow_choice_short[(total_y, total_b)] += 1
                        elif action_val == 1:
                            blue_choice_short[(total_y, total_b)] += 1
                    else:
                        pair_count_long[(total_y, total_b)] += 1
                        if action_val == 0:
                            yellow_choice_long[(total_y, total_b)] += 1
                        elif action_val == 1:
                            blue_choice_long[(total_y, total_b)] += 1

                if full_seq_len <= 8:
                    all_occurrences_short.extend(tuples_yb)
                    results[key]["horizon_condition"] = "short"
                else:
                    all_occurrences_long.extend(tuples_yb)
                    results[key]["horizon_condition"] = "long"

    occ_short = Counter(all_occurrences_short)
    occ_long = Counter(all_occurrences_long)

    def _build(
        decide: Counter, color: Counter, occ: Counter
    ) -> tuple[dict, dict]:
        """Format "count/total_visits" strings for a decide-Counter and a color-Counter."""
        cd, cc = {}, {}
        for tpl, denom in occ.items():
            cd[tpl] = f"{decide.get(tpl, 0)}/{denom}"
            cc[tpl] = f"{color.get(tpl, 0)}/{denom}"
        return cd, cc

    counts_dict_short, counts_dict_yellow_short = _build(
        pair_count_short, yellow_choice_short, occ_short
    )
    counts_dict_long, counts_dict_yellow_long = _build(
        pair_count_long, yellow_choice_long, occ_long
    )
    _, counts_dict_blue_short = _build(pair_count_short, blue_choice_short, occ_short)
    _, counts_dict_blue_long = _build(pair_count_long, blue_choice_long, occ_long)

    return {
        "results": results,
        "num_games": num_games,
        "counts_dict_short": counts_dict_short,
        "counts_dict_long": counts_dict_long,
        "counts_dict_yellow_short": counts_dict_yellow_short,
        "counts_dict_yellow_long": counts_dict_yellow_long,
        "counts_dict_blue_short": counts_dict_blue_short,
        "counts_dict_blue_long": counts_dict_blue_long,
    }
