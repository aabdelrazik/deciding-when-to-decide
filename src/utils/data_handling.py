# Imports
import os
import re
import glob
import pandas as pd
import pickle
import ast
import re
import numpy as np
import importlib.util
from typing import Any

# Add the src directory to the Python path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon, pearsonr
from statsmodels.stats.multitest import multipletests
from .utils import safe_spearman


def clean_and_merge_betas(
    pdecide_betas_arr: np.ndarray,
    n_betas: int,
    ocir_df: pd.DataFrame,
    ybocs_df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create beta dataframe, remove outliers, and merge questionnaire data.

    Args:
        pdecide_betas_arr (np.ndarray): Array of shape (n_subjects, n_betas + 1),
            with the last column holding userID.
        n_betas (int): Number of beta columns in pdecide_betas_arr.
        ocir_df (pd.DataFrame): Questionnaire dataframe with a 'userID' column.
        ybocs_df (pd.DataFrame): YBOCS dataframe with a 'userID' column.
        iqr_multiplier (float): Multiplier applied to the IQR for outlier bounds.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - betas_clean: Outlier-filtered beta dataframe.
        - betas_ocir: betas_clean merged with ocir_df.
        - betas_allq: betas_ocir merged with ybocs_df.
    """

    import pandas as pd
    import numpy as np

    # -------- build beta dataframe --------
    cols = [f"beta{i}" for i in range(1, n_betas + 1)] + ["userID"]
    betas_df = pd.DataFrame(pdecide_betas_arr, columns=cols)

    for c in cols[:-1]:
        betas_df[c] = pd.to_numeric(betas_df[c], errors="coerce")

    # -------- IQR outlier filtering --------
    num_cols = betas_df.select_dtypes(include=[np.number]).columns

    mask = pd.Series(True, index=betas_df.index)

    for col in num_cols:
        q25 = betas_df[col].quantile(0.25)
        q75 = betas_df[col].quantile(0.75)
        iqr = q75 - q25

        mask &= (betas_df[col] >= q25 - iqr_multiplier * iqr) & (
            betas_df[col] <= q75 + iqr_multiplier * iqr
        )

    betas_clean = betas_df.loc[mask].dropna().copy()

    # -------- consistent userID type --------
    betas_clean.loc[:, "userID"] = betas_clean["userID"].astype(int)

    ocir_df = ocir_df.copy()
    ybocs_df = ybocs_df.copy()

    ocir_df.loc[:, "userID"] = ocir_df["userID"].astype(int)
    ybocs_df.loc[:, "userID"] = ybocs_df["userID"].astype(int)

    # -------- merge --------
    betas_ocir = betas_clean.merge(ocir_df, on="userID", how="inner")
    betas_allq = betas_ocir.merge(ybocs_df, on="userID", how="inner")

    return betas_clean, betas_ocir, betas_allq


def prepare_choice_trials(
    pmat_all_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract choice trials and split into long/short termination groups.

    Args:
        pmat_all_df (pd.DataFrame): Trial-level dataframe with 'decide',
            'termination', and 'userID' columns.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - pmat_choice: Rows where decide == 1.
        - pmat_long: pmat_choice rows where termination == 1.
        - pmat_short: pmat_choice rows where termination == 2.
    """

    pmat_choice = pmat_all_df.loc[pmat_all_df["decide"] == 1].copy()
    pmat_choice.loc[:, "userID"] = pmat_choice["userID"].astype(str)

    pmat_long = pmat_choice.loc[pmat_choice["termination"] == 1].copy()
    pmat_short = pmat_choice.loc[pmat_choice["termination"] == 2].copy()

    return pmat_choice, pmat_long, pmat_short


def assemble_glm_outputs(betas_all: list[dict | None], id_all: list) -> dict:
    """
    Convert subject-level GLM outputs into long-form DataFrames.

    Parameters
    ----------
    betas_all : list[dict | None]
        Per-subject GLM fit outputs (each with keys 'pdecide_beta', 'mu',
        'sigma', 'decide', 'pmat', 'pmat_z'), or None for a failed subject.
    id_all : list
        Subject IDs aligned with betas_all.

    Returns
    -------
    dict with:
        pmat_all_df
        pmat_z_all_df
        pdecide_betas
        pmat_choice
        pmat_long
        pmat_short
        mu_df
        sigma_df
        n_betas
    """

    if len(betas_all) == 0:
        raise ValueError("betas_all is empty")

    first_valid = next((b for b in betas_all if b is not None), None)
    if first_valid is None:
        raise RuntimeError("No valid subject betas found")

    n_betas = len(first_valid["pdecide_beta"])

    pmat_rows = []
    pmat_z_rows = []
    beta_rows = []
    mu_rows = []
    sigma_rows = []

    columns = [
        "decide",
        "totevminus",
        "deltaev",
        "trial",
        "termination",
        "totevminusxterm",
        "trialxterm",
    ]

    pred_cols = columns[1:]  # predictors only

    for uid, ba in zip(id_all, betas_all):

        uid = str(uid)

        if ba is None:
            beta_rows.append(np.concatenate([np.full(n_betas, np.nan), [uid]]))
            mu_rows.append(np.concatenate([np.full(len(pred_cols), np.nan), [uid]]))
            sigma_rows.append(np.concatenate([np.full(len(pred_cols), np.nan), [uid]]))
            continue

        beta_rows.append(np.concatenate([ba["pdecide_beta"].reshape(-1), [uid]]))

        mu_rows.append(np.concatenate([ba["mu"].reshape(-1), [uid]]))

        sigma_rows.append(np.concatenate([ba["sigma"].reshape(-1), [uid]]))

        decide = np.asarray(ba["decide"]).reshape(-1)

        pmat = np.asarray(ba["pmat"])
        pmat_z = np.asarray(ba["pmat_z"])

        if pmat.ndim == 1:
            pmat = pmat.reshape(-1, 1)

        if pmat_z.ndim == 1:
            pmat_z = pmat_z.reshape(-1, 1)

        df = pd.DataFrame(
            np.column_stack([decide, pmat[:, :6]]),
            columns=columns,
        )
        df["userID"] = uid

        df_z = pd.DataFrame(
            np.column_stack([decide, pmat_z[:, :6]]),
            columns=columns,
        )
        df_z["userID"] = uid

        pmat_rows.append(df)
        pmat_z_rows.append(df_z)

    pmat_all_df = pd.concat(pmat_rows, ignore_index=True)
    pmat_z_all_df = pd.concat(pmat_z_rows, ignore_index=True)

    pdecide_betas = np.vstack(beta_rows)

    mu_df = pd.DataFrame(mu_rows, columns=pred_cols + ["userID"])
    sigma_df = pd.DataFrame(sigma_rows, columns=pred_cols + ["userID"])

    # choice trials
    pmat_choice = pmat_all_df.loc[pmat_all_df["decide"] == 1].copy()
    pmat_choice["userID"] = pmat_choice["userID"].astype(str)

    pmat_long = pmat_choice.loc[pmat_choice["termination"] == 1].copy()
    pmat_short = pmat_choice.loc[pmat_choice["termination"] == 2].copy()

    return {
        "pmat_all_df": pmat_all_df,
        "pmat_z_all_df": pmat_z_all_df,
        "pdecide_betas": pdecide_betas,
        "pmat_choice": pmat_choice,
        "pmat_long": pmat_long,
        "pmat_short": pmat_short,
        "mu_df": mu_df,
        "sigma_df": sigma_df,
        "n_betas": n_betas,
    }


def assemble_glm_outputs_archived(betas_all: list[dict | None], id_all: list) -> dict:
    """
    Convert subject-level GLM outputs into long-form DataFrames.

    Archived variant of assemble_glm_outputs() that omits the mu_df/sigma_df
    outputs.

    Parameters
    ----------
    betas_all : list[dict | None]
        Per-subject GLM fit outputs (each with keys 'pdecide_beta', 'decide',
        'pmat', 'pmat_z'), or None for a failed subject.
    id_all : list
        Subject IDs aligned with betas_all.

    Returns
    -------
    dict with:
        pmat_all_df
        pmat_z_all_df
        pdecide_betas
        pmat_choice
        pmat_long
        pmat_short
        n_betas
    """

    if len(betas_all) == 0:
        raise ValueError("betas_all is empty")

    first_valid = next((b for b in betas_all if b is not None), None)
    if first_valid is None:
        raise RuntimeError("No valid subject betas found")

    n_betas = len(first_valid["pdecide_beta"])

    pmat_rows = []
    pmat_z_rows = []
    beta_rows = []

    columns = [
        "decide",
        "totevminus",
        "deltaev",
        "trial",
        "termination",
        "totevminusxterm",
        "trialxterm",
    ]

    for uid, ba in zip(id_all, betas_all):

        uid = str(uid)

        if ba is None:
            beta_rows.append(np.concatenate([np.full(n_betas, np.nan), [uid]]))
            continue

        beta_rows.append(np.concatenate([ba["pdecide_beta"].reshape(-1), [uid]]))

        decide = np.asarray(ba["decide"]).reshape(-1)

        pmat = np.asarray(ba["pmat"])
        pmat_z = np.asarray(ba["pmat_z"])

        if pmat.ndim == 1:
            pmat = pmat.reshape(-1, 1)

        if pmat_z.ndim == 1:
            pmat_z = pmat_z.reshape(-1, 1)

        df = pd.DataFrame(
            np.column_stack([decide, pmat[:, :6]]),
            columns=columns,
        )
        df["userID"] = uid

        df_z = pd.DataFrame(
            np.column_stack([decide, pmat_z[:, :6]]),
            columns=columns,
        )
        df_z["userID"] = uid

        pmat_rows.append(df)
        pmat_z_rows.append(df_z)

    pmat_all_df = pd.concat(pmat_rows, ignore_index=True)
    pmat_z_all_df = pd.concat(pmat_z_rows, ignore_index=True)

    pdecide_betas = np.vstack(beta_rows)

    # choice trials
    pmat_choice = pmat_all_df.loc[pmat_all_df["decide"] == 1].copy()
    pmat_choice["userID"] = pmat_choice["userID"].astype(str)

    pmat_long = pmat_choice.loc[pmat_choice["termination"] == 1].copy()
    pmat_short = pmat_choice.loc[pmat_choice["termination"] == 2].copy()

    return {
        "pmat_all_df": pmat_all_df,
        "pmat_z_all_df": pmat_z_all_df,
        "pdecide_betas": pdecide_betas,
        "pmat_choice": pmat_choice,
        "pmat_long": pmat_long,
        "pmat_short": pmat_short,
        "n_betas": n_betas,
    }


def load_questionnaire_data(project_root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess questionnaire data.

    Steps
    -----
    1. Load FA and YBOCS score files
    2. Standardize userID formatting
    3. Detect factor columns
    4. Normalize factor scores (z-score)
    5. Ensure OCIR_total column exists
    6. Rename factors for downstream analysis

    Parameters
    ----------
    project_root : str
        Root directory of the project.

    Returns
    -------
    ocir_df : pd.DataFrame
        Cleaned questionnaire dataframe with normalized factor scores.
    ybocs_df : pd.DataFrame
        Raw YBOCS dataframe (cleaned userID).
    """

    data_root = os.path.join(project_root, "data", "TrHu_NHB_light", "data_MEG")

    # ---------------- Load data ----------------
    fa_path = os.path.join(data_root, "fa_scores.csv")
    ybocs_path = os.path.join(data_root, "ybocs_scores.csv")

    fa_df = pd.read_csv(fa_path)
    ybocs_df = pd.read_csv(ybocs_path)

    # ---------------- Clean userID ----------------
    fa_df["userID"] = fa_df["userID"].astype(str).str.strip()

    if "userID" in ybocs_df.columns:
        ybocs_df["userID"] = ybocs_df["userID"].astype(str).str.strip()

    # ---------------- Detect factor columns ----------------
    numeric_cols = fa_df.select_dtypes(include=[np.number]).columns.tolist()

    fa_candidates = [c for c in numeric_cols if c.lower().startswith("fa")]
    factor_cols = fa_candidates if len(fa_candidates) >= 1 else numeric_cols[:3]

    # ---------------- Normalize factor scores ----------------
    scaler = StandardScaler()

    fa_values = fa_df[factor_cols].to_numpy(dtype=float)

    # detect rows containing NaNs
    nan_rows = np.any(np.isnan(fa_values), axis=1)

    fa_values_norm = fa_values.copy()

    if (~nan_rows).sum() > 0:
        fa_values_norm[~nan_rows] = scaler.fit_transform(fa_values[~nan_rows])

    fa_norm_df = fa_df.copy()
    fa_norm_df.loc[:, factor_cols] = fa_values_norm

    ocir_df = fa_norm_df.copy()

    # ---------------- Ensure OCIR_total exists ----------------
    if "OCIR_total" not in ocir_df.columns:

        ocir_candidates = [
            c
            for c in ocir_df.columns
            if ("ocir" in c.lower()) and (c not in factor_cols)
        ]

        if ocir_candidates:
            ocir_df = ocir_df.rename(columns={ocir_candidates[0]: "OCIR_total"})

    # ---------------- Rename factors for consistency ----------------
    rename_map = {
        "ML1": "FA1",
        "ML2": "FA2",
        "ML3": "FA3",
    }

    ocir_df = ocir_df.rename(
        columns={k: v for k, v in rename_map.items() if k in ocir_df.columns}
    )

    ocir_df["userID"] = ocir_df["userID"].astype(str)

    return ocir_df, ybocs_df


def make_termination(row: pd.Series) -> list:
    """Broadcast a row's scalar 'termination' value to one entry per draw.

    Args:
        row (pd.Series): A trial row with 'termination' and 'totEvRight'
            (used only for its length) fields. Intended for use with
            DataFrame.apply(..., axis=1).

    Returns:
        list: The 'termination' value repeated len(row['totEvRight']) times.
    """
    value = row["termination"]  # e.g., 1
    size = len(row["totEvRight"])
    return [value] * size


def flatten_to_list(cell) -> list:
    """Flatten a list of arrays/lists into one extended Python list"""
    out = []
    for x in cell:
        out.extend(np.ravel(x).tolist())  # flatten each piece and extend
    return out


def load_simulation_results(config_dir: str) -> dict:
    """Load each config's full simulated dataset (FULL_SIM_DF_PATH), keyed by TASK.

    Args:
        config_dir (str): Directory containing simulation_params_*.py config files.

    Returns:
        dict: Mapping from the 13-char TASK code (parsed from the config
            filename) to the full simulated dataframe. Configs whose
            FULL_SIM_DF_PATH is missing on disk are silently skipped.
    """
    from src.config.loader import load_config

    config_files = glob.glob(f"{config_dir}/*.py")
    simulation_df_dict = {}

    for file_path in config_files:
        filename = os.path.basename(file_path)[:-3]  # Remove ".py"
        key = filename[
            -13:
        ]  # TASK code is always 13 chars (see schema.build_task_name)

        cfg = load_config(file_path)
        try:
            simulation_df_dict[key] = pd.read_pickle(cfg.FULL_SIM_DF_PATH)
        except (FileNotFoundError, OSError):
            continue
    return simulation_df_dict


def load_fitted_results(
    config_dir: str, commit: bool = False
) -> tuple[dict, dict]:
    """Load each config's fitted results (RESULTS_PATH) and, where available, its
    recovery-fit results (RESULTS_RECOVERED_PATH), both keyed by TASK.

    Parameters
    ----------
    config_dir : str
        Directory containing simulation_params_*.py config files.
    commit : bool, default False
        If False (default), loads the standard POMDP fits from data/POMDP/
        (excludes *_commit.py configs).
        If True, loads the commit-LL fits from data/POMDP_commit/
        (only reads *_commit.py configs).

    Returns
    -------
    tuple[dict, dict]
        results_df_dict : Mapping from TASK code to the fitted results dataframe.
        results_df_recovered_dict : Mapping from TASK code to the recovery-fit
            results dataframe (only present for configs with a readable
            RESULTS_RECOVERED_PATH).
    """
    from src.config.loader import load_config

    if commit:
        config_files = glob.glob(f"{config_dir}/simulation_params_*_commit.py")
    else:
        config_files = [
            f for f in glob.glob(f"{config_dir}/*.py")
            if not f.endswith("_commit.py")
        ]

    results_df_dict = {}
    results_df_recovered_dict = {}

    for file_path in config_files:
        cfg = load_config(file_path)
        # Use cfg.TASK (always 13 chars, always correct) instead of parsing
        # the filename — commit configs have a _commit suffix that would
        # corrupt the filename[-13:] slice.
        key = cfg.TASK

        try:
            results_df_dict[key] = pd.read_pickle(cfg.RESULTS_PATH)
        except (FileNotFoundError, OSError):
            continue
        try:
            results_df_recovered_dict[key] = pd.read_pickle(cfg.RESULTS_RECOVERED_PATH)
        except (FileNotFoundError, OSError):
            continue

    return results_df_dict, results_df_recovered_dict


def combine_sequences(all_simulated_data: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all per-game rows into one row per subject.

    Adds a 'termination' column (via make_termination) broadcast to
    per-draw length, then for each userID group flattens/concatenates every
    other column across games and trials into a single flat list per column.

    Args:
        all_simulated_data (pd.DataFrame): One row per game/trial, with a
            'userID' column and list-like value columns plus a scalar
            'termination' column.

    Returns:
        pd.DataFrame: One row per subject ('userID' plus one flattened list
            per original column).
    """

    # preprocess to add the termination
    all_simulated_data["termination"] = all_simulated_data.apply(
        make_termination, axis=1
    )

    combined_rows = []
    for uid, sub_df in all_simulated_data.groupby("userID", sort=False):
        row_data = {"userID": uid}
        # detect list-like columns automatically (ignores subject_id)
        list_cols = [c for c in sub_df.columns if c != "userID"]

        for col in list_cols:
            concatenated = []
            for val in sub_df[col]:
                if isinstance(val, (list, tuple)):
                    concatenated.extend(val)
                else:
                    concatenated.append(val)
            row_data[col] = flatten_to_list(concatenated)

        combined_rows.append((row_data))

    # here the data for all trials and games are concatenated for each subject.
    combined_df = pd.DataFrame(combined_rows)
    return combined_df


def get_from_mat(pmat: dict, name: str) -> np.ndarray:
    """Return column from pmat dict where names list matches name.

    Args:
        pmat (dict): Dict with a 'names' list of column names and a 'mat'
            2D array whose columns align with 'names'.
        name (str): Column name to look up.

    Returns:
        np.ndarray: The matching column of pmat['mat'].
    """
    try:
        idx = pmat["names"].index(name)
        return pmat["mat"][:, idx]
    except ValueError:
        raise KeyError(f"Variable {name} not found in pmat")


def extract_hist_data(human_data: pd.DataFrame) -> tuple[list, list, list, list]:
    """
    Extracts the number of draws and rewards per game, avoiding NaNs.

    This function unnests the processed data, and for each game, it reliably
    finds the single row containing the final reward. It achieves this by
    sorting the data so that valid rewards appear before NaNs, then selecting
    the first row for each unique game. It then separates the data into four
    lists based on the game's horizon condition.

    Args:
        human_data (pd.DataFrame): The DataFrame returned by
                                     analyze_all_subjects_behavior.

    Returns:
        tuple[list, list, list, list]: A tuple containing four lists:
        - list: Number of draws for each short-horizon game (termination=2).
        - list: Number of draws for each long-horizon game (termination=1).
        - list: Final, non-NaN reward for each short-horizon game.
        - list: Final, non-NaN reward for each long-horizon game.
    """
    all_data_list = []
    for _, row in human_data.iterrows():
        user_id = row["userID"]
        subject_data = row["data"].copy()
        subject_data["userID"] = user_id
        all_data_list.append(subject_data)

    if not all_data_list:
        return [], [], [], []

    all_trials_df = pd.concat(all_data_list, ignore_index=True)

    # Sort by reward (descending) and place NaNs at the bottom. This ensures
    # that for each game, the row with the actual reward is first.
    all_trials_df_sorted = all_trials_df.sort_values(
        by="reward", ascending=False, na_position="last"
    )

    # Now, when we drop duplicates, we keep the first row, which is guaranteed
    # to have the valid reward if one exists for the game.
    unique_games_df = all_trials_df_sorted.drop_duplicates(
        subset=["userID", "block", "game"], keep="first"
    )

    long_horizon_games = unique_games_df[unique_games_df["termination"] == 1]
    short_horizon_games = unique_games_df[unique_games_df["termination"] == 2]

    long_horizon_draws = long_horizon_games["num_draws"].tolist()
    short_horizon_draws = short_horizon_games["num_draws"].tolist()

    # The rewards are now guaranteed to be non-NaN values (e.g., 2, -2, or -1)
    long_horizon_rewards = long_horizon_games["reward"].tolist()
    short_horizon_rewards = short_horizon_games["reward"].tolist()

    return (
        short_horizon_draws,
        long_horizon_draws,
        short_horizon_rewards,
        long_horizon_rewards,
    )


def parse_evidence(evidence: Any) -> list:
    """
    Return a list-like representation of a single 'ev' entry.
    Handles:
      - actual lists/tuples/ndarrays/Series
      - stringified Python lists (literal_eval)
      - strings with nested-list markers '],[', or space/comma separated numbers
    """
    if evidence is None:
        return []
    # already a sequence
    if isinstance(evidence, (list, tuple, np.ndarray, pd.Series)):
        return list(evidence)
    # single numeric
    if isinstance(evidence, (int, float, np.integer, np.floating)):
        return [evidence]
    # string handling
    if isinstance(evidence, str):
        s = evidence.strip()
        # try safe eval (handles "[[1,2],[3,4]]" etc.)
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                return list(val)
            # scalar returned
            return [val]
        except Exception:
            # if there are nested list separators, count top-level sublists
            if "],[" in s:
                # rough count of top-level sublists
                return ["_sublist_"] * (s.count("],[") + 1)
            # remove outer brackets and split by commas/whitespace
            s2 = re.sub(r"^[\[\(\s]+|[\]\)\s]+$", "", s)
            parts = re.split(r"[,\s]+", s2.strip())
            parts = [p for p in parts if p != ""]
            return parts
    # fallback: convert to string and parse
    return parse_evidence(str(evidence))


# ...existing code...
import ast, re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def _count_draws_from_val(v: Any) -> int | float:
    """Parse a single cell into a draw sequence and return its length.

    Args:
        v (Any): A cell value (list-like, string, or scalar) as handled by
            parse_evidence.

    Returns:
        int | float: The number of draws parsed from v, or np.nan if v is
            missing or unparsable.
    """
    if pd.isna(v):
        return np.nan
    parsed = _parse_evidence(v)
    try:
        return int(len(parsed))
    except Exception:
        return np.nan


def extract_draws_for_subject(
    df: pd.DataFrame,
    subj,
    subject_candidates: tuple[str, ...] | None = None,
    num_draws_col: str = "num_draws",
    ev_candidates: tuple[str, ...] = ("ev", "evidence", "draws"),
    draw_col_candidates: tuple[str, ...] = (
        "draw_yellow_blue_action",
        "draw_action",
        "action",
        "draw",
    ),
) -> list[int]:
    """Get the per-game draw counts for one subject from a trial dataframe.

    Looks for an explicit num_draws_col first; if absent, falls back to
    counting parsed entries in draw_col_candidates, then ev_candidates, then
    scans all remaining columns for anything list-like/parsable.

    Args:
        df (pd.DataFrame): Trial-level dataframe containing a subject-id
            column and one or more draw/evidence columns.
        subj: Subject identifier to filter df on (compared as str).
        subject_candidates (tuple[str, ...] | None): Candidate column names
            to use as the subject-id column; defaults to a built-in list of
            common names if None.
        num_draws_col (str): Column name holding an explicit draw count.
        ev_candidates (tuple[str, ...]): Candidate evidence-column names.
        draw_col_candidates (tuple[str, ...]): Candidate draw-column names.

    Returns:
        list[int]: Number of draws for each of the subject's games.
    """
    subject_candidates = subject_candidates or (
        "subject_ID",
        "subject_id",
        "subject",
        "userID",
        "userId",
        "sim_subject",
    )
    subj_col = next((c for c in subject_candidates if c in df.columns), None)
    if subj_col is None:
        raise RuntimeError(f"No subject id column found (tried {subject_candidates})")
    rows = df[df[subj_col].astype(str) == str(subj)].copy()
    draws = []
    if rows.empty:
        return draws
    # prefer explicit num_draws if present
    if num_draws_col in rows.columns:
        draws = [int(x) for x in rows[num_draws_col].dropna().astype(float).tolist()]
        if draws:
            return draws
    # try draw-specific columns
    for c in draw_col_candidates:
        if c in rows.columns:
            for v in rows[c].tolist():
                d = _count_draws_from_val(v)
                if not pd.isna(d):
                    draws.append(int(d))
            if draws:
                return draws
    # try ev-like columns
    for c in ev_candidates:
        if c in rows.columns:
            for v in rows[c].tolist():
                d = _count_draws_from_val(v)
                if not pd.isna(d):
                    draws.append(int(d))
            if draws:
                return draws
    # fallback: scan all columns for something list-like / string that parses
    for _, r in rows.iterrows():
        for c, v in r.items():
            if c == subj_col:
                continue
            d = _count_draws_from_val(v)
            if not pd.isna(d):
                draws.append(int(d))
    return draws


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


def extract_draws_for_subject(
    df: pd.DataFrame,
    subj,
    subject_candidates: tuple[str, ...] | None = None,
    num_draws_col: str = "num_draws",
    ev_candidates: tuple[str, ...] = ("ev", "evidence", "draws"),
    draw_col_candidates: tuple[str, ...] = (
        "draw_yellow_blue_action",
        "draw_action",
        "action",
        "draw",
    ),
) -> list[int]:
    """Get the per-game draw counts for one subject from a trial dataframe.

    Looks for an explicit num_draws_col first; if absent, falls back to
    counting parsed entries in draw_col_candidates, then ev_candidates, then
    scans all remaining columns for anything list-like/parsable.

    Args:
        df (pd.DataFrame): Trial-level dataframe containing a subject-id
            column and one or more draw/evidence columns.
        subj: Subject identifier to filter df on (compared as str).
        subject_candidates (tuple[str, ...] | None): Candidate column names
            to use as the subject-id column; defaults to a built-in list of
            common names if None.
        num_draws_col (str): Column name holding an explicit draw count.
        ev_candidates (tuple[str, ...]): Candidate evidence-column names.
        draw_col_candidates (tuple[str, ...]): Candidate draw-column names.

    Returns:
        list[int]: Number of draws for each of the subject's games.
    """
    subject_candidates = subject_candidates or (
        "subject_ID",
        "subject_id",
        "subject",
        "userID",
        "userId",
        "sim_subject",
    )
    subj_col = next((c for c in subject_candidates if c in df.columns), None)
    if subj_col is None:
        raise RuntimeError(f"No subject id column found (tried {subject_candidates})")
    rows = df[df[subj_col].astype(str) == str(subj)].copy()
    draws = []
    if rows.empty:
        return draws
    # prefer explicit num_draws if present
    if num_draws_col in rows.columns:
        draws = [int(x) for x in rows[num_draws_col].dropna().astype(float).tolist()]
        if draws:
            return draws
    # try draw-specific columns
    for c in draw_col_candidates:
        if c in rows.columns:
            for v in rows[c].tolist():
                d = _count_draws_from_val(v)
                if not pd.isna(d):
                    draws.append(int(d))
            if draws:
                return draws
    # try ev-like columns
    for c in ev_candidates:
        if c in rows.columns:
            for v in rows[c].tolist():
                d = _count_draws_from_val(v)
                if not pd.isna(d):
                    draws.append(int(d))
            if draws:
                return draws
    # fallback: scan all columns for something list-like / string that parses
    for _, r in rows.iterrows():
        for c, v in r.items():
            if c == subj_col:
                continue
            d = _count_draws_from_val(v)
            if not pd.isna(d):
                draws.append(int(d))
    return draws


# def extract_hist_data(human_data_subject: pd.DataFrame) -> tuple[list, list, list, list]:
#     """
#     Extracts the number of draws and rewards per game, avoiding NaNs.

#     This function unnests the processed data, and for each game, it reliably
#     finds the single row containing the final reward. It achieves this by
#     sorting the data so that valid rewards appear before NaNs, then selecting
#     the first row for each unique game. It then separates the data into four
#     lists based on the game's horizon condition.

#     Args:
#         human_data (pd.DataFrame): The DataFrame returned by
#                                      analyze_all_subjects_behavior.

#     Returns:
#         tuple[list, list, list, list]: A tuple containing four lists:
#         - list: Number of draws for each short-horizon game (termination=2).
#         - list: Number of draws for each long-horizon game (termination=1).
#         - list: Final, non-NaN reward for each short-horizon game.
#         - list: Final, non-NaN reward for each long-horizon game.
#     """

#     all_trials_df = human_data_subject

#     # Sort by reward (descending) and place NaNs at the bottom. This ensures
#     # that for each game, the row with the actual reward is first.
#     # all_trials_df_sorted = all_trials_df.sort_values(
#     #     by='reward', ascending=False, na_position='last'
#     # )

#     # Now, when we drop duplicates, we keep the first row, which is guaranteed
#     # to have the valid reward if one exists for the game.
#     unique_games_df = all_trials_df.drop_duplicates(
#         subset=[ 'block', 'game'], keep='first'
#     )

#     long_horizon_games = unique_games_df[unique_games_df['termination'] == 1]
#     short_horizon_games = unique_games_df[unique_games_df['termination'] == 2]

#     long_horizon_draws = long_horizon_games['num_draws'].tolist()
#     short_horizon_draws = short_horizon_games['num_draws'].tolist()

#     # The rewards are now guaranteed to be non-NaN values (e.g., 2, -2, or -1)
#     long_horizon_rewards = long_horizon_games['reward'].tolist()
#     short_horizon_rewards = short_horizon_games['reward'].tolist()

#     return short_horizon_draws, long_horizon_draws, short_horizon_rewards, long_horizon_rewards


def extract_hist_data_for_user(
    human_data: pd.DataFrame, user_id: int
) -> tuple[list, list, list, list]:
    """
    Extracts the number of draws and rewards per game for a single user, avoiding NaNs.

    Args:
        human_data (pd.DataFrame): The DataFrame returned by analyze_all_subjects_behavior.
        user_id (int): The specific userID to extract data for.

    Returns:
        tuple[list, list, list, list]: A tuple containing four lists:
        - list: Number of draws for each short-horizon game (termination=2).
        - list: Number of draws for each long-horizon game (termination=1).
        - list: Final, non-NaN reward for each short-horizon game.
        - list: Final, non-NaN reward for each long-horizon game.
    """
    # Filter human_data for the specific user
    user_rows = human_data[human_data["userID"] == user_id]

    if user_rows.empty:
        return [], [], [], []

    all_data_list = []
    for _, row in user_rows.iterrows():
        subject_data = row["data"].copy()
        subject_data["userID"] = user_id
        all_data_list.append(subject_data)

    all_trials_df = pd.concat(all_data_list, ignore_index=True)

    # Sort by reward so valid values appear before NaNs
    all_trials_df_sorted = all_trials_df.sort_values(
        by="reward", ascending=False, na_position="last"
    )

    # Keep only one row per unique game (first = guaranteed valid reward)
    unique_games_df = all_trials_df_sorted.drop_duplicates(
        subset=["userID", "block", "game"], keep="first"
    )

    # Separate by horizon
    long_horizon_games = unique_games_df[unique_games_df["termination"] == 1]
    short_horizon_games = unique_games_df[unique_games_df["termination"] == 2]

    long_horizon_draws = long_horizon_games["num_draws"].tolist()
    short_horizon_draws = short_horizon_games["num_draws"].tolist()
    long_horizon_rewards = long_horizon_games["reward"].tolist()
    short_horizon_rewards = short_horizon_games["reward"].tolist()

    return (
        short_horizon_draws,
        long_horizon_draws,
        short_horizon_rewards,
        long_horizon_rewards,
    )


def find_early_deciders_all_simulated(
    all_sim_df: pd.DataFrame,
    subject_id_col_candidates: tuple[str, ...] = (
        "subject_ID",
        "subject",
        "userID",
        "subjectId",
        "sim_subject",
    ),
    ev_col_candidates: tuple[str, ...] = ("ev", "evidence", "draws"),
    draw_col_candidates: tuple[str, ...] = (
        "draw_yellow_blue_action",
        "draw_action",
        "draw",
    ),
    max_draw: int = 3,
    min_games: int = 2,
    max_examples_per_subject: int = 5,
) -> pd.DataFrame:
    """
    Find subjects in a flattened all_simulated_data DataFrame who decided at draw <= max_draw
    in at least min_games. Works with rows that store a per-game sequence in columns like 'ev'
    or 'draw_yellow_blue_action'. Returns DataFrame similar to previous helper.

    Args:
        all_sim_df (pd.DataFrame): Flattened simulated-data rows, one per
            game, with a subject-id column and a per-game sequence column.
        subject_id_col_candidates (tuple[str, ...]): Candidate column names
            for the subject id; the first one present in all_sim_df is used.
        ev_col_candidates (tuple[str, ...]): Candidate evidence-column names.
        draw_col_candidates (tuple[str, ...]): Candidate draw-sequence
            column names (checked before ev_col_candidates).
        max_draw (int): Decisions at or before this draw count as "early".
        min_games (int): Minimum number of early-decision games required for
            a subject to be included in the output.
        max_examples_per_subject (int): Max number of example games stored
            per subject in 'early_details'.

    Returns:
        pd.DataFrame: Columns 'subject_ID', 'early_count' (int), and
            'early_details' (list of example dicts), sorted by early_count
            descending. Empty (with these columns) if no subject qualifies.
    """

    # helpers
    def _parse_evidence(evidence):
        if evidence is None:
            return []
        if isinstance(evidence, (list, tuple, np.ndarray, pd.Series)):
            return list(evidence)
        if isinstance(evidence, (int, float, np.integer, np.floating)):
            return [evidence]
        if isinstance(evidence, str):
            s = evidence.strip()
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    return list(val)
                return [val]
            except Exception:
                if "],[" in s:
                    return ["_sublist_"] * (s.count("],[") + 1)
                s2 = re.sub(r"^[\[\(\s]+|[\]\)\s]+$", "", s)
                parts = re.split(r"[,\s]+", s2.strip())
                return [p for p in parts if p != ""]
        return _parse_evidence(str(evidence))

    def _extract_yellow_from_draw(draw):
        if isinstance(draw, (list, tuple)):
            if len(draw) > 1:
                try:
                    return int(draw[1])
                except Exception:
                    return draw[1]
            return None
        if isinstance(draw, dict):
            for key in (
                "yellow",
                "draw_yellow",
                "yellow_count",
                "n_yellow",
                "left",
                "total_left",
                "y",
                "n_left",
            ):
                if key in draw:
                    try:
                        return int(draw[key])
                    except Exception:
                        return draw[key]
            for v in draw.values():
                if isinstance(v, (int, float)):
                    return int(v)
            return None
        try:
            return int(draw)
        except Exception:
            return None

    def _get_action_from_draw(draw):
        # returns action or None
        if isinstance(draw, (list, tuple)):
            if len(draw) >= 4:
                return draw[3]
            # fallback: last element
            try:
                return draw[-1]
            except Exception:
                return None
        if isinstance(draw, dict):
            for k in (
                "action",
                "draw_action",
                "draw_yellow_blue_action",
                "choice",
                "response",
                "decided",
            ):
                if k in draw:
                    return draw[k]
            # try to find any field that looks like an action (scalar)
            for v in draw.values():
                if isinstance(v, (int, float, str)):
                    return v
            return None
        # scalar
        return draw

    # determine subject id column
    subject_col = None
    for c in subject_id_col_candidates:
        if c in all_sim_df.columns:
            subject_col = c
            break
    if subject_col is None:
        raise RuntimeError(
            f"No subject id column found in all_sim_df. Tried {subject_id_col_candidates}"
        )

    # determine ev/draw column
    ev_col = None
    draw_col = None
    for c in ev_col_candidates:
        if c in all_sim_df.columns:
            ev_col = c
            break
    for c in draw_col_candidates:
        if c in all_sim_df.columns:
            draw_col = c
            break

    rows = []
    # group by subject
    for sid, g in all_sim_df.groupby(subject_col):
        early_count = 0
        examples = []
        # iterate games (rows) for this subject
        for _, row in g.iterrows():
            # prefer explicit draw_col if present (some datasets store per-draw lists in that column)
            seq_raw = None
            if draw_col and draw_col in row and pd.notna(row[draw_col]):
                seq_raw = row[draw_col]
            elif ev_col and ev_col in row and pd.notna(row[ev_col]):
                seq_raw = row[ev_col]
            else:
                # try entire row: some dumps store 'ev' as string in other columns
                # quickly attempt to find any column that looks like a sequence
                for c in all_sim_df.columns:
                    if c in (subject_col,):
                        continue
                    v = row[c]
                    if (
                        isinstance(v, (list, tuple, np.ndarray, pd.Series))
                        and len(v) > 0
                    ):
                        seq_raw = v
                        break
                    if isinstance(v, str) and (
                        v.strip().startswith("[") or "," in v or " " in v
                    ):
                        seq_raw = v
                        break
            if seq_raw is None:
                continue

            seq = _parse_evidence(seq_raw)
            if not isinstance(seq, list) or len(seq) == 0:
                continue

            # detect decision position
            decision_pos = None
            for i, draw in enumerate(seq):
                action = _get_action_from_draw(draw)
                if action is None:
                    continue
                try:
                    if (
                        isinstance(action, (int, float))
                        and not np.isnan(action)
                        and int(action) != 2
                    ) or (
                        isinstance(action, str)
                        and action.lower() not in ("2", "wait", "missing", "nan")
                    ):
                        decision_pos = i + 1
                        break
                except Exception:
                    if str(action).lower() not in ("2", "wait", "missing", "nan"):
                        decision_pos = i + 1
                        break

            if decision_pos is None:
                continue

            if decision_pos <= max_draw:
                early_count += 1
                yellow_seq = [_extract_yellow_from_draw(d) for d in seq[:decision_pos]]
                yellow_seq_pre = (
                    [_extract_yellow_from_draw(d) for d in seq[: decision_pos - 1]]
                    if decision_pos > 1
                    else []
                )
                example = {
                    "decision_draw": int(decision_pos),
                    "yellow_sequence": yellow_seq,
                    "yellow_sequence_predecision": yellow_seq_pre,
                    "example_seq": seq[
                        : min(len(seq), max(2 * max_draw, decision_pos))
                    ],
                }
                if len(examples) < max_examples_per_subject:
                    examples.append(example)

        if early_count >= min_games:
            rows.append(
                {
                    "subject_ID": sid,
                    "early_count": int(early_count),
                    "early_details": examples,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["subject_ID", "early_count", "early_details"])

    out = (
        pd.DataFrame(rows)
        .sort_values("early_count", ascending=False)
        .reset_index(drop=True)
    )
    return out


# The following functions are to handle data
# Save Glmm data


def save_results(data: Any, filename: str, path: str = "data", index: bool = True) -> None:
    """Save `data` to disk as both a pickle (.pkl) and a CSV, under `path`.

    Args:
        data (Any): Object to save. DataFrames are written directly to CSV;
            other types are converted to a DataFrame first (falling back to
            wrapping dicts/scalars in a single-row DataFrame, or a
            multi-row DataFrame for lists, if direct conversion fails).
        filename (str): Output filename (extension, if any, is stripped and
            replaced with .pkl / .csv).
        path (str): Directory to save into; created if missing.
        index (bool): Whether to write the DataFrame index to the CSV.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists

    # Remove file extension if present
    filename_base = os.path.splitext(filename)[0]

    # Save as pickle
    pickle_filepath = os.path.join(path, f"{filename_base}.pkl")
    with open(pickle_filepath, "wb") as file:
        pickle.dump(data, file)
    print(f"Pickle file saved to {pickle_filepath}")

    # Save as CSV
    csv_filepath = os.path.join(path, f"{filename_base}.csv")

    if isinstance(data, pd.DataFrame):
        data.to_csv(csv_filepath, index=index)
    else:
        # If it's not a DataFrame, convert it to one if possible
        try:
            df = pd.DataFrame(data)
            df.to_csv(csv_filepath, index=index)
        except:
            # If conversion to DataFrame fails, write as a simple CSV
            if isinstance(data, dict):
                pd.DataFrame([data]).to_csv(csv_filepath, index=index)
            elif isinstance(data, list):
                pd.DataFrame(data).to_csv(csv_filepath, index=index)
            else:
                pd.DataFrame([data]).to_csv(csv_filepath, index=index)


def load_results(filename: str, path: str = "data") -> Any:
    """Load a pickle file previously written by save_results.

    Args:
        filename (str): Name of the pickle file (with or without extension
            is not handled here; the path is joined as-is).
        path (str): Directory the file lives in.

    Returns:
        Any: The unpickled object.
    """
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")
    with open(filepath, "rb") as file:
        data = pickle.load(file)
    print(f"Results loaded from {filepath}")
    return data


def save_dataframe(
    df: pd.DataFrame, noise_std, threshold=1, path: str = "data"
) -> str:
    """Save `df` as both Parquet and CSV, named by noise_std/threshold.

    Args:
        df (pd.DataFrame): DataFrame to save.
        noise_std: Noise standard deviation used in the simulation (used
            only to build the filename).
        threshold: Threshold value used in the simulation (used only to
            build the filename).
        path (str): Directory (relative to this file) to save into.

    Returns:
        str: Full path to the saved Parquet file (written even if the
            Parquet save fails and only the CSV is written).
    """
    # Get the full path of the base directory relative to the current file
    base_path = os.path.join(os.path.dirname(__file__), path)

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)
    # Create the filename
    filename = f"df_noise{noise_std}_thresh{threshold}.parquet"
    # Join the directory and filename to get the full path
    full_path = os.path.join(base_path, filename)
    # Save the DataFrame to the file
    # Create filenames for both Parquet and CSV
    parquet_filename = f"df_noise{noise_std}_thresh{threshold}.parquet"
    csv_filename = f"df_noise{noise_std}_thresh{threshold}.csv"

    # Full paths for both formats
    parquet_full_path = os.path.join(base_path, parquet_filename)
    csv_full_path = os.path.join(base_path, csv_filename)

    try:
        df.to_parquet(parquet_full_path, engine="fastparquet")
        df.to_csv(csv_full_path, index=False)
    except:
        df.to_csv(csv_full_path, index=False)
    return full_path


def load_saved_dataframes_dict_format(
    noise_stds: list, thresholds: list, path: str = "data"
) -> dict:
    """
    Loads saved DataFrames based on the noise_stds and thresholds provided.
    The results are returned in the same nested dictionary format as the results output.

    Args:
        noise_stds (list): List of noise standard deviations used in the simulations.
        thresholds (list): List of threshold values used in the simulations.
        path (str): Directory where the DataFrames are stored.

    Returns:
        dict: Nested dictionary with the structure results[noise_std][threshold] = DataFrame
    """
    # Initialize the nested dictionary to store results
    results = {}

    # Get the full path of the base directory relative to the current file
    base_path = os.path.join(os.path.dirname(__file__), path)

    # Iterate over noise_stds and thresholds to load the corresponding DataFrames
    for noise_std in noise_stds:
        results[noise_std] = {}
        for threshold in thresholds:
            # Construct the expected filename
            filename = f"df_noise{noise_std}_thresh{threshold}.parquet"
            full_path = os.path.join(base_path, filename)

            try:
                # Load the DataFrame if it exists
                if os.path.exists(full_path):
                    df = pd.read_parquet(full_path, engine="fastparquet")
                    results[noise_std][threshold] = df
                else:
                    print(
                        f"Warning: File not found for noise_std={noise_std}, threshold={threshold} at {full_path}"
                    )
            except Exception as exc:
                print(
                    f"Error loading file for noise_std={noise_std}, threshold={threshold}: {exc}"
                )

    return results


from collections import Counter
import pandas as pd


def prepare_evidence_data(ev_per_subject: list[list[int]], outcome: list) -> pd.DataFrame:
    """Build per-draw (draw, yellow, blue, action) tuples for each game.

    For each game (row of ev_per_subject), builds one [draw_index,
    yellow_count, blue_count, action] tuple per draw, with action defaulting
    to 2 ("waiting") except on the final draw, where it is set to 0/1
    depending on which color was in the majority and whether outcome == 100.

    Args:
        ev_per_subject (list[list[int]]): One entry per game; each entry is
            a sequence of running yellow-draw counts, one per draw in that
            game.
        outcome (list): Per-game outcome value (100 indicates the "yellow"
            choice was rewarded), aligned with ev_per_subject.

    Returns:
        pd.DataFrame: Single column 'draw_yellow_blue_action', one row per
            game, each holding that game's list of per-draw tuples.
    """
    all_results = []
    yellow_blue_counts = []
    all_occurrences_pairs = []

    for i, row in enumerate(ev_per_subject):
        tuples = []
        row_counts = []

        for k, yellow in enumerate(row):
            blue = k - yellow + 1
            y, b = yellow, blue
            action = 2  # default: waiting
            tuples.append([k + 1, int(y), int(b), action])
            row_counts.append((int(y), int(b)))

        last = tuples[-1]
        if last[1] > last[2]:
            last[3] = 0 if outcome[i] == 100 else 1
        else:
            last[3] = 1 if outcome[i] == 100 else 0

        all_results.append(tuples)
        yellow_blue_counts.append(row_counts)
        all_occurrences_pairs.extend(row_counts)

    summary = [(len(row_counts), row_counts[-1]) for row_counts in yellow_blue_counts]
    last_tuple_counts = Counter([x[1] for x in summary])
    all_occurrences_pairs_counts = Counter(all_occurrences_pairs)
    counts_dict = {
        tpl: f"{count}/{all_occurrences_pairs_counts[tpl]}"
        for tpl, count in last_tuple_counts.items()
    }
    df_per_subject = pd.DataFrame({"draw_yellow_blue_action": all_results})

    return df_per_subject


# --- GLM-vs-POMDP ensemble draw-count comparison helpers, used by the
# plot_glm_subject_grid / plot_glm_pooled_comparison / plot_per_draw_discrepancy
# / plot_pomdp_vs_glm_fair_comparison plotting functions in plotting.py
# (see notebooks/Magda's_glm_fitting.ipynb). ---


def slice_decisions_by_lengths(
    decisions_array: np.ndarray, game_lengths_list: list[int]
) -> list[np.ndarray]:
    """Splits a 1D decisions array into a list of arrays based on game lengths."""
    split_indices = np.cumsum(game_lengths_list)[:-1]
    decisions_split = np.split(decisions_array, split_indices)
    return decisions_split


def truncate_at_first_decision(list_of_lists: list) -> list[list]:
    """Truncates each sub-list immediately after the first occurrence of a 1."""
    truncated_games = []
    for game in list_of_lists:
        game_list = list(game)
        if 1 in game_list:
            first_one_index = game_list.index(1)
            truncated_games.append(game_list[: first_one_index + 1])
        else:
            truncated_games.append(game_list)
    return truncated_games


def compute_corrected_glm_draws(
    final_clean_games: list[list], games_lengths: list[int]
) -> list[int]:
    """For each game, if it has a 1 (model decided), use its length. If it's
    empty (no 1 found), the model reached end without deciding -> use full
    game_length.
    """
    glm_draws = []
    for game, game_length in zip(final_clean_games, games_lengths):
        if len(game) > 0:
            glm_draws.append(len(game))
        else:
            glm_draws.append(game_length)
    return glm_draws


def compute_ensemble_glm_counts(
    probabilities: np.ndarray,
    games_lengths: list[int],
    x_draws: list[int],
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Draws `n_samples` stochastic realizations from the GLM's predicted
    probabilities, computes the corrected draw-count histogram for each, and
    returns the mean and std across samples (instead of a single stochastic
    realization).

    Args:
        probabilities (np.ndarray): Per-trial decide-probabilities (flat
            across all games), used to sample Bernoulli decisions.
        games_lengths (list[int]): Number of trials in each game, aligned
            with `probabilities` (used to slice into per-game sequences).
        x_draws (list[int]): Draw-count values (histogram bins) to tabulate.
        n_samples (int): Number of stochastic realizations to average over.

    Returns:
        tuple[np.ndarray, np.ndarray]: (mean_counts, std_counts), each of
            shape (len(x_draws),), across the n_samples realizations.
    """
    all_counts = np.zeros((n_samples, len(x_draws)))

    for s in range(n_samples):
        decisions = np.random.binomial(1, probabilities)
        list_of_lists = slice_decisions_by_lengths(decisions, games_lengths)
        clean_games = truncate_at_first_decision(list_of_lists)
        glm_draws = compute_corrected_glm_draws(clean_games, games_lengths)
        all_counts[s] = [glm_draws.count(x) for x in x_draws]

    return all_counts.mean(axis=0), all_counts.std(axis=0)


def combine_ensemble_horizons(
    ensemble_long: dict, ensemble_short: dict, target_id
) -> dict | None:
    """Sums the long- and short-horizon ensemble distributions for one subject.

    Args:
        ensemble_long (dict): Maps subject id to a dict with 'human_counts',
            'avg_sim_counts', 'std_sim_counts' for the long-horizon condition.
        ensemble_short (dict): Same shape as ensemble_long, for the
            short-horizon condition.
        target_id: Subject id to look up in both dicts.

    Returns:
        dict | None: Combined counts/std for target_id, or whichever of the
            two horizon entries is present if only one exists, or None if
            neither exists.
    """
    long_data = ensemble_long.get(target_id)
    short_data = ensemble_short.get(target_id)
    if long_data is None or short_data is None:
        return long_data or short_data

    return {
        "human_counts": long_data["human_counts"] + short_data["human_counts"],
        "avg_sim_counts": long_data["avg_sim_counts"] + short_data["avg_sim_counts"],
        # Variances add for independent ensembles being summed.
        "std_sim_counts": np.sqrt(
            long_data["std_sim_counts"] ** 2 + short_data["std_sim_counts"] ** 2
        ),
    }


def pad_to_length(arr, length: int) -> np.ndarray:
    """Right-pads `arr` with zeros (or truncates it) to exactly `length`."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)))

    return df_per_subject, counts_dict


def build_commit_vs_full_param_df(
    task: str,
    results_df_full: pd.DataFrame,
    results_df_commit: pd.DataFrame,
    param_order: list[str],
) -> pd.DataFrame:
    """Tidy per-subject, per-parameter comparison of a full-sequence fit
    against a commit-likelihood fit for one model architecture (TASK code).

    Aligns the two fits on subject_ID (inner join -- subjects fitted under
    only one objective are dropped) and unpacks each subject's
    "fit_params_ga" vector into one row per parameter.

    Args:
        task: TASK code identifying the model architecture, e.g.
            "LBEXT-RPHCLUK". Stored verbatim in the output "task" column.
        results_df_full: Full-sequence fit results for `task` (one row per
            subject, with "subject_ID" and a "fit_params_ga" column ordered
            per `param_order`), as returned by `load_fitted_results(...,
            commit=False)`.
        results_df_commit: Commit-likelihood fit results for the same
            `task`, same format, as returned by `load_fitted_results(...,
            commit=True)`.
        param_order: Parameter names in the order `fit_params_ga` is
            packed (i.e. the fitting config's PARAM_ORDER).

    Returns:
        Long-format DataFrame with one row per (subject, parameter) and
        columns "task", "subject_ID", "param", "full_value",
        "commit_value", "diff" (commit_value - full_value), and
        "pct_diff_sym": the symmetric percent difference
        200 * diff / (|full_value| + |commit_value|), NaN when both values
        are 0. Several fitted parameters (e.g. subjective_cost) can be
        negative or straddle 0 across subjects, which makes a plain
        (commit - full) / full * 100 blow up or flip sign spuriously near
        full_value == 0; the symmetric form stays bounded to [-200, 200]
        and is safe to summarize/plot directly.
    """
    full_params = pd.DataFrame(
        results_df_full["fit_params_ga"].tolist(), columns=param_order
    )
    full_params["subject_ID"] = results_df_full["subject_ID"].values

    commit_params = pd.DataFrame(
        results_df_commit["fit_params_ga"].tolist(), columns=param_order
    )
    commit_params["subject_ID"] = results_df_commit["subject_ID"].values

    merged = full_params.merge(
        commit_params, on="subject_ID", suffixes=("_full", "_commit")
    )

    rows = []
    for param in param_order:
        full_vals = merged[f"{param}_full"]
        commit_vals = merged[f"{param}_commit"]
        diff = commit_vals - full_vals
        denom = full_vals.abs() + commit_vals.abs()
        pct_diff_sym = np.where(denom == 0, np.nan, 200 * diff / denom)
        rows.append(
            pd.DataFrame(
                {
                    "task": task,
                    "subject_ID": merged["subject_ID"],
                    "param": param,
                    "full_value": full_vals.values,
                    "commit_value": commit_vals.values,
                    "diff": diff.values,
                    "pct_diff_sym": pct_diff_sym,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize_commit_vs_full(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, param) summary of a commit-vs-full comparison.

    Args:
        long_df: Concatenation of one or more `build_commit_vs_full_param_df`
            outputs (e.g. across every shared model architecture).

    Returns:
        DataFrame with columns "task", "param", "n_subjects", "median_diff",
        "median_pct_diff_sym", "spearman_rho", "spearman_p" (agreement
        between full_value and commit_value across subjects), "wilcoxon_p"
        (paired signed-rank test on "diff", testing whether the
        commit-likelihood objective systematically shifts that parameter;
        NaN if every subject's diff is exactly 0), and "wilcoxon_p_fdr"
        (Benjamini-Hochberg correction applied jointly across every row of
        the output, i.e. across all task/param tests run in one call).
    """
    rows = []
    for (task, param), g in long_df.groupby(["task", "param"], sort=False):
        diff = g["diff"].to_numpy()
        try:
            _, w_p = wilcoxon(diff)
        except ValueError:
            # All diffs are exactly 0 (or too few subjects) -- wilcoxon
            # raises rather than returning a degenerate p-value.
            w_p = np.nan
        rho, rho_p = safe_spearman(g["full_value"], g["commit_value"])
        rows.append(
            {
                "task": task,
                "param": param,
                "n_subjects": len(g),
                "median_diff": np.median(diff),
                "median_pct_diff_sym": np.nanmedian(g["pct_diff_sym"]),
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "wilcoxon_p": w_p,
            }
        )
    summary = pd.DataFrame(rows)

    summary["wilcoxon_p_fdr"] = np.nan
    valid = summary["wilcoxon_p"].notna()
    if valid.sum() > 0:
        summary.loc[valid, "wilcoxon_p_fdr"] = multipletests(
            summary.loc[valid, "wilcoxon_p"], method="fdr_bh"
        )[1]
    return summary


def build_param_pair_comparison_df(
    label_a: str,
    results_df_a: pd.DataFrame,
    param_order_a: list[str],
    label_b: str,
    results_df_b: pd.DataFrame,
    param_order_b: list[str],
) -> pd.DataFrame:
    """Tidy per-subject, per-parameter comparison between two arbitrary
    fits that may be different model architectures (e.g. the best
    short-horizon model vs. the best long-horizon model), and therefore
    only partially overlapping in which parameters they include.

    Unlike `build_commit_vs_full_param_df` (which assumes both fits share
    one `param_order`, as is true for the same architecture fit under two
    objectives), this restricts to the parameters present *by name* in
    both `param_order_a` and `param_order_b` before comparing.

    Args:
        label_a: Name for the first fit (e.g. "short"), stored verbatim in
            the output "label_a" column.
        results_df_a: First fit's results (one row per subject, columns
            "subject_ID" and "fit_params_ga" ordered per `param_order_a`).
        param_order_a: Parameter names for `results_df_a`, in
            `fit_params_ga` order.
        label_b, results_df_b, param_order_b: Same, for the second fit.

    Returns:
        Long-format DataFrame with one row per (subject, shared parameter)
        and columns "param", "subject_ID", "label_a", "value_a",
        "label_b", "value_b", "diff" (value_b - value_a), and
        "pct_diff_sym" (symmetric percent difference -- see
        `build_commit_vs_full_param_df` for why this is used instead of a
        plain (b-a)/a*100). Empty if the two architectures share no
        parameter names (a warning is printed in that case).
    """
    shared_params = [p for p in param_order_a if p in param_order_b]
    if not shared_params:
        print(
            f"No shared parameters between {label_a} ({param_order_a}) "
            f"and {label_b} ({param_order_b})."
        )
        return pd.DataFrame(
            columns=[
                "param", "subject_ID", "label_a", "value_a",
                "label_b", "value_b", "diff", "pct_diff_sym",
            ]
        )

    df_a = pd.DataFrame(results_df_a["fit_params_ga"].tolist(), columns=param_order_a)
    df_a["subject_ID"] = results_df_a["subject_ID"].values

    df_b = pd.DataFrame(results_df_b["fit_params_ga"].tolist(), columns=param_order_b)
    df_b["subject_ID"] = results_df_b["subject_ID"].values

    merged = df_a[shared_params + ["subject_ID"]].merge(
        df_b[shared_params + ["subject_ID"]], on="subject_ID", suffixes=("_a", "_b")
    )

    rows = []
    for param in shared_params:
        va = merged[f"{param}_a"]
        vb = merged[f"{param}_b"]
        diff = vb - va
        denom = va.abs() + vb.abs()
        pct_diff_sym = np.where(denom == 0, np.nan, 200 * diff / denom)
        rows.append(
            pd.DataFrame(
                {
                    "param": param,
                    "subject_ID": merged["subject_ID"],
                    "label_a": label_a,
                    "value_a": va.values,
                    "label_b": label_b,
                    "value_b": vb.values,
                    "diff": diff.values,
                    "pct_diff_sym": pct_diff_sym,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize_param_pair_correlations(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(label_a, label_b, parameter) Pearson and Spearman correlation
    between `value_a` and `value_b` in a `build_param_pair_comparison_df`
    output.

    Args:
        long_df: Output of `build_param_pair_comparison_df` -- either a
            single label_a/label_b pair, or several concatenated together
            (grouping is done on every present "label_a"/"label_b"/"param"
            combination).

    Returns:
        DataFrame with one row per (label_a, label_b, param) and columns
        "n_subjects", "pearson_r", "pearson_p", "spearman_rho",
        "spearman_p", "median_diff", "median_pct_diff_sym". Pearson/
        Spearman are NaN if either side is constant across subjects (no
        variance to correlate).
    """
    group_cols = [c for c in ["label_a", "label_b", "param"] if c in long_df.columns]
    rows = []
    for keys, g in long_df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        va, vb = g["value_a"], g["value_b"]
        if va.nunique() < 2 or vb.nunique() < 2:
            pearson_r, pearson_p = np.nan, np.nan
        else:
            pearson_r, pearson_p = pearsonr(va, vb)
        spearman_rho, spearman_p = safe_spearman(va, vb)
        row.update(
            {
                "n_subjects": len(g),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "median_diff": np.median(g["diff"]),
                "median_pct_diff_sym": np.nanmedian(g["pct_diff_sym"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
