from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families import Binomial
from statsmodels.api import GLM, add_constant
import numpy as np
from src.utils import get_from_mat, nannormalise, lagmatrix
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# from pymer4.models import lmer
# from rpy2.robjects import pandas2ri, conversion, default_converter, r
# from rpy2.robjects.conversion import localconverter, py2rpy
import importlib.util

# These are the specific imports needed for the conversion context
# from rpy2.robjects import pandas2ri, conversion, default_converter
import scipy.stats as stats


def calculate_total_last_evidence(
    obj: dict | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-draw evidence-based predictors from raw trial data.

    Args:
        obj: Either a MATLAB-struct-style dict (e.g. from loadmat, with
            "trial", "termination", "totEvLeft", "totEvRight" fields
            accessed via `get_from_mat`), or a pandas Series/DataFrame
            row with the same fields as columns.

    Returns:
        Tuple of (deltaev, termination, totevminus, trial), each a 1-D
        np.ndarray aligned with the input draws. `totevminus` is the
        previous draw's total evidence (NaN on each game's first draw)
        and `deltaev` is the change in total evidence from that draw.
    """
    # Decide how to access fields based on what obj looks like
    if isinstance(obj, dict):
        # e.g. matlab struct dict from loadmat, or your pmat wrapper
        def get(name):
            return get_from_mat(obj, name)

    else:
        # assume pandas Series / row / DataFrame row-like
        def get(name):
            return np.array(obj[name], dtype=float)

    trial = np.array(get("trial").astype(float))
    termination = np.array(get("termination").astype(float))
    totevleft = np.array(get("totEvLeft").astype(float))
    totevright = np.array(get("totEvRight").astype(float))

    totev = np.abs(totevleft - totevright)

    totevminus = lagmatrix(totev, 1)
    firstdraw_idx = np.where(trial == 1)[0]
    totevminus[firstdraw_idx] = np.nan
    deltaev = totev - totevminus

    return deltaev, termination, totevminus, trial


# def fit_glme(pmat_df, subject_df, user_col="userID", outcome="decide"):
#     """
#     Fit a binomial GLMM like MATLAB's fitglme using R's lme4::glmer.
#     Random intercepts + random slopes per user.

#     Returns:
#         coefs: pandas DataFrame of fixed effects
#         glmm_data: merged DataFrame
#     """
#     # -----------------------------
#     # 1. Merge Data
#     # -----------------------------
#     glmm_data = pd.merge(pmat_df, subject_df, on=user_col, how="inner")

#     # -----------------------------
#     # 2. Type conversions
#     # -----------------------------
#     glmm_data[outcome] = glmm_data[outcome].astype(int)
#     glmm_data[user_col] = glmm_data[user_col].astype(str)  # keep as string for R factor

#     numeric_cols = ["totevminus", "deltaev", "trial", "termination", "FA2"]
#     for col in numeric_cols:
#         if col in glmm_data.columns:
#             glmm_data[col] = pd.to_numeric(glmm_data[col], errors="coerce")

#     # -----------------------------
#     # 3. Define formula (MATLAB style)
#     # -----------------------------
#     formula = (
#         f"{outcome} ~ 1 + totevminus + deltaev + trial + termination + "
#         "totevminus:termination + deltaev:termination + "
#         "FA2 + FA2:totevminus + FA2:deltaev + FA2:trial + FA2:termination + "
#         "FA2:totevminus:termination + FA2:deltaev:termination + "
#         f"(1 + totevminus + deltaev + trial + termination + totevminus:termination + deltaev:termination | {user_col})"
#     )

#     # -----------------------------
#     # 4. Convert DataFrame to R and fit GLMM
#     # -----------------------------
#     with localconverter(default_converter + pandas2ri.converter):
#         r_df = py2rpy(glmm_data)  # explicit conversion
#         r.assign("df", r_df)  # safe assignment

#     # Load lme4 in R
#     r("library(lme4)")

#     # Fit GLMM
#     r(f'model <- glmer({formula}, data=df, family=binomial(link="logit"))')

#     # Extract fixed effects as pandas DataFrame
#     with localconverter(default_converter + pandas2ri.converter):
#         coefs_df = r("as.data.frame(summary(model)$coefficients)")

#     return coefs_df, glmm_data


def fit_glmm(
    pmat_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    user_col: str = "userID",
    outcome: str = "decide",
    n_regressors: int = 13,
    predictors: list[str] | None = None,
) -> tuple:
    """
    Fit a binomial GLM approximating a GLMM using subject fixed effects.

    Parameters
    ----------
    pmat_df : pandas.DataFrame
        Trial-level predictor data.
    subject_df : pandas.DataFrame
        Subject-level variables (e.g. questionnaire scores).
    user_col : str, default="userID"
        Column identifying subjects.
    outcome : str, default="decide"
        Binary response variable.
    n_regressors : int, default=13
        How many of [totevminus, deltaev, trial, termination,
        totevminus:termination, deltaev:termination, FA2,
        FA2:totevminus, FA2:deltaev, FA2:trial, FA2:termination,
        FA2:totevminus:termination, FA2:deltaev:termination] to include,
        in that order (subject fixed effects are always included). Use 4
        to fit on main effects only -- this drops the termination
        interactions and all FA2 (OC factor) terms.
    predictors : list[str] or None, default=None
        Explicit predictor list, overriding n_regressors. Needed for a
        single-horizon fit, where termination is constant and therefore
        collinear with the intercept, so it and its interactions have to be
        dropped rather than truncated away by n_regressors (they are not a
        prefix of the default list).

    Returns
    -------
    results : statsmodels result object
        Fitted GLM results.
    estimates : dict
        Dictionary containing grouped coefficient estimates and SEs, plus
        in-sample "accuracy" and McFadden's "pseudo_r2".
    data_used : pandas.DataFrame
        Data used for model fitting.
    """

    # -----------------------------
    # Copy data to avoid mutation
    # -----------------------------
    pmat_df = pmat_df.copy()
    subject_df = subject_df.copy()

    # -----------------------------
    # Ensure consistent user IDs
    # -----------------------------
    pmat_df[user_col] = pmat_df[user_col].astype(str)
    subject_df[user_col] = subject_df[user_col].astype(str)

    # -----------------------------
    # Merge subject-level variables
    # -----------------------------
    glmm_data = pd.merge(pmat_df, subject_df, on=user_col, how="inner")

    if outcome not in glmm_data.columns:
        raise ValueError(f"{outcome} not found in dataframe")

    glmm_data[outcome] = glmm_data[outcome].astype(int)

    # -----------------------------
    # Create subject fixed effects
    # -----------------------------
    glmm_data = pd.get_dummies(glmm_data, columns=[user_col], drop_first=True)

    # -----------------------------
    # Define predictors
    # -----------------------------
    if predictors is None:
        predictors = [
            "totevminus",
            "deltaev",
            "trial",
            "termination",
            "totevminus:termination",
            "deltaev:termination",
            "FA2",
            "FA2:totevminus",
            "FA2:deltaev",
            "FA2:trial",
            "FA2:termination",
            "FA2:totevminus:termination",
            "FA2:deltaev:termination",
        ][:n_regressors]
    else:
        predictors = list(predictors)

    # add subject dummies
    user_dummies = [c for c in glmm_data.columns if c.startswith(f"{user_col}_")]
    predictors += user_dummies

    formula = f"{outcome} ~ " + " + ".join(predictors)

    # -----------------------------
    # Fit GLM
    # -----------------------------
    model = smf.glm(formula=formula, data=glmm_data, family=sm.families.Binomial())

    results = model.fit()

    # -----------------------------
    # Extract coefficients
    # -----------------------------
    params = results.params
    ses = results.bse

    # keep only what was actually fitted, so a single-horizon fit (no
    # termination terms) returns the four it has rather than raising
    group1 = [p for p in ["totevminus", "deltaev", "trial", "termination"]
              if p in params.index]
    group2 = [p for p in ["FA2:totevminus", "FA2:deltaev", "FA2:trial",
                          "FA2:termination"] if p in params.index]

    # In-sample accuracy and McFadden's pseudo-R^2
    # (use results.model.endog, not glmm_data[outcome] -- statsmodels drops
    # rows with NaNs in any formula term, so the two can differ in length)
    y = results.model.endog
    y_pred_prob = results.predict()
    accuracy = np.mean((y_pred_prob > 0.5).astype(int) == y)
    pseudo_r2 = results.pseudo_rsquared(kind="mcf")

    estimates = {
        "group1_estimates": params.loc[group1],
        "group1_se": ses.loc[group1],
        "group2_estimates": params.loc[group2] if group2 else None,
        "group2_se": ses.loc[group2] if group2 else None,
        "accuracy": accuracy,
        "pseudo_r2": pseudo_r2,
    }

    return results, estimates, glmm_data


def fit_glm_before_saving_mean_and_std_for_predictors(
    data: list | pd.DataFrame, source: str = "mat", ocir_all: pd.DataFrame | None = None
) -> tuple[list[dict | None], list]:
    """Fit a per-subject GLM predicting "decide" without saving z-score mu/sigma.

    Earlier variant of `fit_glm` -- kept for backward compatibility with
    notebooks that consume its output (no "mu"/"sigma"/diagnostics keys).

    Args:
        data: Original MATLAB-style data list (source="mat") or combined_df
            pandas DataFrame (source="df").
        source: "mat" -> `data` is the original MATLAB-style data list.
            "df"  -> `data` is a combined_df pandas DataFrame.
        ocir_all: DataFrame with a "userID" column aligned to `data` by
            index; only used when source="mat".

    Returns:
        Tuple of (betas_all, id_all):
            betas_all: One entry per subject, either None (if
                "distance2choice" was missing) or a dict with keys "id",
                "pdecide_beta", "decide", "pmat", "pmat_z".
            id_all: Subject IDs, one per entry in `betas_all` (in the
                same order, including for `None` entries).
    """

    betas_all = []
    id_all = []

    # ------------------------------------------------------
    # iterate depending on data type
    if source == "mat":
        iterator = range(len(data))
    else:
        iterator = data.iterrows()

    for ii in iterator:

        # ------------------------------------------------------
        # extract subject data depending on format
        if source == "mat":
            idx = ii
            dat = data[idx]
            uid = ocir_all["userID"].iloc[idx]
            pmat = {"mat": dat.beh.dat, "names": list(dat.beh.descr)}

            try:
                dist2ch = get_from_mat(pmat, "distance2choice")
            except KeyError:
                betas_all.append(None)
                id_all.append(uid)
                continue

            deltaev, termination, totevminus, trial = calculate_total_last_evidence(
                pmat
            )

        else:  # dataframe
            _, row = ii
            uid = row["userID"]

            dist2ch = np.array(row["distance2choice"], dtype=float)
            deltaev, termination, totevminus, trial = calculate_total_last_evidence(row)

        # ------------------------------------------------------
        # build decision variable
        decide = np.full_like(dist2ch, np.nan, dtype=np.float64)
        decide[dist2ch > 0] = 1

        idx = np.where(dist2ch == 1)[0]
        if len(idx) > 0 and np.max(idx) + 1 < len(decide):
            decide[idx + 1] = 0

        chidx = np.where(~np.isnan(decide))[0]

        # ------------------------------------------------------
        # flip coding: 0=continue, 1=decide
        decide = decide - 1
        decide[decide == -1] = 1

        # predictors
        # termination is coded {1, 2} -- center it before forming the interaction
        # products so they aren't collinear with the main effects after z-scoring.
        # termination_centered = termination[chidx] - np.nanmean(termination[chidx])
        allvar = np.column_stack(
            [
                totevminus[chidx],
                deltaev[chidx],
                trial[chidx],
                termination[chidx],
                totevminus[chidx] * termination[chidx],
                trial[chidx] * termination[chidx],
            ]
        )

        regs_z = nannormalise(allvar)
        y = decide[chidx]

        valid_mask = (
            (~np.isnan(regs_z).any(axis=1))
            & (~np.isnan(y))
            & (np.isfinite(regs_z).all(axis=1))
            & (np.isfinite(y))
        )

        X_clean = regs_z[valid_mask]
        y_clean = y[valid_mask]

        # ------------------------------------------------------
        # fit GLM
        try:
            model = GLM(
                y_clean, add_constant(X_clean, prepend=False), family=Binomial()
            )
            result = model.fit()
            pdecide_beta = result.params
        except Exception as e:
            print(f"GLM failed for subject {uid}: {e}")
            pdecide_beta = np.full(X_clean.shape[1] + 1, np.nan)

        # ------------------------------------------------------
        betas_all.append(
            {
                "id": uid,
                "pdecide_beta": pdecide_beta,
                "decide": decide[chidx],
                "pmat": allvar,
                "pmat_z": regs_z,
            }
        )

        id_all.append(uid)

    return betas_all, id_all


# I want to define the filter_list hardcoded with all subjects IDs
FILTERED_SUBJECTS = [
    17,
    50,
    93,
    58,
    45,
    38,
    48,
    56,
    11,
    74,
    100,
    101,
    70,
    23,
    5,
    67,
    68,
    87,
    19,
    59,
    1,
    26,
    88,
    79,
    69,
    15,
    71,
    94,
    102,
    85,
    34,
    28,
    66,
    31,
    14,
    65,
    95,
    81,
    27,
    83,
    46,
    43,
    24,
    42,
    90,
    63,
    104,
    16,
    8,
    49,
    33,
    39,
    52,
    37,
    82,
    13,
    77,
    36,
    76,
    89,
    18,
    91,
    10,
    73,
    98,
    92,
    32,
    12,
    35,
    105,
    55,
    7,
    86,
    78,
    2,
    54,
    40,
    6,
    20,
    3,
    62,
    75,
    60,
    22,
    80,
    30,
    97,
    64,
    44,
    57,
    9,
    72,
    99,
    84,
    29,
    25,
    103,
    61,
    51,
    21,
    4,
    53,
    41,
    96,
    47,
]


def fit_glm_specific_subjects(
    data: list | pd.DataFrame,
    source: str = "mat",
    ocir_all: pd.DataFrame | None = None,
    filtered_subjects: list = FILTERED_SUBJECTS,
) -> tuple[list[dict | None], list]:
    """Like `fit_glm`, but skips any subject not in `filtered_subjects`.

    Args:
        data: Original MATLAB-style data list (source="mat") or combined_df
            pandas DataFrame (source="df").
        source: "mat" -> `data` is the original MATLAB-style data list.
            "df"  -> `data` is a combined_df pandas DataFrame.
        ocir_all: DataFrame with a "userID" column aligned to `data` by
            index; only used when source="mat".
        filtered_subjects: Subject IDs to include; all others are skipped
            entirely (no entry is added to the outputs for them).
            Defaults to the module-level `FILTERED_SUBJECTS` list.

    Returns:
        Tuple of (betas_all, id_all), one entry per included subject:
            betas_all: dict per subject (or None if "distance2choice" was
                missing) with keys "id", "pdecide_beta", "pvals", "aic",
                "bic", "llf", "pseudo_r2", "accuracy", "n_trials",
                "decide", "pmat", "pmat_z", "mu", "sigma".
            id_all: Subject IDs, one per entry in `betas_all`.
    """

    betas_all = []
    id_all = []

    if source == "mat":
        iterator = range(len(data))
    else:
        iterator = data.iterrows()

    for ii in iterator:

        # ------------------------------------------------------
        # extract subject data
        if source == "mat":

            idx = ii
            dat = data[idx]
            uid = ocir_all["userID"].iloc[idx]
            if uid in filtered_subjects:
                pmat = {"mat": dat.beh.dat, "names": list(dat.beh.descr)}
            else:
                continue

            try:
                dist2ch = get_from_mat(pmat, "distance2choice")
            except KeyError:
                betas_all.append(None)
                id_all.append(uid)
                continue

            deltaev, termination, totevminus, trial = calculate_total_last_evidence(
                pmat
            )

        else:
            _, row = ii
            uid = row["userID"]

            if uid in filtered_subjects:
                dist2ch = np.array(row["distance2choice"], dtype=float)
                deltaev, termination, totevminus, trial = calculate_total_last_evidence(
                    row
                )
            else:
                continue

        # ------------------------------------------------------
        # build decision variable
        decide = np.full_like(dist2ch, np.nan, dtype=np.float64)
        decide[dist2ch > 0] = 1

        idx = np.where(dist2ch == 1)[0]
        if len(idx) > 0 and np.max(idx) + 1 < len(decide):
            decide[idx + 1] = 0

        chidx = np.where(~np.isnan(decide))[0]

        # flip coding
        decide = decide - 1
        decide[decide == -1] = 1

        # ------------------------------------------------------
        # predictors
        # termination is coded {1, 2} -- center it before forming the interaction
        # products so they aren't collinear with the main effects after z-scoring.
        # termination_centered = termination[chidx] - np.nanmean(termination[chidx])
        allvar = np.column_stack(
            [
                totevminus[chidx],
                deltaev[chidx],
                trial[chidx],
                termination[chidx],
                totevminus[chidx] * termination[chidx],
                trial[chidx] * termination[chidx],
            ]
        )

        # ---- compute z-score parameters ----
        mu = np.nanmean(allvar, axis=0)
        sigma = np.nanstd(allvar, axis=0)

        # avoid divide-by-zero
        sigma[sigma == 0] = 1

        regs_z = (allvar - mu) / sigma

        y = decide[chidx]

        valid_mask = (
            (~np.isnan(regs_z).any(axis=1))
            & (~np.isnan(y))
            & (np.isfinite(regs_z).all(axis=1))
            & (np.isfinite(y))
        )

        X_clean = regs_z[valid_mask]
        y_clean = y[valid_mask]

        # ------------------------------------------------------
        # fit GLM
        try:
            model = GLM(
                y_clean, add_constant(X_clean, prepend=False), family=Binomial()
            )
            result = model.fit()
            pdecide_beta = result.params

            # Model diagnostics
            aic = result.aic
            bic = result.bic_llf
            llf = result.llf

            # Pseudo-R² (McFadden's)
            null_model = GLM(
                y_clean,
                add_constant(np.ones((len(y_clean), 1)), prepend=False),
                family=Binomial(),
            )
            null_result = null_model.fit(disp=0)
            pseudo_r2 = result.pseudo_rsquared(kind="mcf")

            # Prediction accuracy
            y_pred_prob = result.predict()
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = np.mean(y_pred == y_clean)

            # p-values
            pvals = result.pvalues

        except Exception as e:
            print(f"GLM failed for subject {uid}: {e}")
            pdecide_beta = np.full(X_clean.shape[1] + 1, np.nan)

        # ------------------------------------------------------
        # Store results
        betas_all.append(
            {
                "id": uid,
                "pdecide_beta": pdecide_beta,
                "pvals": pvals,
                "aic": aic,
                "bic": bic,
                "llf": llf,
                "pseudo_r2": pseudo_r2,
                "accuracy": accuracy,
                "n_trials": len(y_clean),
                "decide": decide[chidx],
                "pmat": allvar,
                "pmat_z": regs_z,
                "mu": mu,
                "sigma": sigma,
            }
        )
        id_all.append(uid)

    return betas_all, id_all


def fit_glm(
    data: list | pd.DataFrame, source: str = "mat", ocir_all: pd.DataFrame | None = None
) -> tuple[list[dict | None], list]:
    """Fit a per-subject binomial GLM predicting "decide" from evidence predictors.

    Args:
        data: Original MATLAB-style data list (source="mat") or combined_df
            pandas DataFrame (source="df").
        source: "mat" -> `data` is the original MATLAB-style data list.
            "df"  -> `data` is a combined_df pandas DataFrame.
        ocir_all: DataFrame with a "userID" column aligned to `data` by
            index; only used when source="mat".

    Returns:
        Tuple of (betas_all, id_all):
            betas_all: One entry per subject, either None (if
                "distance2choice" was missing) or a dict with keys "id",
                "pdecide_beta", "pvals", "aic", "bic", "llf",
                "pseudo_r2", "accuracy", "n_trials", "decide", "pmat",
                "pmat_z", "mu", "sigma".
            id_all: Subject IDs, one per entry in `betas_all`.
    """

    betas_all = []
    id_all = []

    if source == "mat":
        iterator = range(len(data))
    else:
        iterator = data.iterrows()

    for ii in iterator:

        # ------------------------------------------------------
        # extract subject data
        if source == "mat":

            idx = ii
            dat = data[idx]
            uid = ocir_all["userID"].iloc[idx]
            pmat = {"mat": dat.beh.dat, "names": list(dat.beh.descr)}

            try:
                dist2ch = get_from_mat(pmat, "distance2choice")
            except KeyError:
                betas_all.append(None)
                id_all.append(uid)
                continue

            deltaev, termination, totevminus, trial = calculate_total_last_evidence(
                pmat
            )

        else:
            _, row = ii
            uid = row["userID"]

            dist2ch = np.array(row["distance2choice"], dtype=float)
            deltaev, termination, totevminus, trial = calculate_total_last_evidence(row)

        # ------------------------------------------------------
        # build decision variable
        decide = np.full_like(dist2ch, np.nan, dtype=np.float64)
        decide[dist2ch > 0] = 1

        idx = np.where(dist2ch == 1)[0]
        if len(idx) > 0 and np.max(idx) + 1 < len(decide):
            decide[idx + 1] = 0

        chidx = np.where(~np.isnan(decide))[0]

        # flip coding
        decide = decide - 1
        decide[decide == -1] = 1

        # ------------------------------------------------------
        # predictors
        # termination is coded {1, 2} -- center it before forming the interaction
        # products so they aren't collinear with the main effects after z-scoring.
        # termination_centered = termination[chidx] - np.nanmean(termination[chidx])
        allvar = np.column_stack(
            [
                totevminus[chidx],
                deltaev[chidx],
                trial[chidx],
                termination[chidx],
                totevminus[chidx] * termination[chidx],
                trial[chidx] * termination[chidx],
            ]
        )

        # ---- compute z-score parameters ----
        mu = np.nanmean(allvar, axis=0)
        sigma = np.nanstd(allvar, axis=0)

        # avoid divide-by-zero
        sigma[sigma == 0] = 1

        regs_z = (allvar - mu) / sigma

        y = decide[chidx]

        valid_mask = (
            (~np.isnan(regs_z).any(axis=1))
            & (~np.isnan(y))
            & (np.isfinite(regs_z).all(axis=1))
            & (np.isfinite(y))
        )

        X_clean = regs_z[valid_mask]
        y_clean = y[valid_mask]

        # ------------------------------------------------------
        # fit GLM
        try:
            model = GLM(
                y_clean, add_constant(X_clean, prepend=False), family=Binomial()
            )
            result = model.fit()
            pdecide_beta = result.params

            # Model diagnostics
            aic = result.aic
            bic = result.bic_llf
            llf = result.llf

            # Pseudo-R² (McFadden's)
            null_model = GLM(
                y_clean,
                add_constant(np.ones((len(y_clean), 1)), prepend=False),
                family=Binomial(),
            )
            null_result = null_model.fit(disp=0)
            pseudo_r2 = result.pseudo_rsquared(kind="mcf")

            # Prediction accuracy
            y_pred_prob = result.predict()
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = np.mean(y_pred == y_clean)

            # p-values
            pvals = result.pvalues

        except Exception as e:
            print(f"GLM failed for subject {uid}: {e}")
            pdecide_beta = np.full(X_clean.shape[1] + 1, np.nan)

        # ------------------------------------------------------
        # Store results
        betas_all.append(
            {
                "id": uid,
                "pdecide_beta": pdecide_beta,
                "pvals": pvals,
                "aic": aic,
                "bic": bic,
                "llf": llf,
                "pseudo_r2": pseudo_r2,
                "accuracy": accuracy,
                "n_trials": len(y_clean),
                "decide": decide[chidx],
                "pmat": allvar,
                "pmat_z": regs_z,
                "mu": mu,
                "sigma": sigma,
            }
        )
        id_all.append(uid)

    return betas_all, id_all


def return_number_of_draws(
    processed_human_data: pd.DataFrame, subject_id: int
) -> list[int]:
    """Count the number of draws taken in each game for a single subject.

    Groups `processed_human_data` by ("block", "game") and, for each game,
    counts the draws up to and including the decision draw (or the full
    game length if no decision was made).

    Args:
        processed_human_data: Trial-level data for one subject, with
            "block", "game", and "choiceTrial" columns.
        subject_id: Unused by this function; kept for interface
            compatibility with callers that pass a subject identifier.

    Returns:
        List of per-game draw counts, one entry per ("block", "game")
        group, in groupby iteration order.
    """

    num_draws_list = []

    # Create the groupby object.
    game_groups = processed_human_data.groupby(["block", "game"])

    for name, game_data in game_groups:
        if game_data.empty:
            continue

        decision_index = game_data["choiceTrial"].first_valid_index()

        if decision_index:
            decision_pos = game_data.index.get_loc(decision_index)
            num_draws = decision_pos + 1
        else:
            # If no decision was made, the number of draws is the total rows for that game.
            num_draws = len(game_data)

        num_draws_list.append(int(num_draws))

    return num_draws_list


def return_number_of_valid_draws(
    processed_human_data: pd.DataFrame, keep_mask: np.ndarray
) -> list[int]:
    """
    Like `return_number_of_draws`, but counts only the draws marked True in
    `keep_mask` within each game's truncated window, instead of every raw
    draw. `keep_mask` should mark the rows that survived `valid_mask` in
    `fit_glm_separate_for_human_data` (i.e. the rows actually present in
    `y_pred_prob`). Slicing `y_pred_prob` with lengths from this function
    keeps each game's chunk aligned with the draws it was actually computed
    from, instead of with the raw (pre-filtering) draw count.
    """
    filtered_lengths = []
    offset = 0
    game_groups = processed_human_data.groupby(["block", "game"])

    for name, game_data in game_groups:
        if game_data.empty:
            continue

        game_size = len(game_data)
        decision_index = game_data["choiceTrial"].first_valid_index()

        if decision_index:
            decision_pos = game_data.index.get_loc(decision_index)
            num_draws = decision_pos + 1
        else:
            num_draws = game_size

        filtered_lengths.append(int(keep_mask[offset : offset + num_draws].sum()))
        offset += game_size

    return filtered_lengths


def compute_full_per_draw_probabilities(
    processed_human_data: pd.DataFrame,
    games_lengths: list[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    pdecide_beta: np.ndarray,
) -> np.ndarray:
    """
    Computes a GLM-predicted decide-probability for every draw of every game,
    aligned 1:1 with `games_lengths`.

    `fit_glm_separate_for_human_data` fits/predicts only on a filtered subset
    of rows (chidx/valid_mask), so its `y_pred_prob` has fewer entries than
    `sum(games_lengths)` and cannot be sliced by `games_lengths` directly.
    This recomputes the same features for every draw (using the already-fit
    mu/sigma/pdecide_beta, no refitting) and truncates each game's draws at
    the same point `return_number_of_draws` does, producing a probability
    array of length `sum(games_lengths)`.

    Args:
        processed_human_data: Trial-level data for one subject, with
            "block", "game" columns plus the fields consumed by
            `calculate_total_last_evidence`.
        games_lengths: Per-game draw counts to truncate each game to
            (typically from `return_number_of_draws`).
        mu: Per-predictor mean used to z-score features (from the fit that
            produced `pdecide_beta`).
        sigma: Per-predictor standard deviation used to z-score features.
        pdecide_beta: Fitted GLM coefficients, intercept last.

    Returns:
        1-D np.ndarray of decide-probabilities, length `sum(games_lengths)`,
        with each game's first draw forced to 0.0.
    """
    deltaev, termination, totevminus, trial = calculate_total_last_evidence(
        processed_human_data
    )

    allvar = np.column_stack(
        [
            totevminus,
            deltaev,
            trial,
            termination,
            totevminus * termination,
            trial * termination,
        ]
    )

    regs_z = (allvar - mu) / sigma
    logit = regs_z @ pdecide_beta[:-1] + pdecide_beta[-1]
    prob_full = 1 / (1 + np.exp(-logit))

    game_sizes = processed_human_data.groupby(["block", "game"]).size().tolist()

    final_probs = []
    idx = 0
    for game_size, game_length in zip(game_sizes, games_lengths):
        game_probs = prob_full[idx : idx + game_size]
        idx += game_size

        truncated = game_probs[:game_length].copy()
        # First draw of a game has no prior evidence to decide on.
        truncated[0] = 0.0
        final_probs.append(truncated)

    return np.concatenate(final_probs)


def fit_glm_separate_for_human_data(
    data: pd.DataFrame, ocir_all: pd.DataFrame | None = None, n_regressors: int = 6
) -> tuple[list[dict], list]:
    """Fit a per-subject GLM on human data, tracking per-game draw alignment.

    Args:
        data: combined_df-style pandas DataFrame with one row per subject,
            a "userID" column, and a "data" column holding each subject's
            nested trial-level DataFrame (with "distance2choice", "block",
            "game", "choiceTrial", etc.).
        ocir_all: Unused by this function; kept for signature parity with
            the other `fit_glm*` variants.
        n_regressors: How many columns of [totevminus, deltaev, trial,
            termination, totevminus*termination, trial*termination] to
            include, in that order. Use 4 to fit on main effects only
            (drops the two termination-interaction terms).

    Returns:
        Tuple of (betas_all, id_all):
            betas_all: One dict per subject with keys "id", "pdecide_beta",
                "pvals", "aic", "bic", "llf", "pseudo_r2", "accuracy",
                "n_trials", "decide", "y_pred_prob", "pmat", "pmat_z",
                "mu", "sigma", "games_lengths", "games_lengths_valid".
            id_all: Subject IDs, one per entry in `betas_all`.
    """
    betas_all = []
    id_all = []

    for ii in data.iterrows():
        # ------------------------------------------------------
        # extract subject data
        _, row = ii
        uid = row["userID"]
        games_lengths = []

        dist2ch = np.array(row["data"]["distance2choice"], dtype=float)
        deltaev, termination, totevminus, trial = calculate_total_last_evidence(
            row["data"]
        )

        games_lengths = return_number_of_draws(row["data"], uid)

        # ------------------------------------------------------
        # build decision variable
        decide = np.full_like(dist2ch, np.nan, dtype=np.float64)
        decide[dist2ch > 0] = 1

        idx = np.where(dist2ch == 1)[0]
        if len(idx) > 0 and np.max(idx) + 1 < len(decide):
            decide[idx + 1] = 0

        chidx = np.where(~np.isnan(decide))[0]

        # flip coding
        decide = decide - 1
        decide[decide == -1] = 1

        # ------------------------------------------------------
        # predictors
        # termination is coded {1, 2} -- center it before forming the interaction
        # products so they aren't collinear with the main effects after z-scoring.
        # termination_centered = termination[chidx] - np.nanmean(termination[chidx])
        allvar = np.column_stack(
            [
                totevminus[chidx],
                deltaev[chidx],
                trial[chidx],
                termination[chidx],
                totevminus[chidx] * termination[chidx],
                trial[chidx] * termination[chidx],
            ]
        )[:, :n_regressors]

        # ---- compute z-score parameters ----
        mu = np.nanmean(allvar, axis=0)
        sigma = np.nanstd(allvar, axis=0)

        # avoid divide-by-zero
        sigma[sigma == 0] = 1

        regs_z = (allvar - mu) / sigma

        y = decide[chidx]

        valid_mask = (
            (~np.isnan(regs_z).any(axis=1))
            & (~np.isnan(y))
            & (np.isfinite(regs_z).all(axis=1))
            & (np.isfinite(y))
        )

        X_clean = regs_z[valid_mask]
        y_clean = y[valid_mask]

        # games_lengths (raw per-game draw count) is kept as-is for callers
        # like compute_full_per_draw_probabilities, which recompute a
        # probability for every raw draw. y_pred_prob, however, only has one
        # entry per row that survived valid_mask (e.g. draw 1 of every game
        # is dropped because totevminus/deltaev are NaN there), so slicing
        # it with the raw per-game count would desync game boundaries.
        # games_lengths_valid gives the per-game count that actually matches
        # y_pred_prob/decide[chidx][valid_mask].
        keep_mask = np.zeros(len(decide), dtype=bool)
        keep_mask[chidx[valid_mask]] = True
        games_lengths_valid = return_number_of_valid_draws(row["data"], keep_mask)

        # ------------------------------------------------------
        # fit GLM
        try:
            model = GLM(
                y_clean, add_constant(X_clean, prepend=False), family=Binomial()
            )
            result = model.fit()
            pdecide_beta = result.params

            # Model diagnostics
            aic = result.aic
            bic = result.bic_llf
            llf = result.llf

            # Pseudo-R² (McFadden's)
            null_model = GLM(
                y_clean,
                add_constant(np.ones((len(y_clean), 1)), prepend=False),
                family=Binomial(),
            )
            null_result = null_model.fit(disp=0)
            pseudo_r2 = result.pseudo_rsquared(kind="mcf")

            # Prediction accuracy
            y_pred_prob = result.predict()
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = np.mean(y_pred == y_clean)

            # p-values
            pvals = result.pvalues

        except Exception as e:
            print(f"GLM failed for subject {uid}: {e}")
            pdecide_beta = np.full(X_clean.shape[1] + 1, np.nan)

        # ------------------------------------------------------
        # Store results
        betas_all.append(
            {
                "id": uid,
                "pdecide_beta": pdecide_beta,
                "pvals": pvals,
                "aic": aic,
                "bic": bic,
                "llf": llf,
                "pseudo_r2": pseudo_r2,
                "accuracy": accuracy,
                "n_trials": len(y_clean),
                "decide": decide[chidx],
                "y_pred_prob": y_pred_prob,
                "pmat": allvar,
                "pmat_z": regs_z,
                "mu": mu,
                "sigma": sigma,
                "games_lengths": games_lengths,
                "games_lengths_valid": games_lengths_valid,
            }
        )
        id_all.append(uid)

    return betas_all, id_all


def predict_decision(
    beta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    totevminus: float,
    deltaev: float,
    trial: float,
    termination: float,
) -> float:
    """Predict the "decide" probability for a single draw from fitted GLM coefficients.

    Builds the [totevminus, deltaev, trial, termination,
    totevminus*termination, trial*termination] feature vector, z-scores it
    with `mu`/`sigma`, and applies the logistic function with `beta`
    (intercept last). Scalar inputs only.

    Args:
        beta: Fitted GLM coefficients, length 7 (6 predictors + intercept),
            intercept last.
        mu: Per-predictor mean used to z-score features, length 6.
        sigma: Per-predictor standard deviation used to z-score features,
            length 6.
        totevminus: Previous draw's total evidence.
        deltaev: Change in total evidence from the previous draw.
        trial: Draw/trial index within the game.
        termination: Termination-condition code (e.g. 1 or 2).

    Returns:
        Predicted probability of deciding on this draw.
    """
    X = np.array(
        [
            totevminus,
            deltaev,
            trial,
            termination,
            totevminus * termination,
            trial * termination,
        ]
    ).T

    Xz = (X - mu) / sigma

    X_full = np.hstack([Xz, 1])  # intercept
    logit = X_full @ beta

    return 1 / (1 + np.exp(-logit))


GLMM_PREDICTOR_LABEL_MAP = {
    "totevminus": r"$ES_{t-1}$",
    "deltaev": r"$\Delta ES_t$",
    "trial": r"trial",
    "termination": r"termination",
    "FA2:totevminus": r"FA2 $\times$ $ES_{t-1}$",
    "FA2:deltaev": r"FA2 $\times$ $\Delta ES_t$",
    "FA2:trial": r"FA2 $\times$ trial",
    "FA2:termination": r"FA2 $\times$ termination",
}


def export_glmm_latex_table(
    glmm_summary_df: pd.DataFrame,
    output_path: str,
    label_map: dict | None = None,
    caption: str = "GLMM fixed-effect estimates (ensemble mean across simulated instances).",
    label: str = "tab:glmm_coefficients",
) -> pd.DataFrame:
    """Export a GLMM coefficient summary as a LaTeX table.

    Args:
        glmm_summary_df: Indexed by predictor name, with columns
            "Mean_Estimate" and "Mean_SE" -- the format produced by
            averaging fit_glmm()'s per-instance `estimates` dict across
            an ensemble (see notebooks/glm_multiprocessed_*.py).
        output_path: File path the LaTeX table is written to.
        label_map: Maps predictor names to LaTeX display labels; defaults
            to `GLMM_PREDICTOR_LABEL_MAP`.
        caption: LaTeX table caption.
        label: LaTeX \\label{} reference for the table.

    Returns:
        The formatted table as a pandas DataFrame (also written to
        `output_path` as LaTeX).
    """
    if label_map is None:
        label_map = GLMM_PREDICTOR_LABEL_MAP

    z = glmm_summary_df["Mean_Estimate"] / glmm_summary_df["Mean_SE"]
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    def stars(pval):
        if pval < 0.001:
            return "***"
        elif pval < 0.01:
            return "**"
        elif pval < 0.05:
            return "*"
        return ""

    table_df = pd.DataFrame(
        {
            "Predictor": [label_map.get(name, name) for name in glmm_summary_df.index],
            "Estimate": [f"{v:.3f}" for v in glmm_summary_df["Mean_Estimate"]],
            "SE": [f"{v:.3f}" for v in glmm_summary_df["Mean_SE"]],
            "z": [f"{v:.2f}" for v in z],
            "Sig.": [stars(pv) for pv in p],
        }
    )

    latex_table = table_df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        caption=caption,
        label=label,
    )
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"GLMM LaTeX table exported to {output_path}")
    return table_df


def export_glmm_comparison_latex_table(
    comparison_df: pd.DataFrame,
    output_path: str,
    model_label: str = "Model",
    label_map: dict | None = None,
    caption: str | None = None,
    label: str = "tab:glmm_comparison",
) -> pd.DataFrame:
    """Export a human-vs-model GLMM coefficient comparison as a LaTeX table.

    Args:
        comparison_df: Output of plotting.plot_glmm_betas_comparison() --
            indexed by predictor name, with columns "Human_Estimate",
            "Human_SE", "{model_label}_Estimate", "{model_label}_SE",
            "Difference", "Diff_SE", "z", "p".
        output_path: File path the LaTeX table is written to.
        model_label: Name of the model being compared to human data; used
            as a column header and in the default caption.
        label_map: Maps predictor names to LaTeX display labels; defaults
            to `GLMM_PREDICTOR_LABEL_MAP`.
        caption: LaTeX table caption; defaults to an auto-generated
            "human vs. {model_label}" caption.
        label: LaTeX \\label{} reference for the table.

    Returns:
        The formatted table as a pandas DataFrame (also written to
        `output_path` as LaTeX).
    """
    if label_map is None:
        label_map = GLMM_PREDICTOR_LABEL_MAP

    # \texttt{} keeps repeated hyphens in model_label (e.g. "LB-XT-RPHCLUK",
    # "SBEXT-RPHC---") literal -- plain LaTeX text ligates "--"/"---" into
    # en-/em-dashes, which mangles these config names.
    model_label_tt = r"\texttt{" + model_label + r"}"
    if caption is None:
        caption = f"GLMM fixed-effect estimates: human data vs.\\ {model_label_tt}."

    table_df = pd.DataFrame(
        {
            "Predictor": [label_map.get(name, name) for name in comparison_df.index],
            "Human": [f"{v:.3f}" for v in comparison_df["Human_Estimate"]],
            model_label_tt: [
                f"{v:.3f}" for v in comparison_df[f"{model_label}_Estimate"]
            ],
            "Difference": [f"{v:.3f}" for v in comparison_df["Difference"]],
        }
    )

    # Emit the bare tabular, not a full table environment. The manuscript wraps
    # this file in its own table with its own caption and label; passing
    # caption/label to to_latex() emits a second, nested table environment, which
    # raises "Not in outer par mode", drops text, and defines the label twice.
    latex_table = table_df.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
    )
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"GLMM comparison LaTeX table exported to {output_path}")
    return table_df
