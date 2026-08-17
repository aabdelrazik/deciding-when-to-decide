# -*- coding: utf-8 -*-
# Full integrated DTD stats pipeline with robust questionnaire handling
# Paste into notebook cell and run. Assumes same repo layout as before.

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

project_root = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(project_root)
from src.glm import *
from src.utils import (
    load_simulation_results,
    combine_sequences,
    safe_spearman,
    plot_beta_with_significance,
    plot_glmm_betas,
    load_questionnaire_data,
    assemble_glm_outputs,
    plot_beta_correlations,
    prepare_choice_trials,
    clean_and_merge_betas,
    plot_human_vs_sim_beta,
)

is_latex = False
font_size = 20


# Performance summary:
def summarize_glm_performance(betas_all):
    """Summarize GLM performance across subjects."""
    valid_results = [b for b in betas_all if b is not None]
    df_results = pd.DataFrame(valid_results)

    print("GLM Performance Summary:")
    print(df_results[["id", "pseudo_r2", "accuracy", "aic", "n_trials"]].describe())

    # Beta significance (p < 0.05)
    beta_cols = [f"beta_{i}" for i in range(7)]  # 6 predictors + intercept
    beta_df = pd.DataFrame(df_results["pdecide_beta"].tolist(), columns=beta_cols)
    pval_df = pd.DataFrame(df_results["pvals"].tolist(), columns=beta_cols)

    sig_betas = (pval_df < 0.05).sum()
    print("\nSignificant betas (p<0.05) per predictor:")
    print(sig_betas)

    return df_results


import numpy as np


def predict_draw_by_draw(totevminus, deltaev, trial, termination, subject_params):
    """
    Predicts whether to 'decide' or 'wait' at each draw in a sequence.

    Args:
        totevminus, deltaev, trial, termination: 1D numpy arrays calculated
                                                 for the sequence of draws.
        subject_params: The dictionary from `betas_all` for a specific subject.

    Returns:
        probabilities: Array of probabilities for making a decision at each draw.
        decisions: Array of binary choices (1 = decide, 0 = wait).
    """
    # 1. Build the predictor matrix exactly as done in training
    sequence_vars = np.column_stack(
        [
            totevminus,
            deltaev,
            trial,
            termination,
            totevminus * termination,
            trial * termination,
        ]
    )

    # 2. Extract the subject's specific scaling parameters and betas
    mu = subject_params["mu"]
    sigma = subject_params["sigma"]
    betas = subject_params["pdecide_beta"]

    if np.isnan(betas).any():
        raise ValueError(
            "This subject's model contains NaN betas. Prediction cannot run."
        )

    # 3. Z-score the sequence variables using the TRAINING data's parameters
    regs_z = (sequence_vars - mu) / sigma

    # 4. Append the constant column (statsmodels prepend=False puts it at the end)
    X_seq = np.column_stack([regs_z, np.ones(len(regs_z))])

    # 5. Compute the log-odds using the dot product
    log_odds = X_seq @ betas

    # 6. Apply the sigmoid function to get probabilities
    probabilities = 1 / (1 + np.exp(-log_odds))

    # 7. Convert to binary decision (Threshold = 0.5)
    # 1 = Decide, 0 = Wait
    decisions = (probabilities > 0.5).astype(int)

    return probabilities, decisions


data_root = os.path.join(project_root, "data/TrHu_NHB_light/data_MEG")
ocir_all, ybocs_all = load_questionnaire_data(project_root)
human_data_path = os.path.join(
    os.path.join(data_root, "behdat_preprocessed.pkl")
)

human_data = pd.read_pickle(human_data_path)

# # 1. Create a clean copy of your main dataframe
# filtered_human_data = human_data.copy()

# # 2. Filter the internal 'data' DataFrame for each row using .apply()
# filtered_human_data['data'] = filtered_human_data['data'].apply(
#     lambda df: df[df['termination'] == 1].copy()
# )


betas_all, id_all = fit_glm_separate_for_human_data(human_data, ocir_all=ocir_all)

# betas_all,id_all=fit_glm_separate_for_human_data(filtered_human_data,ocir_all=ocir_all)

glm_outputs = assemble_glm_outputs(betas_all, id_all)

pmat_all_df = glm_outputs["pmat_all_df"]
pmat_z_all_df = glm_outputs["pmat_z_all_df"]
pmat_choice = glm_outputs["pmat_choice"]
pmat_long = glm_outputs["pmat_long"]
pmat_short = glm_outputs["pmat_short"]
N_betas = glm_outputs["n_betas"]
pdecide_betas_arr = glm_outputs["pdecide_betas"]
means = glm_outputs["mu_df"]
stds = glm_outputs["sigma_df"]


plt.figure(figsize=(6, 4))
plt.hist(
    pmat_long["trial"].dropna(),
    bins=np.arange(0, 15) - 0.5,
    density=True,
    edgecolor="w",
    label="long",
    color="#00a676",
)
plt.hist(
    pmat_short["trial"].dropna(),
    bins=np.arange(0, 15) - 0.5,
    density=True,
    edgecolor="w",
    label="short",
    color="#4e4187",
)
plt.xlim([0, 14])
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend()
plt.title("Histogram of DTD by horizon")
plt.xlabel("trial")
plt.ylabel("probability")
plt.show()

# ---------------- participant-level means ----------------
pmat_choice_ocir = pmat_choice.merge(ocir_all, on="userID", how="inner")
pmat_choice_all = (
    pmat_choice_ocir.merge(ybocs_all, on="userID", how="inner")
    if "userID" in ybocs_all.columns
    else pmat_choice_ocir.copy()
)
grp = pmat_choice_all.groupby("userID")
meanDraws = grp["trial"].mean()
meanFA2 = (
    grp["FA2"].mean() if "FA2" in pmat_choice_all.columns else pd.Series(dtype=float)
)
meanOCIR = (
    grp["OCIR_total"].mean()
    if "OCIR_total" in pmat_choice_all.columns
    else pd.Series(dtype=float)
)
meanYBOCS_total = (
    grp["YBOCS_total_score"].mean()
    if "YBOCS_total_score" in pmat_choice_all.columns
    else pd.Series(dtype=float)
)
meanYBOCS_comp = (
    grp["YBOCS_compulsions_subtotal"].mean()
    if "YBOCS_compulsions_subtotal" in pmat_choice_all.columns
    else pd.Series(dtype=float)
)
meanYBOCS_obs = (
    grp["YBOCS_obsess_subtotal"].mean()
    if "YBOCS_obsess_subtotal" in pmat_choice_all.columns
    else pd.Series(dtype=float)
)

print("Draws ~ FA2:", safe_spearman(meanDraws, meanFA2))
print("Draws ~ OCIR:", safe_spearman(meanDraws, meanOCIR))
print("Draws ~ YBOCS total:", safe_spearman(meanDraws, meanYBOCS_total))
print("Draws ~ YBOCS comp:", safe_spearman(meanDraws, meanYBOCS_comp))
print("Draws ~ YBOCS obs:", safe_spearman(meanDraws, meanYBOCS_obs))
pmat_choice, pmat_long, pmat_short = prepare_choice_trials(pmat_all_df)

betas_clean, betas_ocir, betas_allq = clean_and_merge_betas(
    pdecide_betas_arr, N_betas, ocir_all, ybocs_all
)

plot_beta_correlations(
    betas_ocir, betas_allq, beta_col="beta2", is_latex=is_latex, font_size=font_size
)

plot_beta_correlations(
    betas_ocir, betas_allq, beta_col="beta1", is_latex=is_latex, font_size=font_size
)

# save the beta values
output_path = os.path.join(data_root, "glm_betas_human.csv")
betas_clean.to_csv(output_path, index=False)


glm_results, estimates, glmm_data = fit_glmm(pmat_z_all_df, ocir_all)

# print(glm_results.summary())
fig, ax = plot_glmm_betas(glm_results, is_latex=is_latex)
plt.show()
