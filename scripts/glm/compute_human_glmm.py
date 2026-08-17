"""Compute the "true" GLMM fit on real human data -- the canonical "GLMM:
DECIDE vs WAIT" fit from Magda's_glm_fitting.ipynb (cell 33's pipeline,
res_decide = fit_glmm(pmat_z_all_df, ocir_all) on the unfiltered
behdat_preprocessed.pkl) -- and cache it in the same Mean_Estimate/Mean_SE
format as the simulated glmm_estimates_simulated_averaged.pkl files, so it
can be plotted against any model's GLMM coefficients without rerunning the
human GLM fit each time.

Note: human_data is NOT filtered to termination==1 here (unlike the
earlier/unused cells 2/4/5 in that notebook) -- that filter makes the
"termination" predictor constant within a subject's trials, which collapses
its GLMM coefficient to ~0. Cell 33's pipeline fits on the full data instead.

Usage:
    uv run --directory notebooks compute_human_glmm.py
"""

import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.glm import fit_glm_separate_for_human_data, fit_glmm
from src.utils import assemble_glm_outputs, load_questionnaire_data

data_root_human = os.path.join(project_root, "data/TrHu_NHB_light/data_MEG")

if __name__ == "__main__":
    ocir_all, _ = load_questionnaire_data(project_root)

    human_data = pd.read_pickle(
        os.path.join(data_root_human, "behdat_preprocessed.pkl")
    )

    betas_all, id_all = fit_glm_separate_for_human_data(human_data, ocir_all=ocir_all)
    glm_outputs = assemble_glm_outputs(betas_all, id_all)
    pmat_z_all_df = glm_outputs["pmat_z_all_df"]

    res_decide, estimates, glmm_data = fit_glmm(pmat_z_all_df, ocir_all)

    final_estimates = pd.concat(
        [estimates["group1_estimates"], estimates["group2_estimates"]]
    )
    final_ses = pd.concat([estimates["group1_se"], estimates["group2_se"]])
    glmm_human_df = pd.DataFrame(
        {"Mean_Estimate": final_estimates, "Mean_SE": final_ses}
    )

    output_path = os.path.join(data_root_human, "glmm_estimates_human.pkl")
    glmm_human_df.to_pickle(output_path)
    print(f"Human GLMM estimates saved to {output_path}")
    print(glmm_human_df)
