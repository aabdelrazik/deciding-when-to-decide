"""Overlay each model's cached GLMM coefficients on top of the human ("true")
GLMM fit, for every glmm_estimates_simulated_averaged.pkl under data/POMDP/.

Requires data/TrHu_NHB_light/data_MEG/glmm_estimates_human.pkl -- run
compute_human_glmm.py once first to produce it.

For each model, writes:
  - figures/POMDP/<task>/<rest-of-data-path>/glmm_betas_comparison_<task>_<horizon>.pdf/.svg/.png
    (mirrors the model's data/POMDP/ subpath under figures/POMDP/, same
    convention as export_glmm_comparison_commit.py, instead of the flat
    ../GLM/ this script used to write to)
  - glmm_comparison_table.tex (next to the model's own data, in data_root)

Usage:
    uv run --directory notebooks export_glmm_comparison.py
"""

import glob
import os
import re
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.glm import export_glmm_comparison_latex_table
from src.utils import plot_glmm_betas_comparison
from src.utils.plotting import _display_task_name

human_glmm_path = os.path.join(
    project_root, "data/TrHu_NHB_light/data_MEG/glmm_estimates_human.pkl"
)

if __name__ == "__main__":
    if not os.path.exists(human_glmm_path):
        raise FileNotFoundError(
            f"{human_glmm_path} not found -- run compute_human_glmm.py first."
        )
    human_glmm_df = pd.read_pickle(human_glmm_path)

    pkl_paths = sorted(
        glob.glob(
            os.path.join(
                project_root, "data/POMDP/**/glmm_estimates_simulated_averaged.pkl"
            ),
            recursive=True,
        )
    )
    if not pkl_paths:
        print("No glmm_estimates_simulated_averaged.pkl files found under data/POMDP/")

    for pkl_path in pkl_paths:
        data_root = os.path.dirname(pkl_path)
        model_glmm_df = pd.read_pickle(pkl_path)

        # e.g. data/POMDP/LB-XT-RPHCLUK/glm_vs_SBEXT-RPHC---/glmm_estimates_simulated_averaged.pkl
        match = re.search(r"data/POMDP/([^/]+)/", pkl_path)
        task = match.group(1) if match else os.path.basename(data_root)
        # one TASK can have multiple glm_vs_* subdirs (paired against different
        # short configs, to build the combined short+long trial set the GLMM is
        # fit on) -- name the short config in the label so the two pairs for
        # the same TASK are distinguishable; it's a combination, not a
        # short-vs-long comparison, so phrase it as "combined" rather than "vs".
        glm_subdir = os.path.basename(data_root)
        if glm_subdir == "glm_combined":
            horizon = "combined"
            condition_label = "long + short combined"
        else:
            short_cfg = glm_subdir.replace("glm_vs_", "")
            horizon = f"combined_{short_cfg}"
            condition_label = f"+ {short_cfg}"

        # Mirror data_root's data/POMDP/... subpath under figures/POMDP/...
        rel_data_root = os.path.relpath(
            data_root, os.path.join(project_root, "data/POMDP")
        )
        model_figures_path = os.path.join(project_root, "figures/POMDP", rel_data_root)
        os.makedirs(model_figures_path, exist_ok=True)

        print(
            f"\n{os.path.relpath(data_root, project_root)}  (task={task}, horizon={horizon})"
        )

        task_display = _display_task_name(task)

        fig, axes, comparison_df = plot_glmm_betas_comparison(
            human_glmm_df,
            model_glmm_df,
            model_label=task_display,
            horizon=horizon,
            condition_label=condition_label,
            path=model_figures_path,
            fname=f"glmm_betas_comparison_{task}_{horizon}",
        )

        table_path = os.path.join(data_root, "glmm_comparison_table.tex")
        export_glmm_comparison_latex_table(comparison_df, table_path, model_label=task_display)
