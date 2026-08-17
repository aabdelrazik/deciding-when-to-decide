"""Plot-only replay of glm_multiprocessed_simulate.py's figures (sections
4-8) for every *_commit long config already processed by
submit_glm_array_commit.sh / run_glm_array_commit.sh.

That array job is expensive (200-instance GLM/GLMM ensemble fit per long
config) and has already been run -- its outputs
(glm_betas_simulated_averaged.pkl, glmm_estimates_simulated_averaged.pkl)
are cached under data/POMDP_commit/<TASK>/glm_vs_<short>/. This script
does NOT refit anything; it only re-reads those cached pickles and redraws
the figures, writing them to figures/POMDP_commit/<TASK>/glm_vs_<short>/
instead of the flat ../GLM/ directory glm_multiprocessed_simulate.py itself
writes to.

Long configs whose array task never completed (missing either cached
pickle) are skipped with a warning rather than refit.

Usage:
    uv run --directory notebooks plot_glm_array_commit_cached.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.config.loader import load_config
from src.utils import load_questionnaire_data, plot_averaged_glmm_betas
from src.utils.plotting import save_figure, set_plot_style

is_latex = False
font_size = 20
config_dir = os.path.join(project_root, "data/simulation_configs")

set_plot_style()

# per-subject GLM (fit_glm_separate_for_human_data) has no FA2 term -- FA2 is
# constant within a subject, so beta5/6 are the two within-subject
# interaction terms and beta7 is the intercept (see the matching NOTE in
# glm_multiprocessed_simulate.py).
BETA_LABEL_MAP_GLM = {
    1: r"ES_{t-1}",
    2: r"\Delta ES_t",
    3: r"trial",
    4: r"termination",
    5: r"ES_{t-1}\times termination",
    6: r"trial \times termination",
    7: r"\mathrm{intercept}",
}
BETA_LABEL_MAP = {
    1: r"$\beta_{ES_{t-1}}$",
    2: r"$\beta_{\Delta ES_t}$",
    3: r"$\beta_{\mathrm{trial}}$",
    4: r"$\beta_{\mathrm{termination}}$",
    5: r"$\beta_{ES_{t-1} \times \mathrm{term}}$",
    6: r"$\beta_{\mathrm{trial} \times \mathrm{term}}$",
    7: r"$\beta_{\mathrm{intercept}}$",
}
BETA_COLS = [f"beta{i}" for i in BETA_LABEL_MAP.keys()]


def make_figures(config_short, config_long, avg_betas_clean, glmm_summary_df,
                  betas_clean_human, ocir_all, figures_path):
    os.makedirs(figures_path, exist_ok=True)
    results_df_short = pd.read_pickle(config_short.RESULTS_PATH)
    results_df_long = pd.read_pickle(config_long.RESULTS_PATH)

    # ---- PLOT 1: GLM human vs model averages ----
    df = betas_clean_human.merge(
        avg_betas_clean, on="userID", suffixes=("_real", "_model")
    )
    betas = [f"beta{i}" for i in range(1, 8)]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, beta in enumerate(betas):
        x = df[f"{beta}_real"]
        y = df[f"{beta}_model"]
        sns.scatterplot(x=x, y=y, ax=axes[i])
        sns.regplot(x=x, y=y, ax=axes[i], scatter=False, color="red")
        corr = x.corr(y)
        label = BETA_LABEL_MAP_GLM[i + 1]
        axes[i].set_title(f"${label}$\n(r = {corr:.2f})", fontsize=13)
        axes[i].set_xlabel("Real")
        axes[i].set_ylabel("Model (Ensemble Avg)")
    axes[7].set_visible(False)
    fig.tight_layout(w_pad=2.0, h_pad=2.5)
    save_figure(fig, os.path.join(figures_path, "glm_human_vs_model_averaged"))

    # ---- PLOT 2: GLMM averaged betas ----
    fig_glmm, _ = plot_averaged_glmm_betas(
        glmm_summary_df, is_latex=is_latex, font_size=font_size, save_fig=False
    )
    save_figure(fig_glmm, os.path.join(figures_path, "glmm_averaged_betas_plot"))

    # ---- PLOT 3: model parameters vs averaged betas heatmap ----
    params_matrix = np.vstack(results_df_long["fit_params_ga"].values)
    params_df = pd.DataFrame(params_matrix, columns=config_long.PARAM_ORDER)
    params_df["userID"] = results_df_short["subject_ID"].values

    betas_df = avg_betas_clean.copy()
    if "userID" not in betas_df.columns:
        betas_df = betas_df.rename(columns={"subject_ID": "userID"})
    betas_df = betas_df[["userID"] + BETA_COLS]

    merged_df = params_df.merge(betas_df, on="userID", how="inner")

    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    r_matrix = pd.DataFrame(index=config_long.PARAM_ORDER, columns=BETA_COLS)
    stars_matrix = pd.DataFrame(index=config_long.PARAM_ORDER, columns=BETA_COLS)
    for param in config_long.PARAM_ORDER:
        for beta in BETA_COLS:
            df_tmp = merged_df[[param, beta]].dropna()
            if len(df_tmp) > 2:
                r, p = pearsonr(df_tmp[param], df_tmp[beta])
                r_matrix.loc[param, beta] = r
                stars_matrix.loc[param, beta] = significance_stars(p)
            else:
                r_matrix.loc[param, beta] = np.nan
                stars_matrix.loc[param, beta] = ""

    r_matrix = r_matrix.astype(float)
    pretty_labels = {f"beta{i}": BETA_LABEL_MAP[i] for i in BETA_LABEL_MAP}
    r_matrix = r_matrix.rename(columns=pretty_labels)
    stars_matrix = stars_matrix.rename(columns=pretty_labels)

    annot_matrix = r_matrix.copy().astype(str)
    for i in r_matrix.index:
        for j in r_matrix.columns:
            r = r_matrix.loc[i, j]
            s = stars_matrix.loc[i, j]
            annot_matrix.loc[i, j] = f"{r:.2f}{s}" if pd.notna(r) else ""

    fig_heatmap = plt.figure(figsize=(10, 6))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        r_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 9},
        cbar_kws={"label": "Pearson r"},
    )
    plt.title("Correlation: Model Parameters vs Averaged Betas")
    plt.xlabel("Averaged Betas")
    plt.ylabel("Parameters")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig_heatmap, os.path.join(figures_path, "param_vs_avg_betas_heatmap"))

    # ---- PLOT 4/5: OCIR correlations (exaggeration / patience) ----
    merged_df["userID_str"] = merged_df["userID"].astype(str)
    ocir_all["userID_str"] = ocir_all["userID"].astype(str)
    analysis_df = pd.merge(merged_df, ocir_all, on="userID_str", how="inner")

    if "exaggeration_factor" in analysis_df.columns:
        corr_amp, p_value_amp = stats.pearsonr(
            analysis_df["exaggeration_factor"], analysis_df["OCIR_total"]
        )
        sig_label_amp = "*" if p_value_amp < 0.05 else "n.s."

        fig_amp = plt.figure(figsize=(10, 7))
        plt.scatter(
            analysis_df["exaggeration_factor"],
            analysis_df["OCIR_total"],
            c="steelblue",
            alpha=0.6,
            s=80,
            edgecolors="white",
            linewidth=0.5,
        )
        sns.regplot(
            data=analysis_df,
            x="exaggeration_factor",
            y="OCIR_total",
            scatter=False,
            color="black",
            line_kws={"linestyle": "--", "linewidth": 2},
        )
        plt.axvline(
            x=2, color="orange", linestyle=":", linewidth=2,
            label="Exaggeration threshold (>2)",
        )
        plt.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        plt.title(
            f"OCIR_total vs Exaggeration Factor (All Subjects)\n"
            f"r = {corr_amp:.3f} ({sig_label_amp}, p = {np.round(p_value_amp, 4)})",
            fontsize=14, pad=20,
        )
        plt.xlabel("Exaggeration Factor", fontsize=12)
        plt.ylabel("OCIR_total Score", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig_amp, os.path.join(figures_path, "ocir_correlation_exaggeration"))

    if "patience" in analysis_df.columns:
        corr_pat, p_value_pat = stats.pearsonr(
            analysis_df["patience"], analysis_df["OCIR_total"]
        )
        sig_label_pat = "*" if p_value_pat < 0.05 else "n.s."

        fig_pat = plt.figure(figsize=(10, 7))
        plt.scatter(
            analysis_df["patience"],
            analysis_df["OCIR_total"],
            c="mediumseagreen",
            alpha=0.6,
            s=80,
            edgecolors="white",
            linewidth=0.5,
        )
        sns.regplot(
            data=analysis_df,
            x="patience",
            y="OCIR_total",
            scatter=False,
            color="black",
            line_kws={"linestyle": "--", "linewidth": 2},
        )
        plt.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        plt.title(
            f"OCIR_total vs Patience (All Subjects)\n"
            f"r = {corr_pat:.3f} ({sig_label_pat}, p = {np.round(p_value_pat, 4)})",
            fontsize=14, pad=20,
        )
        plt.xlabel("Patience", fontsize=12)
        plt.ylabel("OCIR_total Score", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig_pat, os.path.join(figures_path, "ocir_correlation_patience"))


if __name__ == "__main__":
    ocir_all, ybocs_all = load_questionnaire_data(project_root)
    data_root_human = os.path.join(project_root, "data/TrHu_NHB_light/data_MEG")
    betas_clean_human = pd.read_csv(
        os.path.join(data_root_human, "glm_betas_human.csv")
    )

    with open(os.path.join(project_root, "BIC_commit/best_models.json")) as f:
        best_models = json.load(f)
    config_short = load_config(
        os.path.join(config_dir, f"simulation_params_{best_models['short']}_commit.py")
    )
    print(f"short config: {config_short.TASK} (fixed, best short commit model)")

    long_files = sorted(
        glob.glob(os.path.join(config_dir, "simulation_params_L*_commit.py"))
    )

    for long_file in long_files:
        config_long = load_config(long_file)
        data_root = os.path.normpath(
            os.path.join(config_long.DATA_PATH, "..", f"glm_vs_{config_short.TASK}")
        )
        glm_pkl = os.path.join(data_root, "glm_betas_simulated_averaged.pkl")
        glmm_pkl = os.path.join(data_root, "glmm_estimates_simulated_averaged.pkl")

        if not (os.path.exists(glm_pkl) and os.path.exists(glmm_pkl)):
            print(f"Skipping {config_long.TASK}: no cached ensemble outputs at {data_root}")
            continue

        print(f"\n{config_long.TASK} vs {config_short.TASK}")
        avg_betas_clean = pd.read_pickle(glm_pkl)
        glmm_summary_df = pd.read_pickle(glmm_pkl)

        figures_path = os.path.normpath(
            os.path.join(config_long.FIGURE_PATH, "..", f"glm_vs_{config_short.TASK}")
        )
        make_figures(
            config_short, config_long, avg_betas_clean, glmm_summary_df,
            betas_clean_human, ocir_all, figures_path,
        )
        print(f"Saved figures to: {figures_path}")
