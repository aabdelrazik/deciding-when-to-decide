import sys
import os
import glob
import json

import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import scipy.stats as stats
from tqdm.auto import tqdm

# Add the src directory to the Python path
project_root = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(project_root)

from src.config.loader import load_config
from src.utils import (
    combine_sequences,
    load_questionnaire_data,
    assemble_glm_outputs,
    clean_and_merge_betas,
    plot_averaged_glmm_betas,
)
from src.glm import export_glmm_latex_table, fit_glm, fit_glmm
from src.utils.plotting import (
    IS_LATEX as is_latex,
    save_figure,
    set_plot_style,
)

font_size = 20
config_dir = os.path.join(project_root, "data/simulation_configs")

# Combined-horizon configs (TASK starting with "C") already fit short+long
# trials together in one config, so each raw_simulations/sim_run_{i}_*.pkl
# already holds the complete combined dataset for that run -- no separate
# short/long configs to load and concatenate, unlike glm_multiprocessed_simulate.py.
n_instances = 120






set_plot_style()


def worker_read_and_fit(args):
    """Read this instance's pre-simulated combined trial data and fit the GLM/GLMM."""
    instance_idx, sim_file_path, ocir_all, ybocs_all = args

    if not os.path.exists(sim_file_path):
        return None, None

    all_simulated_data = pd.read_pickle(sim_file_path)
    combined_df = combine_sequences(all_simulated_data)

    # GLM Fit
    betas_all, id_all = fit_glm(combined_df, source="df")
    glm_outputs = assemble_glm_outputs(betas_all, id_all)

    pmat_z_all_df = glm_outputs["pmat_z_all_df"]
    N_betas = glm_outputs["n_betas"]
    pdecide_betas_arr = glm_outputs["pdecide_betas"]

    # Clean Betas
    betas_clean, _, _ = clean_and_merge_betas(
        pdecide_betas_arr, N_betas, ocir_all, ybocs_all
    )

    # GLMM Fit
    glm_results, estimates, glmm_data = fit_glmm(pmat_z_all_df, ocir_all)

    # Check for GLMM divergence (absolute estimate > 100)
    exploded = False
    for key in ["group1_estimates", "group2_estimates"]:
        if key in estimates:
            if (estimates[key].abs() > 100).any():
                exploded = True
                break

    if exploded:
        return betas_clean, None

    return betas_clean, estimates


# ---------------------------------------------------------
# MAIN EXECUTION BLOCK (Required for Multiprocessing)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading data and configuration...")

    ocir_all, ybocs_all = load_questionnaire_data(project_root)
    data_root_human = os.path.join(project_root, "data/TrHu_NHB_light/data_MEG")
    betas_clean_human = pd.read_csv(
        os.path.join(data_root_human, "glm_betas_human.csv")
    )

    # SIM_CONFIG_PATH picks the combined config (single config, same convention as
    # fit_data.py). Defaults to BIC/best_models.json's "combined" pick if unset.
    config_path = os.environ.get("SIM_CONFIG_PATH")
    if not config_path:
        best_models_path = os.path.join(project_root, "BIC", "best_models.json")
        with open(best_models_path) as f:
            best_models = json.load(f)
        config_path = os.path.join(
            config_dir, f"simulation_params_{best_models['combined']}.py"
        )

    cfg = load_config(config_path)
    print(f"combined config: {cfg.TASK} (POMDP_TYPE={cfg.POMDP_TYPE})")

    results_df = pd.read_pickle(cfg.RESULTS_PATH)
    data_root = os.path.join(cfg.DATA_PATH, "glm_combined")
    os.makedirs(data_root, exist_ok=True)

    # Figures mirror data_root's data/{POMDP,POMDP_commit}/<task>/... subpath
    # under figures/{POMDP,POMDP_commit}/<task>/... (same convention as
    # export_glmm_comparison_commit.py and the fixed glm_multiprocessed_simulate.py)
    # instead of living in data_root itself, which is gitignored for POMDP
    # (unlike figures/) and so was never actually trackable in git like every
    # other figure in the repo. Pickles/.tex tables stay in data_root.
    _data_subdir = "POMDP_commit" if cfg.POMDP_COMMIT else "POMDP"
    _rel_data_root = os.path.relpath(
        data_root, os.path.join(project_root, "data", _data_subdir)
    )
    glm_figures_path = os.path.join(project_root, "figures", _data_subdir, _rel_data_root)
    os.makedirs(glm_figures_path, exist_ok=True)

    raw_sim_dir = os.path.join(cfg.DATA_PATH, "raw_simulations")
    horizon_str = cfg.FIT_HORIZON[
        0
    ]  # whichever fit_data.py happened to name these with
    n_available = len(
        glob.glob(os.path.join(raw_sim_dir, f"sim_run_*_{horizon_str}.pkl"))
    )
    if n_available == 0:
        raise FileNotFoundError(
            f"No raw_simulations found in {raw_sim_dir} (looked for sim_run_*_{horizon_str}.pkl). "
            f"Run fit_data.py for this config first -- it writes these as part of its ensemble step."
        )
    n_instances = min(n_instances, n_available)
    print(f"Using {n_instances} pre-simulated instances ({n_available} available)")

    # ---------------------------------------------------------
    # 2. RUN MULTIPROCESSED FITTING ON PRE-SIMULATED DATA
    # ---------------------------------------------------------
    print(f"\nStarting Multiprocessed Ensemble Fitting ({n_instances} instances)...")

    pool_args = [
        (
            i,
            os.path.join(raw_sim_dir, f"sim_run_{i}_{horizon_str}.pkl"),
            ocir_all,
            ybocs_all,
        )
        for i in range(n_instances)
    ]

    num_cores = 64
    print(f"Utilizing {num_cores} CPU cores for parallel processing.")

    with mp.Pool(processes=num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(worker_read_and_fit, pool_args),
                total=n_instances,
                desc="Fitting GLMs",
            )
        )

    valid_results = [
        res
        for res in results
        if res is not None and res[0] is not None and res[1] is not None
    ]

    num_discarded = len(results) - len(valid_results)
    if num_discarded > 0:
        print(
            f"\nWarning: Discarded {num_discarded} instance(s) due to GLMM divergence (estimates > 100) or missing files."
        )

    if not valid_results:
        raise ValueError(
            "All simulated instances diverged! Please check your simulation parameters or data."
        )

    all_betas_clean = [res[0] for res in valid_results]
    all_glmm_estimates = [res[1] for res in valid_results]
    all_betas_path = os.path.join(data_root, "all_betas_clean_simulated.pkl")
    all_glmm_path = os.path.join(data_root, "all_glmm_estimates_simulated.pkl")
    pd.to_pickle(all_betas_clean, all_betas_path)
    pd.to_pickle(all_glmm_estimates, all_glmm_path)

    # ---------------------------------------------------------
    # 3. CALCULATE AVERAGES ACROSS THE ENSEMBLE
    # ---------------------------------------------------------
    print(
        f"\nCalculating ensemble averages from {len(valid_results)} valid instances..."
    )

    concat_betas = pd.concat(all_betas_clean)
    avg_betas_clean = concat_betas.groupby("userID").mean().reset_index()

    output_path_glm = os.path.join(data_root, "glm_betas_simulated_averaged.pkl")
    avg_betas_clean.to_pickle(output_path_glm)
    print(f"Averaged GLM betas saved to {output_path_glm}")

    print("Averaging GLMM estimates...")
    avg_glmm_estimates = {}

    # fit_glmm's estimates dict mixes pandas Series (group*_estimates/se)
    # with plain scalars (accuracy, pseudo_r2) -- these need different
    # averaging strategies, so they can't share one pd.concat call (see
    # glm_multiprocessed_simulate.py, which already has this fix).
    series_keys = ["group1_estimates", "group1_se", "group2_estimates", "group2_se"]
    scalar_keys = ["accuracy", "pseudo_r2"]

    for key in series_keys:
        series_list = [
            est[key] for est in all_glmm_estimates if est.get(key) is not None
        ]
        if series_list:
            avg_glmm_estimates[key] = pd.concat(series_list, axis=1).mean(axis=1)

    for key in scalar_keys:
        avg_glmm_estimates[key] = np.mean([est[key] for est in all_glmm_estimates])

    final_glmm_estimates = pd.concat(
        [avg_glmm_estimates["group1_estimates"], avg_glmm_estimates["group2_estimates"]]
    )
    final_glmm_ses = pd.concat(
        [avg_glmm_estimates["group1_se"], avg_glmm_estimates["group2_se"]]
    )

    glmm_summary_df = pd.DataFrame(
        {"Mean_Estimate": final_glmm_estimates, "Mean_SE": final_glmm_ses}
    )
    output_path_glmm = os.path.join(data_root, "glmm_estimates_simulated_averaged.pkl")
    glmm_summary_df.to_pickle(output_path_glmm)
    print(f"Averaged GLMM estimates saved to {output_path_glmm}")

    glmm_table_path = os.path.join(data_root, "glmm_coefficients_table.tex")
    export_glmm_latex_table(glmm_summary_df, glmm_table_path)

    # ---------------------------------------------------------
    # 4. PLOT 1: GLM HUMAN VS MODEL AVERAGES
    # ---------------------------------------------------------
    print("\nPlotting Averaged Human vs Model GLM Betas...")

    df = betas_clean_human.merge(
        avg_betas_clean, on="userID", suffixes=("_real", "_model")
    )

    beta_label_map_glm = {
        1: r"ES_{t-1}",
        2: r"\Delta ES_t",
        3: r"trial",
        4: r"termination",
        5: r"FA2:ES_{t-1}",
        6: r"FA2:\Delta ES_t",
        7: r"FA2:trial",
        8: r"FA2:termination",
    }
    betas = [f"beta{i}" for i in range(1, 8)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, beta in enumerate(betas):
        x = df[f"{beta}_real"]
        y = df[f"{beta}_model"]
        sns.scatterplot(x=x, y=y, ax=axes[i])
        sns.regplot(x=x, y=y, ax=axes[i], scatter=False, color="red")
        corr = x.corr(y)
        label = beta_label_map_glm[i + 1]
        axes[i].set_title(f"${label}$ (r = {corr:.2f})")
        axes[i].set_xlabel("Real")
        axes[i].set_ylabel("Model (Ensemble Avg)")
    axes[7].set_visible(False)
    plt.tight_layout()

    glm_plot_base = os.path.join(glm_figures_path, "glm_human_vs_model_averaged")
    save_figure(fig, glm_plot_base)
    print(f"Saved GLM scatter plot to: {glm_plot_base}.[pdf|svg|png]")

    # ---------------------------------------------------------
    # 5. PLOT 2: GLMM AVERAGED BETAS
    # ---------------------------------------------------------
    print("\nPlotting Averaged GLMM Betas...")

    fig_glmm, axes_glmm = plot_averaged_glmm_betas(
        glmm_summary_df, is_latex=is_latex, font_size=font_size
    )
    glmm_plot_base = os.path.join(glm_figures_path, "glmm_averaged_betas_plot")
    save_figure(fig_glmm, glmm_plot_base)
    print(f"Saved GLMM plot to: {glmm_plot_base}.[pdf|svg|png]")

    # ---------------------------------------------------------
    # 6. PLOT 3: MODEL PARAMETERS VS AVERAGED BETAS HEATMAP
    # ---------------------------------------------------------
    print("\nPlotting Model Parameters vs Averaged Betas Heatmap...")

    beta_label_map = {
        1: r"$\beta_{ES_{t-1}}$",
        2: r"$\beta_{\Delta ES_t}$",
        3: r"$\beta_{\mathrm{trial}}$",
        4: r"$\beta_{\mathrm{termination}}$",
        5: r"$\beta_{FA2:ES_{t-1}}$",
        6: r"$\beta_{FA2:\Delta ES_t}$",
        7: r"$\beta_{FA2:\mathrm{trial}}$",
    }
    beta_cols = [f"beta{i}" for i in beta_label_map.keys()]

    params_matrix = np.vstack(results_df["fit_params_ga"].values)
    params_df = pd.DataFrame(params_matrix, columns=cfg.PARAM_ORDER)
    params_df["userID"] = results_df["subject_ID"].values

    betas_df = avg_betas_clean.copy()
    if "userID" not in betas_df.columns:
        betas_df = betas_df.rename(columns={"subject_ID": "userID"})
    betas_df = betas_df[["userID"] + beta_cols]

    merged_df = params_df.merge(betas_df, on="userID", how="inner")

    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    r_matrix = pd.DataFrame(index=cfg.PARAM_ORDER, columns=beta_cols)
    stars_matrix = pd.DataFrame(index=cfg.PARAM_ORDER, columns=beta_cols)

    for param in cfg.PARAM_ORDER:
        for beta in beta_cols:
            df_tmp = merged_df[[param, beta]].dropna()
            if len(df_tmp) > 2:
                r, p = pearsonr(df_tmp[param], df_tmp[beta])
                r_matrix.loc[param, beta] = r
                stars_matrix.loc[param, beta] = significance_stars(p)
            else:
                r_matrix.loc[param, beta] = np.nan
                stars_matrix.loc[param, beta] = ""

    r_matrix = r_matrix.astype(float)
    pretty_labels = {f"beta{i}": beta_label_map[i] for i in beta_label_map}
    r_matrix = r_matrix.rename(columns=pretty_labels)
    stars_matrix = stars_matrix.rename(columns=pretty_labels)

    annot_matrix = r_matrix.copy().astype(str)
    for i in r_matrix.index:
        for j in r_matrix.columns:
            r = r_matrix.loc[i, j]
            stars = stars_matrix.loc[i, j]
            annot_matrix.loc[i, j] = f"{r:.2f}{stars}" if pd.notna(r) else ""

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

    heatmap_base = os.path.join(glm_figures_path, "param_vs_avg_betas_heatmap")
    save_figure(fig_heatmap, heatmap_base)
    print(f"Saved Heatmap plot to: {heatmap_base}.[pdf|svg|png]")

    # ---------------------------------------------------------
    # 7. PLOT 4: EXAGGERATION FACTOR VS OCIR TOTAL SCATTER PLOT
    # ---------------------------------------------------------
    print("\nPlotting Exaggeration Factor vs OCIR_total...")

    merged_df["userID_str"] = merged_df["userID"].astype(str)
    ocir_all["userID_str"] = ocir_all["userID"].astype(str)
    analysis_df = pd.merge(merged_df, ocir_all, on="userID_str", how="inner")

    if "exaggeration_factor" in analysis_df.columns:
        corr_amp, p_value_amp = stats.pearsonr(
            analysis_df["exaggeration_factor"], analysis_df["OCIR_total"]
        )
        sig_label_amp = "*" if p_value_amp < 0.05 else "n.s."
        print(f"Pearson r: {corr_amp:.4f}, p: {p_value_amp:.2e} ({sig_label_amp})")

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
            x=2,
            color="orange",
            linestyle=":",
            linewidth=2,
            label="Exaggeration threshold (>2)",
        )
        plt.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        plt.title(
            f"OCIR_total vs Exaggeration Factor (All Subjects)\nr = {corr_amp:.3f} ({sig_label_amp}, p = {np.round(p_value_amp,4)})",
            fontsize=14,
            pad=20,
        )
        plt.xlabel("Exaggeration Factor", fontsize=12)
        plt.ylabel("OCIR_total Score", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        scatter_base_amp = os.path.join(glm_figures_path, "ocir_correlation_exaggeration")
        save_figure(fig_amp, scatter_base_amp)
        print(
            f"Saved OCIR vs Exaggeration scatter plot to: {scatter_base_amp}.[pdf|svg|png]"
        )

    # ---------------------------------------------------------
    # 8. PLOT 5: PATIENCE VS OCIR TOTAL SCATTER PLOT
    # ---------------------------------------------------------
    print("\nPlotting Patience vs OCIR_total...")

    if "patience" in analysis_df.columns:
        corr_pat, p_value_pat = stats.pearsonr(
            analysis_df["patience"], analysis_df["OCIR_total"]
        )
        sig_label_pat = "*" if p_value_pat < 0.05 else "n.s."
        print(f"Pearson r: {corr_pat:.4f}, p: {p_value_pat:.2e} ({sig_label_pat})")

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
            f"OCIR_total vs Patience (All Subjects)\nr = {corr_pat:.3f} ({sig_label_pat}, p = {np.round(p_value_pat,4)})",
            fontsize=14,
            pad=20,
        )
        plt.xlabel("Patience", fontsize=12)
        plt.ylabel("OCIR_total Score", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        scatter_base_pat = os.path.join(glm_figures_path, "ocir_correlation_patience")
        save_figure(fig_pat, scatter_base_pat)
        print(
            f"Saved OCIR vs Patience scatter plot to: {scatter_base_pat}.[pdf|svg|png]"
        )

        summary_pat = pd.DataFrame(
            {
                "Metric": ["Pearson r", "p-value", "Significance", "N total"],
                "Value": [
                    f"{corr_pat:.4f}",
                    f"{p_value_pat:.2e}",
                    sig_label_pat,
                    len(analysis_df),
                ],
            }
        )
        print("\nSummary (Patience):")
        print(summary_pat)
    else:
        print("Warning: 'patience' not found in parameters. Plotting skipped.")
