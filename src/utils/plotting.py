# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from math import ceil
import statsmodels.api as sm
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib as mpl
from .utils import safe_spearman
from .data_handling import (
    combine_ensemble_horizons,
    compute_corrected_glm_draws,
    compute_ensemble_glm_counts,
    pad_to_length,
)
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

def _detect_latex_available() -> bool:
    """Try an actual minimal LaTeX render via matplotlib's usetex machinery,
    not just a `latex` on-PATH check -- a texlive install can be present but
    still broken (e.g. missing dvipng, a broken mktexfmt/perl setup, as has
    happened on this project's dev machines before) and only fail once
    actually invoked. Runs once at import time: near-instant if `latex`
    isn't even on PATH, ~1s if it renders successfully.
    """
    import shutil

    if shutil.which("latex") is None:
        return False
    try:
        with mpl.rc_context({"text.usetex": True}):
            fig = plt.figure()
            fig.text(0, 0, r"$x$")
            fig.canvas.draw()
            plt.close(fig)
        return True
    except Exception:
        return False


# Single global switch for `is_latex` everywhere in this codebase -- every
# plot_* function here defaults to it, and notebooks/scripts should import it
# (`from src.utils.plotting import IS_LATEX as is_latex`) rather than
# hardcoding their own local True/False, which is how model_comparison.ipynb
# and model_comparison_commit.ipynb ended up silently disagreeing with each
# other. Auto-detected by default (see _detect_latex_available); set
# IS_LATEX_OVERRIDE below to True/False to force it regardless of what's
# actually installed on the current machine.
IS_LATEX_OVERRIDE: bool | None = None
IS_LATEX = IS_LATEX_OVERRIDE if IS_LATEX_OVERRIDE is not None else _detect_latex_available()

# Where figures go when a caller does not say. Resolved from this file so it
# does not depend on the working directory; POMDP_ROOT overrides it.
DEFAULT_FIGURE_PATH = os.path.join(
    os.environ.get("POMDP_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
    "figures",
)

# Maps internal parameter names → (LaTeX symbol, plain-text symbol)
PARAM_LABEL_MAP = {
    "belief_bias": (r"$\beta$", "B"),
    "exaggeration_factor": (r"$E$", "E"),
    "xi": (r"$\xi$", "ξ"),
    "tau": (r"$\tau$", "τ"),
    "gamma": (r"$\gamma$", "γ"),
    "subjective_cost": (r"$R_{\mathrm{risk}}$", "R_risk"),
    "patience": (r"$t_p$", "t_p"),
    "is_hazardous": (r"$H$", "H"),
    "c_max": (r"$\phi_{\max}$", "φ_max"),
    "hazard_lapse": (r"$L$", "L"),
    "urgency_coefficient": (r"$\phi_{\min}$", "φ_min"),
    "urgency_slope": (r"$k$", "k"),
}


def _get_param_labels(param_order: list[str], is_latex: bool = IS_LATEX) -> list[str]:
    """Return display labels for param_order using the paper symbol mapping."""
    idx = 0 if is_latex else 1
    return [PARAM_LABEL_MAP.get(p, (p, p))[idx] for p in param_order]


_CONDITION_TAG_RE = re.compile(r"^(.*?)_+(F|NF)$")

# Retained for reference only: these nine models are the ones whose hazard is
# genuinely off. Under the earlier naming convention a bare "-" at the hazard
# position stood both for "on by default" and for "off", so these nine were
# given a trailing prime at display time to keep them distinguishable. Task
# names now encode the three states directly ("H" fit, "h" on, "-" off), so
# the set is no longer consulted by any display path.
_HAZARD_DISPLAY_FALSE_OUTLIERS = frozenset({
    "C--XT-R------", "CB-XT-RP-C-UK", "CBEXT-RP-C-UK",
    "L--XT-R------", "LB-XT-RP-C-UK", "LBEXT-RP-C-UK",
    "S--XT-R------", "SB-XT-RP-C-UK", "SBEXT-RP-C-UK",
})


def _hazard_display_transform(base: str) -> tuple[str, bool]:
    """Display a task name as-is, returning (base, needs_prime=False).

    The hazard slot is now unambiguous in the name itself -- "H" when the
    hazard is fit, "h" when it is on but fixed, "-" when it is off -- so no
    display-time transformation is required. This previously collapsed "h"
    to "-" to match an earlier convention in which the dash stood for both
    hazard-on-by-default and hazard-off, and appended a prime to the handful
    of genuinely hazard-off models to keep them distinguishable. Both
    devices are obsolete; the function is kept so the LaTeX (\\texttt{}) and
    plain-text (matplotlib label) display paths retain a single shared hook.
    """
    return base, False


def _display_task_name(name: str) -> str:
    """Plain-text (non-LaTeX) version of the hazard h->'-' display collapse,
    for matplotlib figure labels/titles (axis tick labels, bar labels, plot
    titles) that show a raw model/task name outside of any \\texttt{}/LaTeX
    context. The three hazard=False outlier models (see
    _HAZARD_DISPLAY_FALSE_OUTLIERS) get a trailing prime instead of a LaTeX
    subscript. Does not handle the "_F"/"_NF" condition tag (that's LaTeX
    table-only via _texttt_with_condition_subscript); pass the base task
    name here.
    """
    if "+" in name:
        return "+".join(_display_task_name(part) for part in name.split("+"))
    base, needs_prime = _hazard_display_transform(name)
    return f"{base}'" if needs_prime else base


def _texttt_with_condition_subscript(name: str) -> str:
    """\\texttt{} a model/task name, rendering a trailing "_F"/"_NF" condition
    tag (future- vs. no-future-exaggeration, see load_fitted_results) as a
    LaTeX subscript instead of a literal underscore, e.g.
    "CBEXT-RPhCL--__F" -> r"\\texttt{CBEXT-RPhCL--}_{\\text{F}}". The result
    is math-mode LaTeX (needs surrounding "$...$") so it composes both in raw
    .tex table cells and in matplotlib usetex labels. Uses \\mathrm{} (core
    LaTeX) rather than \\text{} (amsmath) since matplotlib's default usetex
    preamble doesn't load amsmath.

    A "+"-joined pair of names (e.g. "SHORT_F+LONG_F", as used for the
    separate-fit row in plot_shortlong_vs_combined_latex) is split and each
    side formatted independently, rather than treating the whole string as
    one name -- otherwise only the trailing "_F"/"_NF" would be matched,
    leaving the first name's underscore literal inside \texttt{} (LaTeX:
    "Missing $ inserted").
    """
    if "+" in name:
        return "+".join(_texttt_with_condition_subscript(part) for part in name.split("+"))
    match = _CONDITION_TAG_RE.match(name)
    if match:
        base, tag = match.groups()
    else:
        base, tag = name, None
    # Display-only convention: the hazard position (index 8) uses a
    # three-way H/h/- code internally (H=free, h=fixed on, -=fixed off or
    # left at the schema default), but the paper only distinguishes free
    # (H) vs. fixed (-) in model *names* -- the actual True/False value is
    # shown separately in each table's own H column. This does not rename
    # any underlying file/config/directory, only how the name is displayed.
    # A bare "-" always means "defaults to True" (the case for 9 of 12
    # dash-hazard models); the three genuine hazard=False outliers get a
    # trailing prime instead, so an unmarked "-" means the same thing
    # everywhere in the paper.
    base, needs_prime = _hazard_display_transform(base)
    # Defensive: escape any underscore that didn't get consumed as a
    # condition tag, so a stray "_" never reaches \texttt{} as a literal
    # text-mode underscore (which errors under most LaTeX font encodings).
    base = base.replace("_", r"\_")
    if needs_prime:
        base += "'"
    if tag:
        return rf"\texttt{{{base}}}_{{\mathrm{{{tag}}}}}"
    return rf"\texttt{{{base}}}"


def export_best_models_macros(best_models: dict, path: str):
    """Write LaTeX \\newcommand macros for the winning model of each category
    in `best_models` (e.g. {"short": "SBFXT-RPHC---_F", "long": ...,
    "combined": ...} -- the same dict written to BIC/best_models.json by
    model_comparison.ipynb), so the paper can \\input{} this file once and
    refer to \\BestShortModel / \\BestLongModel / \\BestCombinedModel in
    prose instead of hand-typing model names that go stale whenever the
    comparison is rerun.

    Also writes a plain-text \\Best{Key}ModelName{} per category (the raw
    TASK string, with no $\texttt{}$/math-mode wrapping or F/NF subscript)
    so it can be used inside \\includegraphics paths, e.g.
    figures/POMDP/\\BestShortModelName/short/..., to pick up that
    category's figures automatically whenever the winning model changes.
    """
    lines = []
    for key, name in best_models.items():
        cap_key = key.capitalize()
        lines.append(
            rf"\newcommand{{\Best{cap_key}Model}}"
            rf"{{${_texttt_with_condition_subscript(name)}$}}"
        )
        lines.append(rf"\newcommand{{\Best{cap_key}ModelName}}{{{name}}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


colors = ["#3FD24B", "#E1D0D8", "#E92424"]
cmap = mcolors.LinearSegmentedColormap.from_list("RedGreyBlack", colors)


def _set_plot_style(font_size: int = 20, is_latex: bool = IS_LATEX):
    r"""Helper to set consistent plot styles.

    Font family is sans-serif (Arial/Helvetica) in both branches: PLOS figure
    guidelines (https://journals.plos.org/plosbiology/s/figures) require
    Arial, Helvetica, or Verdana for figure text rather than a serif/Computer
    Modern face, so `is_latex=True` still renders usetex-typeset text (e.g.
    math symbols) through Helvetica via the `helvet` package instead of
    LaTeX's default Computer Modern serif. `helvet` + `\familydefault` only
    swaps the *text* font, not the *math* font -- without `sfmath`, anything
    inside `$...$` (e.g. "$R^2$", "$\xi$") would still render in the default
    Computer Modern math italic, clashing with the surrounding sans-serif
    text.
    """
    if is_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}",
                "font.family": "sans-serif",
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
            }
        )


def plot_human_vs_sim_beta(
    human_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    beta_idx: int,
    beta_label_map: dict | None = None,
    is_latex: bool = False,
    font_size: int = 12,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
):
    """Scatter human vs. simulated per-subject GLM beta coefficients for one
    regressor (column f"beta{beta_idx}" in both DataFrames), with a linear
    fit line.

    Args:
        human_df: Per-subject human GLM betas, must contain "beta{beta_idx}".
        sim_df: Per-subject simulated GLM betas, same column requirement.
        beta_idx: Index used to build the "beta{beta_idx}" column name.
        beta_label_map: Optional {beta_idx: display label} override for the
            axis labels; falls back to the raw column name.
        is_latex: Passed to `_set_plot_style`.
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.

    Returns:
        matplotlib.figure.Figure, or None if the beta column is missing from
        either DataFrame or no overlapping non-NaN rows exist.
    """
    beta_col = f"beta{beta_idx}"
    beta_label = beta_label_map.get(beta_idx, beta_col) if beta_label_map else beta_col

    if beta_col not in human_df.columns or beta_col not in sim_df.columns:
        print(f"{beta_col} not found in both datasets")
        return

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    mask = human_df[beta_col].notna() & sim_df[beta_col].notna()

    x = human_df.loc[mask, beta_col]
    y = sim_df.loc[mask, beta_col]

    if len(x) == 0:
        print(f"No valid data for {beta_col}")
        return

    fig = plt.figure()

    plt.scatter(x, y, facecolors="white", edgecolors="k")

    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    plt.plot(xx, m * xx + b, color=(0.85, 0.325, 0.098))

    plt.xlabel(rf"Human Beta ${beta_label}$")
    plt.ylabel(rf"Simulated Beta ${beta_label}$")

    plt.tight_layout()

    print(rf"Simulated ${beta_label}$ ~ Human ${beta_label}$:", safe_spearman(x, y))

    if save_fig:
        _save_figure(fig, f"human_vs_sim_beta_{beta_idx}", path)

    return fig


def plot_best_actions_with_subject_choices(
    best_actions: np.ndarray,
    max_cards_per_draw: int,
    counts_dict: dict,
    label: str | None = None,
    path: str = DEFAULT_FIGURE_PATH,
    min_visits: int = 5,
    horizon_condition: str = "short",
    ytick_step: int = 5,
):
    """
    Plots empirical decide-vs-wait heatmap (PLOS CB style matched).

    Args:
        best_actions: Array of shape (num_draws, num_yellow, num_blue); only
            its shape is used, to determine the (draw, yellow-blue diff)
            grid extent.
        max_cards_per_draw: Max cards drawable per turn (used to map each
            draw count to its reachable yellow/blue index range).
        counts_dict: {(yellow, blue): "numerator/denominator"} empirical
            decision-fraction lookup.
        label: Optional label appended to the saved filename.
        path: Directory to save the figure into.
        min_visits: Minimum visit count (denominator) for a cell to be
            plotted; cells below this are masked out.
        horizon_condition: Used to build the saved filename.
        ytick_step: Spacing between labeled y-axis (yellow-blue diff) ticks.

    Returns:
        None.
    """

    num_draws, num_yellow, num_blue = best_actions.shape
    max_diff = num_yellow - 1
    min_diff = -max_diff

    diff_range = np.arange(min_diff, max_diff + 1)

    # --- Build empirical decision fraction array ---
    empirical = np.full((num_draws, len(diff_range)), np.nan)
    visit_mask = np.ones((num_draws, len(diff_range)), dtype=bool)

    for draw in range(num_draws):
        for yellow in range(num_yellow):

            blue = draw * max_cards_per_draw - yellow

            if blue < 0 or blue >= num_blue:
                continue

            diff = yellow - blue

            if min_diff <= diff <= max_diff:
                diff_index = diff - min_diff
                key = (float(yellow), float(blue))

                if key in counts_dict:
                    val = counts_dict[key]
                    numerator, denominator = val.split("/")
                    numerator = int(numerator)
                    denominator = int(denominator)

                    if denominator >= min_visits:
                        empirical[draw, diff_index] = numerator / denominator
                        visit_mask[draw, diff_index] = False

    # Skip draw 0 (match normative function)
    empirical = empirical[1:, :]
    visit_mask = visit_mask[1:, :]
    empirical = np.ma.array(empirical, mask=visit_mask)

    # --- Style (MATCHED) ---
    _set_plot_style()

    empirical_cmap = plt.cm.RdYlBu

    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()

    im = ax.imshow(
        empirical.T,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=1,
    )

    # --- Labels (same style as normative) ---
    ax.set_xlabel("Number of Draws")
    ax.set_ylabel("Yellow - Blue Difference")

    # --- X ticks (IDENTICAL STYLE) ---
    num_draws_plot = empirical.shape[0]
    ax.set_xticks(np.arange(0, num_draws_plot, 1))
    ax.set_xticklabels(np.arange(1, num_draws_plot + 1, 1))

    # --- Y ticks ---
    indices_step_5 = [i for i, val in enumerate(diff_range) if val % ytick_step == 0]

    ax.set_yticks(indices_step_5)
    ax.set_yticklabels(diff_range[indices_step_5])

    # --- Grid (IDENTICAL) ---
    ax.grid(which="major", color="w", linestyle="-", linewidth=0.5)

    # --- Colorbar (PLOS clean style) ---
    cbar = fig.colorbar(im, ticks=np.linspace(0, 1, 3))
    cbar.set_label("Decision Fraction")

    fig.tight_layout()

    # --- Save ---
    fname = f"decision_policy_heatmap_{horizon_condition}"

    if label is not None:
        fname += f"_{label}"

    _save_figure(fig, fname, path=path)


def _build_yellow_blue_arrays(
    best_actions,
    counts_dict_yellow,
    counts_dict_blue,
    n_draws_out,
    ytick_step,
    max_cards_per_draw,
    min_visits,
):
    """Build masked fraction/numerator arrays for one horizon's panel row of
    `plot_yellow_blue_fractions_stacked` / `plot_yellow_blue_fractions_separate`."""
    num_draws, num_yellow, num_blue = best_actions.shape
    max_diff = num_yellow - 1
    min_diff = -max_diff
    diff_range = np.arange(min_diff, max_diff + 1)
    n_diff = len(diff_range)

    f_wait = np.full((n_draws_out, n_diff), np.nan)
    f_yellow = np.full((n_draws_out, n_diff), np.nan)
    f_blue = np.full((n_draws_out, n_diff), np.nan)
    n_wait_arr = np.zeros((n_draws_out, n_diff))
    n_yellow_arr = np.zeros((n_draws_out, n_diff))
    n_blue_arr = np.zeros((n_draws_out, n_diff))
    mask = np.ones((n_draws_out, n_diff), dtype=bool)
    # States the task cannot reach: after d draws only d * max_cards_per_draw
    # cards have been seen, so |n_y - n_b| cannot exceed that. Blanking these
    # keeps them distinct from states that were reachable but never visited.
    impossible = np.zeros((n_draws_out, n_diff), dtype=bool)
    for d_out in range(n_draws_out):
        total = (d_out + 1) * max_cards_per_draw
        # unreachable on two counts: |n_y - n_b| cannot exceed the number of
        # cards seen, and since n_y + n_b = total the difference must share the
        # parity of that total, so alternate rows are impossible as well
        impossible[d_out, :] = (np.abs(diff_range) > total) | (
            (np.abs(diff_range - total) % 2) != 0
        )

    for draw in range(1, num_draws):
        if draw - 1 >= n_draws_out:
            break
        for yellow in range(num_yellow):
            blue = draw * max_cards_per_draw - yellow
            if blue < 0 or blue >= num_blue:
                continue
            diff = yellow - blue
            if not (min_diff <= diff <= max_diff):
                continue
            key = (float(yellow), float(blue))
            if key not in counts_dict_yellow or key not in counts_dict_blue:
                continue
            n_y, denom = counts_dict_yellow[key].split("/")
            n_b = counts_dict_blue[key].split("/")[0]
            denom, n_y, n_b = int(denom), int(n_y), int(n_b)
            if denom < min_visits:
                continue
            col_idx = diff - min_diff
            f_y = n_y / denom
            f_b = n_b / denom
            n_w = max(0, denom - n_y - n_b)
            f_yellow[draw - 1, col_idx] = f_y
            f_blue[draw - 1, col_idx] = f_b
            f_wait[draw - 1, col_idx] = max(0.0, 1.0 - f_y - f_b)
            n_yellow_arr[draw - 1, col_idx] = n_y
            n_blue_arr[draw - 1, col_idx] = n_b
            n_wait_arr[draw - 1, col_idx] = n_w
            mask[draw - 1, col_idx] = False

    indices = [i for i, v in enumerate(diff_range) if v % ytick_step == 0]
    # numerator arrays in the same order as the fraction panels: wait, yellow, blue
    numerator_arrays = [n_wait_arr, n_yellow_arr, n_blue_arr]
    return (
        [
            np.ma.array(f_wait, mask=mask),
            np.ma.array(f_yellow, mask=mask),
            np.ma.array(f_blue, mask=mask),
        ],
        diff_range,
        indices,
        n_diff,
        numerator_arrays,
        mask,
        impossible,
    )


def _resolve_yellow_blue_colors(dark, bg_color):
    """Resolve figure/panel/text/line colors for the yellow/blue heatmap plots.

    For ``dark=True`` these are pulled from matplotlib's built-in
    "dark_background" style sheet rather than hand-picked hex values, so the
    dark variant automatically uses whatever foreground/background contrast
    matplotlib considers legible, instead of us re-deriving it by hand.
    """
    if dark:
        style = plt.style.library["dark_background"]
        fig_color = style["figure.facecolor"]
        text_color = style["text.color"]
        line_color = style["axes.edgecolor"]
        panel_color = bg_color if bg_color is not None else style["axes.facecolor"]
    else:
        fig_color = "white"
        text_color = "black"
        line_color = "black"
        # clearly darker than the white "visited" cells, so a state nobody
        # reached is distinguishable from one where this action was never chosen
        panel_color = bg_color if bg_color is not None else "#c9c9c9"
    shade_color = "#888888" if dark else "#b3b3b3"
    zero_color = panel_color if dark else "white"
    return fig_color, panel_color, text_color, line_color, shade_color, zero_color


def _draw_yellow_blue_panel(
    fig,
    ax,
    frac_arr,
    num_arr,
    mask,
    diff_range,
    indices,
    n_diff,
    n_draws,
    ylim,
    cmap_p,
    norm,
    title,
    cbar_lbl,
    text_color,
    line_color,
    BG_COLOR,
    show_xlabel,
    xtick_pos,
    xtick_labels,
    shade_region: tuple[float, float] | None = None,
    shade_color: str | None = None,
    impossible=None,
    visited_color: str = "white",
    impossible_color: str = "white",
):
    """Draw one wait/yellow/blue heatmap panel (rectangles + colorbar + axes
    decorations) onto `ax`. Shared by the stacked multi-panel figure and the
    one-figure-per-panel variant so both stay in sync.

    Three cell states are drawn differently, because a bar whose height encodes
    a count of zero is invisible and would otherwise be indistinguishable from a
    state nobody ever reached:
      unreachable      painted `impossible_color`, blanked out
      never visited    left as the panel background
      visited          painted `visited_color`, with the coloured bar on top
    """
    ax.set_facecolor(BG_COLOR)

    # blank the states the task cannot produce
    if impossible is not None:
        for d in range(n_draws):
            for j in range(n_diff):
                if impossible[d, j]:
                    # stroked in its own colour: neighbouring patches are
                    # anti-aliased independently, which otherwise leaves a
                    # hairline seam between adjacent cells
                    ax.add_patch(Rectangle((d - 0.5, j - 0.5), 1.0, 1.0,
                                           facecolor=impossible_color,
                                           edgecolor=impossible_color,
                                           linewidth=0.6, snap=True,
                                           antialiased=False, zorder=0.5))

    # every visited state gets a full cell, so "action never chosen" reads as an
    # empty cell rather than as the background
    for d in range(n_draws):
        for j in range(n_diff):
            if not mask[d, j]:
                ax.add_patch(Rectangle((d - 0.5, j - 0.5), 1.0, 1.0,
                                       facecolor=visited_color,
                                       edgecolor=visited_color,
                                       linewidth=0.6, snap=True,
                                       antialiased=False, zorder=0.8))

    # Normalise height by the maximum numerator count within this panel
    # (wait / yellow / blue independently), so the busiest cell = full height
    max_num = num_arr.max() if num_arr.max() > 0 else 1.0

    for d in range(n_draws):
        for j in range(n_diff):
            if mask[d, j]:
                continue
            frac = float(frac_arr[d, j])
            h = num_arr[d, j] / max_num  # normalised height [0..1]
            color = cmap_p(norm(frac))
            rect = Rectangle(
                (d - 0.5, j - h / 2),  # bottom-left corner
                1.0,  # width (always full column)
                h,  # variable height
                facecolor=color,
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(rect)

    ax.set_ylabel("Yellow − Blue", color=text_color)
    ax.set_title(title, color=text_color)
    ax.set_xlim(-0.5, n_draws - 0.5)
    ax.set_ylim(-0.5, n_diff - 0.5)

    ax.set_yticks(indices)
    ax.set_yticklabels(diff_range[indices])
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    # Colorbar via ScalarMappable (no imshow object)
    sm = ScalarMappable(cmap=cmap_p, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[0, 0.5, 1])
    cbar.set_label(cbar_lbl, color=text_color)
    cbar.ax.tick_params(colors=text_color)
    cbar.outline.set_edgecolor(text_color)

    # Horizontal line at Yellow − Blue = 0
    zero_idx = list(diff_range).index(0)
    ax.axhline(
        y=zero_idx,
        color=line_color,
        linestyle="-",
        linewidth=1.2,
        alpha=0.8,
        zorder=4,
    )

    # Y-axis limits
    y_lo = zero_idx - ylim
    y_hi = zero_idx + ylim
    ax.set_ylim(max(y_lo, -0.5), min(y_hi, n_diff - 0.5))

    # X-axis ticks
    ax.set_xticks(xtick_pos)
    if show_xlabel:
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Number of Draws", color=text_color)
    else:
        ax.set_xticklabels([])

    # Shade columns beyond the short-horizon deadline
    if shade_region is not None:
        x_lo, x_hi = shade_region
        # opaque, so draws past the end of the horizon carry exactly the same
        # grey as the unreachable wedge: both are states the task cannot produce
        ax.axvspan(x_lo, x_hi, color=shade_color, alpha=1.0, zorder=2)
        ax.axvline(
            x=x_lo,
            color=line_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )


def plot_yellow_blue_fractions_stacked(
    best_actions_short,
    best_actions_long,
    max_cards_per_draw,
    counts_dict_yellow_short,
    counts_dict_blue_short,
    counts_dict_yellow_long,
    counts_dict_blue_long,
    label: str | None = None,
    path: str = DEFAULT_FIGURE_PATH,
    min_visits: int = 0,
    ytick_step_short: int = 5,
    ytick_step_long: int = 10,
    ylim_short: int = 25,
    ylim_long: int = 40,
    bg_color: str | None = None,
    dark: bool = False,
    font_size: int = 20,
):
    """
    Six-panel heatmap: short horizon (top row) and long horizon (bottom row),
    each row showing fraction-wait / fraction-yellow / fraction-blue.

    The x-axis (number of draws) is shared across rows; the short horizon is
    padded with masked columns to match the long-horizon draw range so the
    shared axis is meaningful.

    Figure background is light grey to visually separate masked (white) cells
    from the figure margins. Pass ``dark=True`` for a dark-background variant
    (figure/panel backgrounds, text, and gridlines flipped for a dark
    Beamer/Overleaf slide), and the zero-value end of each colormap is
    matched to the panel background so empty cells blend in.
    """

    _set_plot_style(font_size=font_size)

    fig_color, BG_COLOR, text_color, line_color, shade_color, zero_color = (
        _resolve_yellow_blue_colors(dark, bg_color)
    )

    cmap_wait = LinearSegmentedColormap.from_list("wait", [zero_color, "#1a8c1a"])
    # the yellow-choice panel is drawn in red: a yellow ramp from a white or
    # light-grey zero point is nearly invisible at low fractions
    cmap_yellow = LinearSegmentedColormap.from_list("yellow", [zero_color, "#E92424"])
    cmap_blue = LinearSegmentedColormap.from_list("blue", [zero_color, "#1f5fe0"])
    cmaps = [cmap_wait, cmap_yellow, cmap_blue]
    titles = ["Fraction Wait", "Fraction Yellow", "Fraction Blue"]
    cbar_labels = ["Wait", "Fraction", "Fraction"]

    # Use long-horizon draw count as the shared x dimension
    n_draws_long = best_actions_long.shape[0] - 1
    n_draws_short_raw = best_actions_short.shape[0] - 1

    arrays_s, diff_range_s, idx_s, n_diff_s, num_s, mask_s, imp_s = _build_yellow_blue_arrays(
        best_actions_short,
        counts_dict_yellow_short,
        counts_dict_blue_short,
        n_draws_long,
        ytick_step_short,
        max_cards_per_draw,
        min_visits,
    )
    arrays_l, diff_range_l, idx_l, n_diff_l, num_l, mask_l, imp_l = _build_yellow_blue_arrays(
        best_actions_long,
        counts_dict_yellow_long,
        counts_dict_blue_long,
        n_draws_long,
        ytick_step_long,
        max_cards_per_draw,
        min_visits,
    )

    fig, axes = plt.subplots(2, 3, figsize=(30, 14), sharex=True)
    fig.patch.set_facecolor(fig_color)

    xtick_pos = np.arange(0, n_draws_long)
    xtick_labels = np.arange(1, n_draws_long + 1)

    horizon_configs = [
        (
            arrays_s,
            diff_range_s,
            idx_s,
            n_diff_s,
            "Short",
            ylim_short,
            num_s,
            mask_s,
            imp_s,
        ),
        (
            arrays_l,
            diff_range_l,
            idx_l,
            n_diff_l,
            "Long",
            ylim_long,
            num_l,
            mask_l,
            imp_l,
        ),
    ]

    norm = Normalize(vmin=0, vmax=1)

    for col, (cmap_p, title, cbar_lbl) in enumerate(zip(cmaps, titles, cbar_labels)):
        for row, (
            arrays,
            diff_range,
            indices,
            n_diff,
            horizon,
            ylim,
            num_arrays,
            mask,
            impossible,
        ) in enumerate(horizon_configs):
            ax = axes[row][col]
            shade_region = (
                (n_draws_short_raw - 0.5, n_draws_long - 0.5) if row == 0 else None
            )
            _draw_yellow_blue_panel(
                fig,
                ax,
                arrays[col],
                num_arrays[col],
                mask,
                diff_range,
                indices,
                n_diff,
                n_draws_long,
                ylim,
                cmap_p,
                norm,
                f"{title} ({horizon} Horizon)",
                cbar_lbl,
                text_color,
                line_color,
                BG_COLOR,
                show_xlabel=(row == 1),
                xtick_pos=xtick_pos,
                xtick_labels=xtick_labels,
                shade_region=shade_region,
                shade_color=shade_color,
                impossible=impossible,
                impossible_color=shade_color,
            )

    fig.tight_layout()

    fname = "yellow_blue_fractions_stacked"
    if label is not None:
        fname += f"_{label}"
    if dark:
        fname += "_dark"
    _save_figure(fig, fname, path=path)


def plot_yellow_blue_fractions_separate(
    best_actions_short,
    best_actions_long,
    max_cards_per_draw,
    counts_dict_yellow_short,
    counts_dict_blue_short,
    counts_dict_yellow_long,
    counts_dict_blue_long,
    label: str | None = None,
    path: str = DEFAULT_FIGURE_PATH,
    min_visits: int = 0,
    ytick_step_short: int = 5,
    ytick_step_long: int = 10,
    ylim_short: int = 25,
    ylim_long: int = 40,
    bg_color: str | None = None,
    dark: bool = False,
    font_size: int = 26,
    figsize: tuple[float, float] = (10, 8),
):
    """
    Same six wait/yellow/blue heatmap panels as `plot_yellow_blue_fractions_stacked`,
    but each panel (horizon x quantity) is saved as its own single-panel figure
    rather than as a combined 2x3 grid, with a larger default font size for
    use as standalone slide/poster figures.
    """

    _set_plot_style(font_size=font_size)

    fig_color, BG_COLOR, text_color, line_color, shade_color, zero_color = (
        _resolve_yellow_blue_colors(dark, bg_color)
    )

    cmap_wait = LinearSegmentedColormap.from_list("wait", [zero_color, "#1a8c1a"])
    cmap_yellow = LinearSegmentedColormap.from_list("yellow", [zero_color, "#DCE925"])
    cmap_blue = LinearSegmentedColormap.from_list("blue", [zero_color, "#1f5fe0"])
    cmaps = [cmap_wait, cmap_yellow, cmap_blue]
    titles = ["Fraction Wait", "Fraction Yellow", "Fraction Blue"]
    cbar_labels = ["Wait", "Fraction", "Fraction"]
    keys = ["wait", "yellow", "blue"]

    # Use long-horizon draw count as the shared x dimension
    n_draws_long = best_actions_long.shape[0] - 1
    n_draws_short_raw = best_actions_short.shape[0] - 1

    arrays_s, diff_range_s, idx_s, n_diff_s, num_s, mask_s, imp_s = _build_yellow_blue_arrays(
        best_actions_short,
        counts_dict_yellow_short,
        counts_dict_blue_short,
        n_draws_long,
        ytick_step_short,
        max_cards_per_draw,
        min_visits,
    )
    arrays_l, diff_range_l, idx_l, n_diff_l, num_l, mask_l, imp_l = _build_yellow_blue_arrays(
        best_actions_long,
        counts_dict_yellow_long,
        counts_dict_blue_long,
        n_draws_long,
        ytick_step_long,
        max_cards_per_draw,
        min_visits,
    )

    xtick_pos = np.arange(0, n_draws_long)
    xtick_labels = np.arange(1, n_draws_long + 1)

    horizon_configs = [
        (
            arrays_s,
            diff_range_s,
            idx_s,
            n_diff_s,
            "short",
            "Short",
            ylim_short,
            num_s,
            mask_s,
            imp_s,
            (n_draws_short_raw - 0.5, n_draws_long - 0.5),
        ),
        (
            arrays_l,
            diff_range_l,
            idx_l,
            n_diff_l,
            "long",
            "Long",
            ylim_long,
            num_l,
            mask_l,
            imp_l,
            None,
        ),
    ]

    norm = Normalize(vmin=0, vmax=1)

    for col, (cmap_p, title, cbar_lbl, key) in enumerate(
        zip(cmaps, titles, cbar_labels, keys)
    ):
        for (
            arrays,
            diff_range,
            indices,
            n_diff,
            horizon_key,
            horizon_label,
            ylim,
            num_arrays,
            mask,
            impossible,
            shade_region,
        ) in horizon_configs:
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor(fig_color)

            _draw_yellow_blue_panel(
                fig,
                ax,
                arrays[col],
                num_arrays[col],
                mask,
                diff_range,
                indices,
                n_diff,
                n_draws_long,
                ylim,
                cmap_p,
                norm,
                f"{title} ({horizon_label} Horizon)",
                cbar_lbl,
                text_color,
                line_color,
                BG_COLOR,
                show_xlabel=True,
                xtick_pos=xtick_pos,
                xtick_labels=xtick_labels,
                shade_region=shade_region,
                shade_color=shade_color,
                impossible=impossible,
                impossible_color=shade_color,
            )

            fig.tight_layout()

            fname = f"yellow_blue_fractions_{horizon_key}_{key}"
            if label is not None:
                fname += f"_{label}"
            if dark:
                fname += "_dark"
            _save_figure(fig, fname, path=path)
            plt.close(fig)


def plot_beta_correlations(
    betas_ocir,
    betas_allq,
    beta_col: str = "beta2",
    ocir_col: str = "FA2",
    ybocs_col: str = "YBOCS_obsess_subtotal",
    is_latex: bool = False,
    font_size: int = 20,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
):
    """
    Plot beta correlations with OCIR factor and YBOCS score.
    """
    if beta_col == "beta2":
        y_label = r"Beta $\Delta ES_t$"
    elif beta_col == "beta1":
        y_label = r"Beta $ES_{t-1}$"

    # ---------- OCIR ----------
    if beta_col in betas_ocir.columns and ocir_col in betas_ocir.columns:

        mask = betas_ocir[ocir_col].notna() & betas_ocir[beta_col].notna()

        plot_beta_with_significance(
            betas_ocir.loc[mask, ocir_col],
            betas_ocir.loc[mask, beta_col],
            "OC Factor",
            y_label,
            is_latex=is_latex,
            font_size=font_size,
            path=path,
            save_fig=save_fig,
            fname=f"beta_correlations_{beta_col}_ocir",
        )

    # ---------- YBOCS ----------
    if beta_col in betas_allq.columns and ybocs_col in betas_allq.columns:

        mask = betas_allq[ybocs_col].notna() & betas_allq[beta_col].notna()

        plot_beta_with_significance(
            betas_allq.loc[mask, ybocs_col],
            betas_allq.loc[mask, beta_col],
            "YBOCS obsessions subtotal",
            y_label,
            is_latex=is_latex,
            font_size=font_size,
            path=path,
            save_fig=save_fig,
            fname=f"beta_correlations_{beta_col}_ybocs",
        )


def plot_glmm_betas(
    glm_results,
    main_effects: list[str] | None = None,
    interaction_effects: list[str] | None = None,
    is_latex: bool = IS_LATEX,
    font_size: int = 20,
    main_ylim: tuple[float, float] = (-2.5, 2.5),
    interaction_ylim: tuple[float, float] = (-0.2, 0.2),
    figsize: tuple[float, float] = (10, 7),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "glmm_betas",
):
    """Plot fitted GLM main-effect and FA2-interaction beta coefficients
    (with 95% CI error bars and significance stars) on twin y-axes.

    Args:
        glm_results: Fitted statsmodels GLM results object (exposes
            .params/.bse/.pvalues).
        main_effects: Regressor names for the main-effects axis; defaults to
            ["totevminus", "deltaev", "trial", "termination"].
        interaction_effects: Regressor names for the FA2-interaction axis;
            defaults to the "FA2:"-prefixed counterparts of `main_effects`.
        is_latex: Passed to `_set_plot_style`.
        font_size: Base font size for the plot.
        main_ylim: Y-limits for the main-effects axis.
        interaction_ylim: Y-limits for the interaction axis.
        figsize: Figure size.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename (without extension) used when saving.

    Returns:
        tuple: (fig, (ax, ax2)) -- the figure and its two y-axes.
    """
    if main_effects is None:
        main_effects = ["totevminus", "deltaev", "trial", "termination"]

    if interaction_effects is None:
        interaction_effects = [
            "FA2:totevminus",
            "FA2:deltaev",
            "FA2:trial",
            "FA2:termination",
        ]

    params = glm_results.params
    ses = glm_results.bse
    pvals = glm_results.pvalues  # Use pre-computed p-values

    # --- Extract coefficients ---
    est_main = params.loc[main_effects]
    se_main = ses.loc[main_effects]
    err_main = 1.96 * se_main
    p_main = pvals.loc[main_effects]

    est_inter = params.loc[interaction_effects]
    se_inter = ses.loc[interaction_effects]
    err_inter = 1.96 * se_inter
    p_inter = pvals.loc[interaction_effects]

    def p_to_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # --- LaTeX formatting ---
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)

    # x positions
    x_main = np.arange(len(main_effects))
    x_inter = np.arange(len(main_effects), len(main_effects) + len(interaction_effects))

    # --- Main effects ---
    ax.errorbar(
        x_main,
        est_main,
        yerr=err_main,
        fmt="o",
        color="k",
        elinewidth=2,
        capsize=5,
        markersize=8,
        label="Main Effects",
    )

    # --- Interaction effects ---
    ax2 = ax.twinx()
    ax2.errorbar(
        x_inter,
        est_inter,
        yerr=err_inter,
        fmt="o",
        color="b",
        elinewidth=2,
        capsize=5,
        markersize=8,
        label="FA2 Interactions",
    )

    # zero line
    ax.axhline(0, linestyle="--", linewidth=1.25)

    # --- Add significance stars ---
    main_offset = (main_ylim[1] - main_ylim[0]) * 0.03
    inter_offset = (interaction_ylim[1] - interaction_ylim[0]) * 0.03

    # Main effects stars
    for i, (x, beta, err, p) in enumerate(zip(x_main, est_main, err_main, p_main)):
        stars = p_to_stars(p)
        if stars:
            y = beta + err + main_offset
            ax.text(
                x,
                y,
                stars,
                ha="center",
                va="bottom",
                color="k",
                fontsize=font_size * 0.8,
            )

    # Interaction effects stars
    for i, (x, beta, err, p) in enumerate(zip(x_inter, est_inter, err_inter, p_inter)):
        stars = p_to_stars(p)
        if stars:
            y = beta + err + inter_offset
            ax2.text(
                x,
                y,
                stars,
                ha="center",
                va="bottom",
                color="b",
                fontsize=font_size * 0.8,
            )

    # --- Axis styling ---
    ax.set_ylabel("Beta Estimate (Main Effects)", color="k")
    ax.set_ylim(main_ylim)
    ax.tick_params(axis="y", colors="k")

    ax2.set_ylabel("Beta Estimate (FA2 Interactions)", color="b")
    ax2.set_ylim(interaction_ylim)
    ax2.tick_params(axis="y", colors="b")

    # --- Labels ---
    all_names = main_effects + [x.replace("FA2:", "OC:") for x in interaction_effects]

    def latexify(name: str) -> str:
        if "deltaev" in name:
            return name.replace("deltaev", r"$\Delta ES_t$")
        if "totevminus" in name:
            return name.replace("totevminus", r"$ES_{t-1}$")
        if "trial" in name:
            return name.replace("trial", r"$draw\_seq$")
        return name

    ax.set_xticks(np.concatenate([x_main, x_inter]))
    ax.set_xticklabels([latexify(n) for n in all_names], rotation=45, ha="right")

    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, (ax, ax2)


def plot_averaged_glmm_betas(
    glmm_summary_df: pd.DataFrame,
    main_effects: list[str] | None = None,
    interaction_effects: list[str] | None = None,
    is_latex: bool = False,
    font_size: int = 20,
    main_ylim: tuple[float, float] = (-2.5, 2.5),
    interaction_ylim: tuple[float, float] = (-0.2, 0.2),
    figsize: tuple[float, float] = (10, 7),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "averaged_glmm_betas",
):
    """Plot ensemble-averaged GLMM main-effect and FA2-interaction beta
    coefficients (with 95% CI error bars and significance stars from a
    Z-test on Mean_Estimate/Mean_SE) on twin y-axes.

    Args:
        glmm_summary_df: DataFrame indexed by predictor name with columns
            "Mean_Estimate" and "Mean_SE" (e.g. a glmm_estimates_*.pkl
            summary, see notebooks/glm_multiprocessed_*.py).
        main_effects: Regressor names for the main-effects axis; defaults to
            ["totevminus", "deltaev", "trial", "termination"].
        interaction_effects: Regressor names for the FA2-interaction axis;
            defaults to the "FA2:"-prefixed counterparts of `main_effects`.
        is_latex: Passed to `_set_plot_style`.
        font_size: Base font size for the plot.
        main_ylim: Y-limits for the main-effects axis.
        interaction_ylim: Y-limits for the interaction axis.
        figsize: Figure size.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename (without extension) used when saving.

    Returns:
        tuple: (fig, (ax, ax2)) -- the figure and its two y-axes.
    """
    if main_effects is None:
        main_effects = ["totevminus", "deltaev", "trial", "termination"]

    if interaction_effects is None:
        interaction_effects = [
            "FA2:totevminus",
            "FA2:deltaev",
            "FA2:trial",
            "FA2:termination",
        ]

    # --- Extract coefficients and calculate p-values ---
    params = glmm_summary_df["Mean_Estimate"]
    ses = glmm_summary_df["Mean_SE"]

    # Calculate pseudo p-values from Z-scores (two-tailed)
    z_scores = params / ses
    pvals = pd.Series(stats.norm.sf(abs(z_scores)) * 2, index=params.index)

    est_main = params.loc[main_effects]
    se_main = ses.loc[main_effects]
    err_main = 1.96 * se_main
    p_main = pvals.loc[main_effects]

    est_inter = params.loc[interaction_effects]
    se_inter = ses.loc[interaction_effects]
    err_inter = 1.96 * se_inter
    p_inter = pvals.loc[interaction_effects]

    def p_to_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # --- Setup plot ---
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)

    # x positions
    x_main = np.arange(len(main_effects))
    x_inter = np.arange(len(main_effects), len(main_effects) + len(interaction_effects))

    # --- Main effects ---
    ax.errorbar(
        x_main,
        est_main,
        yerr=err_main,
        fmt="o",
        color="k",
        elinewidth=2,
        capsize=5,
        markersize=8,
        label="Main Effects",
    )

    # --- Interaction effects ---
    ax2 = ax.twinx()
    ax2.errorbar(
        x_inter,
        est_inter,
        yerr=err_inter,
        fmt="o",
        color="b",
        elinewidth=2,
        capsize=5,
        markersize=8,
        label="FA2 Interactions",
    )

    # zero line
    ax.axhline(0, linestyle="--", linewidth=1.25, color="gray")

    # --- Add significance stars ---
    main_offset = (main_ylim[1] - main_ylim[0]) * 0.03
    inter_offset = (interaction_ylim[1] - interaction_ylim[0]) * 0.03

    # Main effects stars
    for x, beta, err, p in zip(x_main, est_main, err_main, p_main):
        stars = p_to_stars(p)
        if stars:
            y = beta + err + main_offset
            ax.text(
                x,
                y,
                stars,
                ha="center",
                va="bottom",
                color="k",
                fontsize=font_size * 0.8,
            )

    # Interaction effects stars
    for x, beta, err, p in zip(x_inter, est_inter, err_inter, p_inter):
        stars = p_to_stars(p)
        if stars:
            y = beta + err + inter_offset
            ax2.text(
                x,
                y,
                stars,
                ha="center",
                va="bottom",
                color="b",
                fontsize=font_size * 0.8,
            )

    # --- Axis styling ---
    ax.set_ylabel("Beta Estimate (Main Effects)", color="k", fontsize=font_size)
    ax.set_ylim(main_ylim)
    ax.tick_params(axis="y", colors="k", labelsize=font_size * 0.8)

    ax2.set_ylabel("Beta Estimate (FA2 Interactions)", color="b", fontsize=font_size)
    ax2.set_ylim(interaction_ylim)
    ax2.tick_params(axis="y", colors="b", labelsize=font_size * 0.8)

    # --- Labels ---
    all_names = main_effects + [x.replace("FA2:", "OC:") for x in interaction_effects]

    def latexify(name: str) -> str:
        if "deltaev" in name:
            return name.replace("deltaev", r"$\Delta ES_t$")
        if "totevminus" in name:
            return name.replace("totevminus", r"$ES_{t-1}$")
        if "trial" in name:
            return name.replace("trial", r"$draw\_seq$")
        return name

    ax.set_xticks(np.concatenate([x_main, x_inter]))
    ax.set_xticklabels(
        [latexify(n) for n in all_names],
        rotation=45,
        ha="right",
        fontsize=font_size * 0.9,
    )

    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, (ax, ax2)


def plot_glmm_betas_comparison(
    human_estimates,
    model_estimates,
    model_label: str = "Model",
    main_effects: list[str] | None = None,
    interaction_effects: list[str] | None = None,
    is_latex: bool = IS_LATEX,
    font_size: int = 20,
    main_ylim: tuple[float, float] = (-2.5, 2.5),
    interaction_ylim: tuple[float, float] = (-0.2, 0.2),
    figsize: tuple[float, float] = (10, 6),
    horizon: str = "",
    condition_label: str | None = None,
    path: str = "../GLM",
    save_fig: bool = True,
    fname: str | None = None,
):
    """Overlay human ("true") and model (simulated, ensemble-averaged) GLMM
    coefficients on the same axes, with a difference panel below showing
    model - human per regressor (z-test on the difference, pooled SE).

    human_estimates, model_estimates : statsmodels GLM result (exposing
        .params/.bse) or a DataFrame with "Mean_Estimate"/"Mean_SE" columns
        indexed by predictor name -- i.e. either glm_results from fit_glmm()
        or a glmm_estimates_*.pkl summary (see compute_human_glmm.py /
        notebooks/glm_multiprocessed_*.py).
    horizon : used only to keep saved filenames unique (e.g. when one model
        has several GLM fits paired against different short configs).
    condition_label : human-readable text shown in the title instead of
        `horizon` (e.g. "long + short combined"). Falls back to `horizon`
        if not given.
    fname : filename override (without extension). Since `model_label` is
        not part of the default `horizon`-derived filename, callers that
        save multiple models into the *same* `path` (e.g.
        notebooks/export_glmm_comparison.py writing every model into
        ../GLM) must pass a model-qualified fname themselves, or different
        models sharing a `horizon` string will silently overwrite each
        other's figure.
    """
    if main_effects is None:
        main_effects = ["totevminus", "deltaev", "trial", "termination"]
    if interaction_effects is None:
        interaction_effects = [
            "FA2:totevminus",
            "FA2:deltaev",
            "FA2:trial",
            "FA2:termination",
        ]
    all_predictors = main_effects + interaction_effects

    def _extract(obj):
        if hasattr(obj, "params") and hasattr(obj, "bse"):
            return obj.params.loc[all_predictors], obj.bse.loc[all_predictors]
        return (
            obj.loc[all_predictors, "Mean_Estimate"],
            obj.loc[all_predictors, "Mean_SE"],
        )

    est_h, se_h = _extract(human_estimates)
    est_m, se_m = _extract(model_estimates)

    def _dash_safe(text):
        # Under real usetex rendering (is_latex=True), plain "--"/"---" in
        # config names (e.g. "LB-XT-RPHCLUK", "SBEXT-RPHC---") get ligated
        # into en-/em-dashes; \texttt{} keeps them literal (and renders a
        # trailing "_F"/"_NF" condition tag as a subscript). Skip when
        # is_latex=False (mathtext/plain text) since matplotlib would
        # otherwise print the literal "\texttt{...}" characters -- but still
        # apply the plain-text h->'-' display transform either way.
        return f"${_texttt_with_condition_subscript(text)}$" if is_latex else _display_task_name(text)

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    x_main = np.arange(len(main_effects))
    x_inter = np.arange(len(main_effects), len(all_predictors))
    x_all = np.concatenate([x_main, x_inter])
    dodge = 0.12
    # Color encodes data source (human vs. model) and stays fixed across both
    # axes; marker shape encodes which axis/scale a point belongs to (circle
    # = main effect / left axis, square = FA2 interaction / right axis).
    # Previously the right axis's ticks/label were colored with color_model,
    # which made the right axis look like it "belonged" to the model series
    # specifically -- misleading, since both human and model points appear on
    # it. The axis is now a neutral gray instead, and marker shape (not
    # color) signals the axis-group.
    color_human = "k"
    color_model = "#2166AC"
    axis2_color = "0.35"
    marker_main = "o"
    marker_inter = "s"

    fig, ax_top = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax_top.twinx()

    def p_to_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    def _plot_group(ax, x, est, se, color, label, offset, ylim, marker: str = "o"):
        ax.errorbar(
            x + offset,
            est,
            yerr=1.96 * se,
            fmt=marker,
            color=color,
            elinewidth=2,
            capsize=4,
            markersize=7,
            label=label,
        )
        # per-dot significance (each estimate vs. 0), annotated above/below
        # its own error bar, with a small gap (relative to the axis range)
        # so the stars don't sit right on top of the error bar caps.
        pad = 0.04 * (ylim[1] - ylim[0])
        z_dot = est / se
        p_dot = pd.Series(stats.norm.sf(np.abs(z_dot)) * 2, index=est.index)
        for xi, name in zip(x, est.index):
            stars = p_to_stars(p_dot[name])
            if not stars:
                continue
            e, val = est[name], se[name]
            top = e + 1.96 * val
            bottom = e - 1.96 * val
            y = top + pad if e >= 0 else bottom - pad
            va = "bottom" if e >= 0 else "top"
            ax.text(
                xi + offset,
                y,
                stars,
                ha="center",
                va=va,
                fontsize=font_size * 0.6,
                color=color,
            )

    _plot_group(
        ax_top,
        x_main,
        est_h.loc[main_effects],
        se_h.loc[main_effects],
        color_human,
        "Human (main effect)",
        -dodge,
        main_ylim,
        marker=marker_main,
    )
    _plot_group(
        ax_top,
        x_main,
        est_m.loc[main_effects],
        se_m.loc[main_effects],
        color_model,
        f"{_dash_safe(model_label)} (main effect)",
        dodge,
        main_ylim,
        marker=marker_main,
    )
    _plot_group(
        ax2,
        x_inter,
        est_h.loc[interaction_effects],
        se_h.loc[interaction_effects],
        color_human,
        "Human (FA2 interaction)",
        -dodge,
        interaction_ylim,
        marker=marker_inter,
    )
    _plot_group(
        ax2,
        x_inter,
        est_m.loc[interaction_effects],
        se_m.loc[interaction_effects],
        color_model,
        f"{_dash_safe(model_label)} (FA2 interaction)",
        dodge,
        interaction_ylim,
        marker=marker_inter,
    )

    ax_top.axhline(0, linestyle="--", linewidth=1.0, color="gray")
    ax_top.set_ylabel("Beta (Main Effects)", fontsize=font_size)
    ax_top.set_ylim(main_ylim)
    ax_top.spines["top"].set_visible(False)
    ax2.set_ylabel("Beta (FA2 Interactions)", color=axis2_color, fontsize=font_size)
    ax2.set_ylim(interaction_ylim)
    ax2.tick_params(axis="y", colors=axis2_color)
    ax2.spines["right"].set_color(axis2_color)
    ax2.spines["top"].set_visible(False)

    display_text = condition_label if condition_label is not None else horizon
    title_suffix = f" {_dash_safe(display_text)}" if display_text else ""
    ax_top.set_title(
        f"GLMM coefficients: Human vs. {_dash_safe(model_label)}{title_suffix}",
        fontsize=font_size,
    )
    # Combine handles from both axes (previously the interaction-effect
    # series were "_nolegend_" and only the main-effect/left-axis legend was
    # shown) so the marker-shape <-> axis-scale mapping above is spelled out
    # for the reader instead of left implicit.
    handles_top, labels_top = ax_top.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax_top.legend(
        handles_top + handles_2,
        labels_top + labels_2,
        fontsize=font_size * 0.55,
        loc="lower left",
        ncol=2,
        frameon=True,
    )

    # Model - human difference (pooled-SE z-test) -- kept for the comparison
    # table export even though it's no longer plotted as a separate panel.
    diff = est_m - est_h
    se_diff = np.sqrt(se_m**2 + se_h**2)
    z = diff / se_diff
    pvals = pd.Series(stats.norm.sf(np.abs(z)) * 2, index=diff.index)

    all_names = main_effects + [n.replace("FA2:", "OC:") for n in interaction_effects]

    def latexify(name: str) -> str:
        if "deltaev" in name:
            return name.replace("deltaev", r"$\Delta ES_t$")
        if "totevminus" in name:
            return name.replace("totevminus", r"$ES_{t-1}$")
        if "trial" in name:
            return name.replace("trial", r"$draw\_seq$")
        return name

    ax_top.set_xticks(x_all)
    ax_top.set_xticklabels(
        [latexify(n) for n in all_names],
        rotation=45,
        ha="right",
        fontsize=font_size * 0.75,
    )
    ax_top.grid(alpha=0.3, axis="y")

    fig.tight_layout()

    if save_fig:
        if fname is None:
            suffix = f"_{horizon}" if horizon else ""
            fname = f"glmm_betas_comparison{suffix}"
        _save_figure(fig, fname, path)

    comparison_df = pd.DataFrame(
        {
            "Human_Estimate": est_h,
            "Human_SE": se_h,
            f"{model_label}_Estimate": est_m,
            f"{model_label}_SE": se_m,
            "Difference": diff,
            "Diff_SE": se_diff,
            "z": z,
            "p": pvals,
        }
    )

    return fig, (ax_top, ax2), comparison_df


def plot_beta_with_significance(
    x,
    y,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | list[float] | None = None,
    is_latex: bool = False,
    font_size: int = 20,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str | None = None,
):
    """Scatter `x` vs. `y` with a linear regression line, annotated with
    Spearman significance stars above the line if the correlation is
    significant.

    Args:
        x: Values for the x-axis.
        y: Values for the y-axis.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        ylim: Y-axis limits; defaults to the data range padded by 0.2.
        is_latex: Passed to `_set_plot_style`.
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename slug override; defaults to a slug built from
            `xlabel`/`ylabel`.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    corr, pval = safe_spearman(x, y)
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    fig = plt.figure()
    plt.scatter(x, y, facecolors="white", edgecolors="k")
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    yy = m * xx + b
    plt.plot(xx, yy, color=(0.85, 0.325, 0.098))

    # determine significance
    stars = ""
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"

    if stars:  # add stars above regression line
        # place at middle x-value
        x_pos = x.min() + (x.max() - x.min()) / 2
        y_pos = m * x_pos + b + (0.05 * (y.max() - y.min()))  # slightly above line
        plt.text(
            x_pos,
            y_pos,
            stars,
            color="red",
            fontsize=font_size + 4,
            ha="center",
            va="bottom",
        )
    # for the ylim, make it between the max and min values +- 0.2
    if ylim is None:
        ylim = [min(y.min(), yy.min()) - 0.2, max(y.max(), yy.max()) + 0.2]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.tight_layout()

    if save_fig:
        slug = fname or re.sub(r"[^a-zA-Z0-9]+", "_", f"{xlabel}_vs_{ylabel}").strip("_")
        _save_figure(fig, f"beta_significance_{slug}", path)

    return fig


def _save_figure(fig, fname: str, path: str = DEFAULT_FIGURE_PATH, dpi: int = 600):
    """Save fig as .pdf, .svg, and .png, creating the directory if needed."""
    base, _ext = os.path.splitext(fname)
    if path is not None:
        os.makedirs(path, exist_ok=True)
        base = os.path.join(path, base)
    for ext, extra in ((".pdf", {}), (".svg", {}), (".png", {"dpi": dpi})):
        fig.savefig(f"{base}{ext}", bbox_inches="tight", pad_inches=0.03, **extra)


def plot_human_vs_simulated_data(
    outcome_human,
    draws_human,
    outcome_simulated,
    draws_simulated,
    ontop: bool = True,  # True: overlayed/ontop, False: separate
    bins_outcome: int = 25,
    bins_draws: int = 25,
    font_size: int = 20,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "",
    is_latex: bool = IS_LATEX,
    horizon: str | None = None,
):
    """Histogram human vs. simulated outcome values and draw counts, either
    overlaid on shared axes (`ontop=True`) or as a 2x2 side-by-side grid.

    Args:
        outcome_human: Human per-game outcome values.
        draws_human: Human per-game draw counts.
        outcome_simulated: Simulated per-game outcome values.
        draws_simulated: Simulated per-game draw counts.
        ontop: If True, overlay human/simulated histograms on 2 shared axes;
            if False, plot them side by side in a 2x2 grid.
        bins_outcome: Number of bins for the outcome histograms.
        bins_draws: Number of bins for the draws histogram (only used when
            `ontop=True`; the `ontop=False` branch uses a fixed bin range).
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Unused -- the saved filename is always derived from `horizon`.
        is_latex: Passed to `_set_plot_style`.
        horizon: Used to build the saved filename.

    Returns:
        None.
    """
    fname = f"human_vs_simulated_{horizon}.pdf"
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    if ontop:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        num_bins = np.arange(0, max(max(draws_human), max(draws_simulated)) + 2, 1)

        sns.histplot(
            outcome_human,
            bins=bins_outcome,
            color="skyblue",
            label="Human",
            ax=axes[0],
            kde=False,
            alpha=0.5,
        )
        sns.histplot(
            outcome_simulated,
            bins=bins_outcome,
            color="salmon",
            label="Simulated",
            ax=axes[0],
            kde=False,
            alpha=0.5,
        )
        axes[0].set_title("Outcome Values: Human vs. Simulated")
        axes[0].set_xlabel("Outcome")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        sns.histplot(
            draws_human,
            bins=num_bins,
            color="skyblue",
            label="Human",
            ax=axes[1],
            kde=False,
            alpha=0.5,
        )
        sns.histplot(
            draws_simulated,
            bins=num_bins,
            color="salmon",
            label="Simulated",
            ax=axes[1],
            kde=False,
            alpha=0.5,
        )
        axes[1].set_title("Number of Draws: Human vs. Simulated")
        axes[1].set_xlabel("Number of Draws")
        axes[1].set_ylabel("Frequency")
        axes[1].set_xticks(num_bins)
        axes[1].legend()
        plt.tight_layout()
        plt.suptitle("Comparison: Human vs. Simulated Data", fontsize=font_size, y=1.02)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey="row")
        num_bins_draws = np.arange(0, 15, 1)
        sns.histplot(outcome_human, bins=bins_outcome, color="skyblue", ax=axes[0, 0])
        axes[0, 0].set_title("Human Data: Outcome Values")
        axes[0, 0].set_xlabel("Outcome")
        axes[0, 0].set_ylabel("Frequency")

        sns.histplot(
            outcome_simulated, bins=bins_outcome, color="salmon", ax=axes[0, 1]
        )
        axes[0, 1].set_title("Simulated Data: Outcome Values")
        axes[0, 1].set_xlabel("Outcome")
        axes[0, 1].set_ylabel("Frequency")

        sns.histplot(draws_human, bins=num_bins_draws, color="skyblue", ax=axes[1, 0])
        axes[1, 0].set_title("Human Data: Number of Draws")
        axes[1, 0].set_xlabel("Number of Draws")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_xticks(np.arange(0, max(draws_human) + 1, 1))

        sns.histplot(
            draws_simulated, bins=num_bins_draws, color="salmon", ax=axes[1, 1]
        )
        axes[1, 1].set_title("Simulated Data: Number of Draws")
        axes[1, 1].set_xlabel("Number of Draws")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_xticks(np.arange(0, max(draws_simulated) + 1, 1))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(
            "Histogram: Human vs. Simulated (Side by Side)", fontsize=font_size + 2
        )

    if save_fig:
        _save_figure(fig, fname, path)


def plot_true_vs_recovered_params(
    results_df: pd.DataFrame,
    param_order: list[str],
    results_df_recovered: pd.DataFrame | None = None,
    font_size: int = 20,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "",
    is_latex: bool = False,  # Assuming default if IS_LATEX isn't defined
    horizon: str = "",
):
    """Histogram the distribution of true (and optionally recovered)
    parameter values, one column per parameter in `param_order`.

    Args:
        results_df: DataFrame with a "fit_params_ga" column of per-subject
            parameter vectors (ordered per `param_order`), used as the
            "true" parameters.
        param_order: Ordered parameter names, used as DataFrame columns.
        results_df_recovered: If given, a second DataFrame in the same
            format plotted as a second ("recovered") row.
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename override; defaults to a name derived from `horizon`.
        is_latex: Passed to `_set_plot_style`.
        horizon: Used to build the default filename.

    Returns:
        None.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    # Determine if we are plotting recovered parameters
    has_recovered = results_df_recovered is not None
    nrows = 2 if has_recovered else 1

    # Adjust filename if not explicitly provided
    if not fname:
        prefix = "true_and_recovered" if has_recovered else "true_only"
        fname = f"{prefix}_params_hist_{horizon}.pdf"

    # Process true parameters
    params_df_true = pd.DataFrame(
        results_df["fit_params_ga"].tolist(), columns=param_order
    )

    # Process recovered parameters if they exist
    if has_recovered:
        params_df_recovered = pd.DataFrame(
            results_df_recovered["fit_params_ga"].tolist(), columns=param_order
        )

    # squeeze=False ensures 'axes' is ALWAYS a 2D array, even with 1 row
    fig, axes = plt.subplots(
        nrows,
        len(param_order),
        figsize=(5 * len(param_order), 4 * nrows),  # Scale height by number of rows
        sharey="row",
        squeeze=False,
    )

    param_labels = _get_param_labels(param_order, is_latex=is_latex)
    for i, param in enumerate(param_order):
        plabel = param_labels[i]
        # 1. Plot True Parameters (Always happens)
        sns.histplot(params_df_true[param], ax=axes[0, i], bins=20)
        axes[0, i].set_title(f"True: {plabel}")

        # 2. Plot Recovered Parameters (Only if provided)
        if has_recovered:
            sns.histplot(params_df_recovered[param], ax=axes[1, i], bins=20)
            axes[1, i].set_title(f"Recovered: {plabel}")

    # Set y-axis labels on the first column only
    axes[0, 0].set_ylabel("True parameters")
    if has_recovered:
        axes[1, 0].set_ylabel("Recovered parameters")

    plt.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)


def _correlation_significance_stars(p: float) -> str:
    """"*"/"**"/"***" at p<0.05/0.01/0.001, "" otherwise (incl. NaN) --
    shared convention across every correlation annotation in this module
    (see `plot_fitted_param_correlations`, `plot_fitted_param_correlations_scatter`)."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _plot_value_pair_scatter_grid(
    df: pd.DataFrame,
    param_order: list[str],
    xlabel: str,
    ylabel: str,
    suptitle: str,
    font_size: int = 16,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "",
    is_latex: bool = IS_LATEX,
):
    """Shared implementation behind `plot_commit_vs_full_param_scatter` and
    `plot_param_pair_scatter`: one scatter panel per parameter, plotting
    `df["value_x"]` against `df["value_y"]` (one point per subject) with a
    y=x reference line, and a per-panel title giving the parameter symbol
    plus its Pearson r and Spearman rho (with significance stars) -- kept
    as a title (above the axes) rather than an in-plot text box so it
    never sits on top of the scatter points, and kept out of the x/y axis
    labels so those can stay as short, reusable identity labels (e.g.
    "Short fit" / "Long fit") instead of repeating the parameter name on
    every panel.

    Each panel keeps a fixed physical width (unlike the `n x n` PairGrid in
    `plot_fitted_param_correlations_scatter`, whose whole grid is squeezed
    into one fixed column width, so that function inflates its font with
    `n` to compensate); here the figure just gets wider as `n` grows, so
    title/tick/label sizes are fixed, tuned to stay comfortably within
    each panel's width without a per-panel legend/text box overlapping the
    scatter points, and to stay >=8pt at typical print size (PLOS
    Computational Biology's figure-text minimum).

    Args:
        df: Must have columns "param", "value_x", "value_y", and
            "subject_ID" (used only to report N in `suptitle`'s caller,
            not recomputed here).
        param_order: Parameter names/plot order (must be a subset of the
            "param" values in `df`).
        xlabel, ylabel: Shared axis labels used on every panel.
        suptitle: Figure-level title.
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename (without extension).
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.

    Returns:
        (fig, axes) matplotlib objects.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    param_labels = _get_param_labels(param_order, is_latex=is_latex)
    rho_symbol = r"$\rho$" if is_latex else "ρ"

    n = len(param_order)
    title_fs = font_size * 0.72
    tick_fs = font_size * 0.55
    label_fs = font_size * 0.65

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.0), squeeze=False)
    axes = axes[0]
    for ax, param, plabel in zip(axes, param_order, param_labels):
        g = df[df["param"] == param]
        x, y = g["value_x"], g["value_y"]
        ax.scatter(
            x, y, alpha=0.6, s=40, color="steelblue", edgecolors="black", linewidths=0.4
        )
        if len(g) > 0:
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, zorder=0)

        if x.nunique() > 1 and y.nunique() > 1:
            r, p = pearsonr(x, y)
        else:
            r, p = np.nan, np.nan
        rho, rho_p = safe_spearman(x, y)
        stats_line = (
            f"r={r:.2f}{_correlation_significance_stars(p)}, "
            f"{rho_symbol}={rho:.2f}{_correlation_significance_stars(rho_p)}"
        )
        ax.set_title(f"{plabel}\n{stats_line}", fontsize=title_fs)
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(suptitle, fontsize=font_size * 0.95)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, axes


def plot_commit_vs_full_param_scatter(
    long_df: pd.DataFrame,
    task: str,
    param_order: list[str],
    font_size: int = 16,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "",
    is_latex: bool = IS_LATEX,
):
    """Per-parameter scatter of full-fit vs. commit-fit value (one point
    per subject) for one model architecture -- see
    `_plot_value_pair_scatter_grid` for the per-panel title (parameter +
    Pearson r + Spearman rho) and y=x reference line.

    Args:
        long_df: Output of `build_commit_vs_full_param_df` (data_handling),
            optionally concatenated across several tasks; rows where
            `long_df["task"] != task` are ignored.
        task: TASK code to plot.
        param_order: Parameter names/plot order (must match the "param"
            values in `long_df`).
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename override; defaults to a name derived from `task`.
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.

    Returns:
        (fig, axes) matplotlib objects.
    """
    df = long_df[long_df["task"] == task].rename(
        columns={"full_value": "value_x", "commit_value": "value_y"}
    )
    n_subjects = df["subject_ID"].nunique()
    return _plot_value_pair_scatter_grid(
        df,
        param_order,
        xlabel="Full fit",
        ylabel="Commit fit",
        suptitle=f"{task} (N={n_subjects} subjects)",
        font_size=font_size,
        path=path,
        save_fig=save_fig,
        fname=fname or f"commit_vs_full_scatter_{task}.pdf",
        is_latex=is_latex,
    )


def plot_commit_vs_full_summary_heatmap(
    summary_df: pd.DataFrame,
    value_col: str = "median_pct_diff_sym",
    sig_col: str = "wilcoxon_p_fdr",
    alpha: float = 0.05,
    value_fmt: str = "{:.0f}",
    cbar_label: str = "Median symmetric % diff (commit - full)",
    font_size: int = 14,
    figsize: tuple[float, float] | None = None,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "commit_vs_full_summary_heatmap",
    is_latex: bool = IS_LATEX,
):
    """Model-architecture x parameter heatmap of a commit-vs-full summary
    statistic, diverging around 0 (same coolwarm/center=0 convention as
    `plot_param_correlations`).

    Args:
        summary_df: Output of `summarize_commit_vs_full` (data_handling),
            with (at least) columns "task", "param", `value_col`,
            `sig_col`.
        value_col: Column to plot as the heatmap color/annotation; default
            is the symmetric percent difference (bounded, sign-safe -- see
            `build_commit_vs_full_param_df`). Pass "spearman_rho" to show
            per-subject rank agreement between the two fits instead.
        sig_col: FDR-corrected paired-test p-value column; cells with
            `sig_col < alpha` are marked with "*".
        alpha: Significance threshold applied to `sig_col`.
        value_fmt: `str.format` spec used for cell annotations (e.g.
            "{:.0f}" for a percent, "{:.2f}" for a correlation).
        cbar_label: Colorbar label.
        font_size: Base font size for the plot.
        figsize: Figure size; auto-scaled from the pivot table shape if
            None.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename (without extension).
        is_latex: Passed to `_set_plot_style`.

    Returns:
        (fig, ax) matplotlib objects. A (task, param) pair absent from
        `summary_df` (that architecture doesn't have that parameter) is
        left blank in the heatmap.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    pivot = summary_df.pivot(index="task", columns="param", values=value_col)
    sig = (summary_df.pivot(index="task", columns="param", values=sig_col) < alpha)
    sig = sig.reindex_like(pivot).fillna(False)

    annot = pivot.map(lambda v: "" if pd.isna(v) else value_fmt.format(v))
    annot = annot.where(~sig, annot + "*")

    if figsize is None:
        figsize = (1.1 * pivot.shape[1] + 3, 0.45 * pivot.shape[0] + 2)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        annot=annot,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Model architecture (TASK)")
    ax.set_title(f"{cbar_label}\n(* = FDR-significant paired Wilcoxon, p<{alpha})")
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, ax


def plot_param_pair_scatter(
    long_df: pd.DataFrame,
    param_order: list[str],
    axis_label_a: str | None = None,
    axis_label_b: str | None = None,
    font_size: int = 16,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "",
    is_latex: bool = IS_LATEX,
):
    """Per-parameter scatter of `value_a` vs. `value_b` (one point per
    subject) for a single label_a/label_b comparison -- see
    `_plot_value_pair_scatter_grid` for the per-panel title (parameter +
    Pearson r + Spearman rho) and y=x reference line.

    Args:
        long_df: Output of `build_param_pair_comparison_df`
            (data_handling), for one label_a/label_b pair (the "label_a"/
            "label_b" columns are read from the first row to build the
            figure title, and the axis labels too if `axis_label_a`/
            `axis_label_b` aren't given).
        param_order: Parameter names/plot order (must be a subset of the
            "param" values in `long_df`).
        axis_label_a, axis_label_b: Short axis labels (e.g. "Short fit" /
            "Long fit") -- the figure title already names the full
            label_a/label_b (e.g. "full: short (SBEXT-RPHC---) vs. full:
            long (LBE-T-RPhCL--)"), so repeating that on every panel's
            axis is redundant; defaults to the full label_a/label_b if not
            given.
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename override; defaults to a name derived from the two
            labels.
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.

    Returns:
        (fig, axes) matplotlib objects.
    """
    label_a = long_df["label_a"].iloc[0]
    label_b = long_df["label_b"].iloc[0]
    df = long_df.rename(columns={"value_a": "value_x", "value_b": "value_y"})
    n_subjects = df["subject_ID"].nunique()
    return _plot_value_pair_scatter_grid(
        df,
        param_order,
        xlabel=axis_label_a or label_a,
        ylabel=axis_label_b or label_b,
        suptitle=f"{label_a} vs. {label_b} (N={n_subjects} subjects)",
        font_size=font_size,
        path=path,
        save_fig=save_fig,
        fname=fname or f"param_pair_scatter_{label_a}_vs_{label_b}.pdf",
        is_latex=is_latex,
    )


def plot_param_pair_correlation_heatmap(
    corr_df: pd.DataFrame,
    param_order: list[str],
    title: str = "",
    font_size: int = 16,
    figsize: tuple[float, float] | None = None,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "param_pair_correlation_heatmap",
    is_latex: bool = IS_LATEX,
):
    """Two-column heatmap (Pearson r, Spearman rho) of per-parameter
    agreement for one label_a/label_b comparison, annotated with
    significance stars -- same convention as `plot_fitted_param_correlations`
    (* p<0.05, ** p<0.01, *** p<0.001).

    Args:
        corr_df: Output of `summarize_param_pair_correlations`
            (data_handling), restricted to a single (label_a, label_b)
            pair (e.g. via `corr_df[corr_df["param"].isin(param_order)]`
            after filtering to one comparison upstream).
        param_order: Row order; must be a subset of `corr_df["param"]`.
        title: Heatmap title.
        font_size: Base font size for the plot.
        figsize: Figure size; auto-scaled from `param_order` length if
            None.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename (without extension).
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.

    Returns:
        (fig, ax) matplotlib objects.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    df = corr_df.set_index("param").loc[param_order]
    tick_labels = _get_param_labels(param_order, is_latex=is_latex)

    def _stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    values = df[["pearson_r", "spearman_rho"]].to_numpy(dtype=float)
    annot = np.empty(values.shape, dtype=object)
    for i, (_, row) in enumerate(df.iterrows()):
        annot[i, 0] = f"{row['pearson_r']:.2f}{_stars(row['pearson_p'])}"
        annot[i, 1] = f"{row['spearman_rho']:.2f}{_stars(row['spearman_p'])}"

    if figsize is None:
        figsize = (5, 0.6 * len(param_order) + 2)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        values,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        yticklabels=tick_labels,
        xticklabels=["Pearson r", "Spearman rho"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title or "Short-fit vs. long-fit parameter agreement")
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, ax


#


def plot_recovery_and_ll(
    true_params_all: np.ndarray,
    fit_params_ga: np.ndarray,
    after_lls_ga,
    results_df: pd.DataFrame,
    results_df_recovered: pd.DataFrame,
    param_order: list[str],
    param_ranges: dict,
    ocd_userID: list | np.ndarray | None = None,
    font_size: int = 20,
    plot_regression: bool = True,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
    horizon: str = "",
):
    """Plot true-vs-recovered parameter scatter grid (one panel per
    parameter, with optional OLS regression line/CI) plus a separate
    per-subject log-likelihood figure.

    Args:
        true_params_all: (n_subjects, n_params) array of ground-truth
            parameter values, ordered per `param_order`.
        fit_params_ga: (n_subjects, n_params) array of recovered parameter
            values, same ordering.
        after_lls_ga: Currently unused; the log-likelihood figure instead
            reads `results_df["after_lls_ga"]` directly.
        results_df: DataFrame with an "after_lls_ga" column used for the
            log-likelihood figure.
        results_df_recovered: DataFrame with a "subject_ID" column (used to
            build a per-subject OCD mask, currently unused in the plot).
        param_order: Ordered parameter names.
        param_ranges: {param_name: (min, max)} axis limits per parameter.
        ocd_userID: Subject IDs flagged as OCD; currently only used to
            compute an unused mask and has no visible effect on the plot.
        font_size: Base font size for the plot.
        plot_regression: Whether to overlay an OLS fit line + 95% CI band
            per parameter panel.
        path: Directory to save the figures into (if `save_fig`).
        save_fig: Whether to save the figures via `_save_figure`.
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.
        horizon: Used to build the saved filenames.

    Returns:
        None.
    """
    method_colors = {"Recovered": "tab:blue"}
    methods = [("Recovered", fit_params_ga, method_colors["Recovered"])]
    n_methods = len(methods)
    n_params = len(param_order)
    param_labels = _get_param_labels(param_order, is_latex=is_latex)

    _set_plot_style(font_size=font_size, is_latex=is_latex)
    subject_ids = results_df_recovered["subject_ID"].to_numpy()
    ocd_mask = np.isin(subject_ids, ocd_userID) if ocd_userID is not None else None

    # Lay the n_params panels out in a roughly square grid (one block of
    # rows per method) instead of a single n_params-wide row: a 1x8 row is
    # 32in wide x 4in tall, so fitting it to a page/column shrinks width far
    # more than height, leaving markers/text illegible even though each
    # panel looked fine at native size. A square-ish grid shrinks evenly.
    ncols = max(1, ceil(n_params**0.5))
    nrows_per_method = ceil(n_params / ncols)
    total_rows = nrows_per_method * n_methods

    # Each panel keeps a fixed 4x4in size regardless of ncols/nrows (the
    # figure below is sized `(4*ncols, 4*total_rows)`, i.e. it grows with
    # the grid instead of the grid shrinking into a fixed canvas), so text
    # sizes are fixed rather than scaled up with the grid -- an earlier
    # version scaled `grid_font_size` by `(4*ncols)/6.5` on the assumption
    # this grid gets squeezed into a fixed-width column like
    # `plot_fitted_param_correlations_scatter`'s PairGrid, which made
    # titles/labels/legend text grow faster than the (also-growing) canvas
    # and overflow into neighboring panels and the scatter points for any
    # model with more than a handful of parameters.
    grid_font_size = font_size

    fig, axs_flat = plt.subplots(
        total_rows,
        ncols,
        figsize=(4 * ncols, 4 * total_rows),
        sharex=False,
        sharey=False,
    )
    axs_flat = np.array(axs_flat).reshape(total_rows, ncols)

    for method_idx, (method_name, fit_params, color) in enumerate(methods):
        row_offset = method_idx * nrows_per_method
        # hide any unused trailing cells in this method's block
        for idx in range(n_params, nrows_per_method * ncols):
            r, c = divmod(idx, ncols)
            axs_flat[row_offset + r, c].set_visible(False)

        for col, pname in enumerate(param_order):
            grid_row, grid_col = divmod(col, ncols)
            ax = axs_flat[row_offset + grid_row, grid_col]
            plabel = param_labels[col]

            ax.plot(
                param_ranges[pname],
                param_ranges[pname],
                "k--",
                label="Perfect",
                zorder=0,
            )
            ax.scatter(
                true_params_all[:, col],
                fit_params[:, col],
                alpha=0.7,
                color=color,
                s=55,
                label=method_name,
            )

            x = true_params_all[:, col]
            y = fit_params[:, col]

            title = plabel
            if not np.allclose(x, x[0]):
                r, p = pearsonr(x, y)
                stars = (
                    "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
                # Appended to the title (above the axes) rather than an
                # in-axes text box: a semi-transparent box anchored inside
                # the panel routinely sat on top of the scatter points
                # closest to that corner (worst for parameters whose
                # recovered values cluster near an axis limit, e.g. a
                # bounded parameter near 0 or 1).
                title = f"{plabel}\n$r={r:.2f}{stars}$"

            if plot_regression:
                if not np.allclose(x, x[0]):
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()
                    intercept, slope = (
                        model.params
                        if len(model.params) == 2
                        else (model.params[0], 0.0)
                    )
                    print(f"{pname}: slope={slope:.2f}, intercept={intercept:.2f}")

                    x_range = np.linspace(
                        param_ranges[pname][0], param_ranges[pname][1], 200
                    )
                    X_range = sm.add_constant(x_range)
                    y_pred = model.predict(X_range)
                    ci = model.get_prediction(X_range).conf_int(alpha=0.05)
                    lower, upper = ci[:, 0], ci[:, 1]

                    ax.plot(
                        x_range,
                        y_pred,
                        color=color,
                        lw=2,
                        label="Fit",
                    )
                    ax.fill_between(x_range, lower, upper, color=color, alpha=0.2)
                else:
                    intercept, slope = np.mean(y), 0.0
                    print(f"{pname}: slope={slope:.2f}, intercept={intercept:.2f}")

            ax.set_title(title, fontsize=grid_font_size * 0.85)
            if grid_col == 0:
                ax.set_ylabel(method_name, fontsize=grid_font_size * 0.85)
            ax.set_xlabel(f"True {plabel}", fontsize=grid_font_size * 0.85)

            if pname in param_ranges:
                ax.set_xlim(param_ranges[pname])
                ax.set_ylim(param_ranges[pname])

            ax.tick_params(axis="both", which="major", labelsize=grid_font_size * 0.55)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    # One shared legend below the whole grid instead of inside a single
    # panel's corner: an in-panel legend (previously the last panel's
    # lower-right corner) routinely overlapped that panel's own scatter
    # points/regression band, since corner placement isn't guaranteed clear
    # of data. "Perfect"/"Recovered"/"Fit" are the same across every panel,
    # so pulling handles from just the first axes is sufficient.
    handles, labels = axs_flat[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            fontsize=grid_font_size * 0.5,
            bbox_to_anchor=(0.5, 0.0),
            frameon=True,
        )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    if save_fig:
        _save_figure(fig, f"parameter_recovery_{horizon}", path)


    fig2 = plt.figure(figsize=(10, 6))
    before_lls = results_df["after_lls_ga"]

    plt.plot(before_lls, marker="o", color="gray")
    # plt.plot(after_lls_ga, label='After fit', marker='o', color=method_colors['Recovered'])

    plt.xlabel("Subject", fontsize=font_size)
    plt.ylabel("  Log-likelihood", fontsize=font_size)
    plt.title("Log-likelihood for each subject and method", fontsize=font_size)
    plt.legend()
    plt.tight_layout()

    if save_fig:
        _save_figure(fig2, f"ll_of_true_vs_recovered_{horizon}", path)


def plot_fitted_param_correlations(
    fit_params_dict: dict,
    param_order: list[str],
    font_size: int = 20,
    figsize: tuple[float, float] = (14, 12),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
    horizon: str = "",
):
    """Plot a Pearson correlation-matrix heatmap between fitted parameters,
    one figure per method in `fit_params_dict`.

    Args:
        fit_params_dict: {method_name: (n_subjects, n_params) array} of
            fitted parameter values, ordered per `param_order`.
        param_order: Ordered parameter names, used as heatmap tick labels.
        font_size: Base font size for the plot.
        figsize: Figure size.
        path: Directory to save the figures into (if `save_fig`).
        save_fig: Whether to save the figures via `_save_figure`.
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.
        horizon: Used to build the saved filenames.

    Returns:
        None.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    tick_labels = _get_param_labels(param_order, is_latex=is_latex)

    for method_name, fit_params in fit_params_dict.items():

        df = pd.DataFrame(fit_params, columns=param_order)

        n = len(param_order)

        corr_matrix = np.zeros((n, n))
        annot_matrix = np.empty((n, n), dtype=object)

        for i in range(n):
            for j in range(n):

                r, p = pearsonr(df.iloc[:, i], df.iloc[:, j])

                corr_matrix[i, j] = r

                # --- significance stars ---
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                else:
                    stars = ""

                annot_matrix[i, j] = f"{r:.2f}{stars}"

        fig = plt.figure(figsize=figsize)

        sns.heatmap(
            corr_matrix,
            annot=annot_matrix,
            fmt="",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            annot_kws={"fontsize": font_size * 0.55},
        )

        plt.title(f"Correlation matrix")
        plt.tight_layout()

        if save_fig:
            _save_figure(
                fig,
                f"correlation_matrix_{method_name}_{horizon}.pdf",
                path,
            )


def plot_fitted_param_correlations_scatter(
    fit_params_dict: dict,
    param_order: list[str],
    font_size: int = 16,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = False,
    horizon: str = "",
):
    """Plot a pairwise scatter grid (lower triangle regplots + diagonal
    histograms, via seaborn PairGrid) between fitted parameters, one figure
    per method in `fit_params_dict`, annotated with per-pair Pearson r and
    significance stars.

    Args:
        fit_params_dict: {method_name: (n_subjects, n_params) array} of
            fitted parameter values, ordered per `param_order`.
        param_order: Ordered parameter names, used as grid labels.
        font_size: Base font size for the plot.
        path: Directory to save the figures into (if `save_fig`).
        save_fig: Whether to save the figures via `_save_figure`.
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.
        horizon: Used to build the saved filenames.

    Returns:
        None.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    # --- helper for significance stars ---
    def significance_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    # Each PairGrid cell keeps a fixed 2.8in size regardless of `n` (the
    # grid below is explicitly sized to `(2.8*n, 2.8*n)`, i.e. it grows
    # with `n` instead of a fixed-size grid shrinking as `n` grows), so
    # `grid_scale` stays 1.0 -- text was previously scaled up by
    # `(2.8*n)/6.5` on the assumption this n x n grid gets squeezed into a
    # fixed 6.5in-wide column, which instead made the r-value text box (and
    # tick/axis labels) grow faster than the also-growing canvas and
    # overlap the scatter points for any model with more than a handful of
    # parameters. `scatter_with_reg` closes over this name and reads it at
    # call time.
    grid_scale = 1.0

    # --- custom plotting function ---
    def scatter_with_reg(x, y, **kwargs):

        ax = plt.gca()

        sns.regplot(
            x=x,
            y=y,
            scatter_kws={"s": 15, "alpha": 0.7},
            line_kws={"linewidth": 1.5},
            ci=None,
            ax=ax,
        )

        # --- Pearson correlation + significance ---
        r, p = pearsonr(x, y)

        stars = significance_stars(p)

        ax.text(
            0.95,
            0.95,
            f"r = {r:.2f}{stars}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=font_size * grid_scale * 0.55,
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    tick_labels = _get_param_labels(param_order, is_latex=is_latex)

    # --- main loop ---
    for method_name, fit_params in fit_params_dict.items():

        df = pd.DataFrame(fit_params, columns=param_order)
        df.columns = tick_labels

        n = len(tick_labels)

        g = sns.PairGrid(df, height=2.8)

        # lower triangle
        g.map_lower(scatter_with_reg)

        # diagonal
        g.map_diag(sns.histplot, kde=False)

        # hide upper triangle
        for i in range(n):
            for j in range(n):
                if i < j:
                    g.axes[i, j].set_visible(False)

        # --- clean labels ---
        for ax in g.axes.flat:
            if ax is not None and ax.get_visible():

                ax.label_outer()

                ax.tick_params(
                    axis="x",
                    rotation=45,
                    labelsize=font_size * grid_scale * 0.6,
                )

                ax.tick_params(
                    axis="y",
                    labelsize=font_size * grid_scale * 0.6,
                )
                ax.xaxis.label.set_size(font_size * grid_scale * 0.7)
                ax.yaxis.label.set_size(font_size * grid_scale * 0.7)

        # --- sizing ---
        g.fig.set_size_inches(2.8 * n, 2.8 * n)

        g.fig.subplots_adjust(
            top=0.95,
            bottom=0.08,
            left=0.08,
            right=0.95,
            hspace=0.2,
            wspace=0.2,
        )

        g.fig.suptitle(
            f"Parameter correlations",
            fontsize=font_size * grid_scale,
            y=1.02,
        )

        if save_fig:
            _save_figure(
                g.fig,
                f"pairwise_correlations_{method_name}_{horizon}.pdf",
                path,
            )


def plot_param_correlations(
    true_params_all: np.ndarray,
    fit_params: np.ndarray,
    param_order: list[str],
    method_name: str = "Recovered",
    figsize: tuple[float, float] = (10, 10),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
    horizon: str = "",
    export_latex_table: bool = True,
):
    """Plot a single-column heatmap of per-parameter true-vs-recovered
    correlation coefficients, optionally also exporting a matching LaTeX
    table.

    Args:
        true_params_all: (n_subjects, n_params) array of ground-truth
            parameter values, ordered per `param_order`.
        fit_params: (n_subjects, n_params) array of recovered parameter
            values, same ordering.
        param_order: Ordered parameter names.
        method_name: Column label for the correlation heatmap/table.
        figsize: Figure size.
        path: Directory to save the figure/table into.
        save_fig: Whether to save the figure via `_save_figure`.
        is_latex: Passed to `_set_plot_style` and `_get_param_labels`.
        horizon: Used to build the saved filenames.
        export_latex_table: If True (and `path` is truthy), also write a
            LaTeX correlation table to `path`.

    Returns:
        None.
    """
    _set_plot_style(is_latex=is_latex)
    tick_labels = _get_param_labels(param_order, is_latex=is_latex)
    correlations = {}
    corr_vals = []
    for i in range(len(param_order)):
        corr = np.corrcoef(true_params_all[:, i], fit_params[:, i])[0, 1]
        corr_vals.append(corr)
    correlations[method_name] = corr_vals

    correlation_df = pd.DataFrame(correlations, index=tick_labels)

    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5
    )
    plt.title(f"Correlation coefficients between true parameters and fitted parameters")
    plt.tight_layout()

    if save_fig:
        _save_figure(fig, f"param_correlations_{horizon}", path)

    if export_latex_table and path:
        latex_labels = _get_param_labels(param_order, is_latex=True)
        table_df = pd.DataFrame(
            {
                "Parameter": latex_labels,
                "Correlation (true vs.\\ recovered)": [f"{c:.2f}" for c in corr_vals],
            }
        )
        latex_table = table_df.to_latex(
            index=False,
            escape=False,
            column_format="lc",
            caption="Pearson correlation between true and recovered parameters.",
            label=f"tab:param_recovery_corr_{horizon}",
        )
        table_path = os.path.join(path, f"param_recovery_correlations_{horizon}.tex")
        with open(table_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX correlation table exported to {table_path}")


def plot_sim_vs_fit(
    out_sim_fit,
    draws_sim_fit,
    out_sim,
    draws_sim,
    font_size: int = 20,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "",
    is_latex: bool = IS_LATEX,
    horizon: str = "",
):
    """Histogram outcome and draw-count values for a simulate-then-refit
    recovery check: original simulated data vs. data simulated from the
    parameters recovered by fitting it.

    Args:
        out_sim_fit: Outcome values simulated from the recovered (refit)
            parameters.
        draws_sim_fit: Draw counts simulated from the recovered parameters.
        out_sim: Outcome values from the original simulation.
        draws_sim: Draw counts from the original simulation.
        font_size: Base font size for the plot.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Unused -- the saved filename is always derived from `horizon`.
        is_latex: Passed to `_set_plot_style`.
        horizon: Used to build the saved filename.

    Returns:
        None.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    fname = f"recovered_data_{horizon}.pdf"

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    bins_cards = np.arange(0, 14, 1)

    sns.histplot(
        out_sim, bins=30, color="salmon", alpha=0.6, label="Simulated", ax=axes[0]
    )
    sns.histplot(
        out_sim_fit, bins=30, color="#5db8b2", alpha=0.6, label="Recovery", ax=axes[0]
    )
    axes[0].set_title("Outcome")
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    sns.histplot(
        draws_sim,
        bins=bins_cards,
        color="salmon",
        alpha=0.6,
        label="Simulated",
        ax=axes[1],
    )
    sns.histplot(
        draws_sim_fit,
        bins=bins_cards,
        color="#5db8b2",
        alpha=0.6,
        label="Recovery",
        ax=axes[1],
    )
    axes[1].set_title("Number of Draws")
    axes[1].set_xlabel("Number of Draws")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Histogram: Recovery vs. Simulated Data", fontsize=font_size)

    if save_fig:
        _save_figure(fig, fname, path)


font = {"size": 18}
# plt.style.use('dark_background')
# using rc function
plt.rc("font", **font)
# How to instantiate POMDP class For MEG Task and plot everything, short horizon
# Arguments:


def plot_num_draws(
    num_draws_both,
    num_draws_long,
    num_draws_short,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "num_draws",
):
    """Plot three side-by-side histograms of the number of draws per game:
    short-horizon, long-horizon, and both combined.

    Args:
        num_draws_both: Draw counts for short+long games combined.
        num_draws_long: Draw counts for long-horizon games.
        num_draws_short: Draw counts for short-horizon games.
        path: Directory to save the figure into (if `save_fig`).
        save_fig: Whether to save the figure via `_save_figure`.
        fname: Filename (without extension) used when saving.

    Returns:
        tuple: (fig, axes) -- the figure and its 3 axes.
    """
    _set_plot_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), sharey=True
    )  # horizontal subplots, shared y-axis

    sns.histplot(num_draws_short, bins=8, kde=True, color="skyblue", ax=axes[0])
    sns.histplot(num_draws_long, bins=14, kde=True, color="skyblue", ax=axes[1])
    sns.histplot(num_draws_both, bins=14, kde=True, color="skyblue", ax=axes[2])

    for ax in axes:
        # Determine x-axis limits to set ticks accordingly
        xmin, xmax = ax.get_xlim()
        # Set x-ticks every 2 units, adjusting range to cover the data limits
        ax.set_xticks(np.arange(int(np.floor(xmin)), int(np.ceil(xmax)) + 1, 2))
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Number of Draws")

    axes[0].set_title("Short Sequence")
    axes[1].set_title("Long Sequence")
    axes[2].set_title("Both Sequences")

    plt.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, axes


# plot the dataset


### POMDP plotting


def plot_best_actions_symm(
    best_actions: np.ndarray,
    max_cards_per_draw: int,
    noise_std: float = 0,
    noise_trend: str = "noise_free",
    label: str | None = None,
    path: str = DEFAULT_FIGURE_PATH,
    title: str | None = None,
    title_fontsize: int = 20,
):
    """
    Plots the best actions heatmap.

    Args:
        best_actions (np.ndarray): Array of best actions, shape
            (num_draws, num_yellow, num_blue).
        max_cards_per_draw (int): Max cards drawable per turn (used to map
            each draw count to its reachable yellow/blue index range).
        noise_std (float, optional): Only used to build the default filename
            when `label` is not given.
        noise_trend (str, optional): Unused; kept for call-site compatibility.
        label (str, optional): Label for the figure filename.
        path (str, optional): Path to save the figures.
        title (str, optional): Panel title, which should name the manipulation
            the panel illustrates. Without it the panels are indistinguishable
            once they are laid out side by side in a multi-panel figure.
        title_fontsize (int, optional): Point size of that title. Kept large
            because these panels are reduced when tiled.

    Returns:
        np.ma.MaskedArray: The masked, draw x (yellow-blue) array plotted in
        the heatmap.
    """
    num_draws, num_yellow, num_blue = best_actions.shape
    max_diff = num_yellow - 1
    min_diff = -max_diff

    diff_range = np.arange(min_diff, max_diff + 1)
    adjusted_best_actions = np.full((num_draws, len(diff_range)), np.nan)

    for draw in range(num_draws):
        for yellow in range(num_yellow):
            blue = draw * max_cards_per_draw - yellow
            if blue < 0 or blue >= num_blue:
                continue  # skip invalid blue indices
            diff = yellow - blue
            if min_diff <= diff <= max_diff:
                diff_index = diff - min_diff
                adjusted_best_actions[draw, diff_index] = best_actions[
                    draw, yellow, blue
                ]

    # Mask out invalid entries (where NaN values exist or value == 5)
    mask_nan = np.isnan(adjusted_best_actions)
    mask_eq5 = adjusted_best_actions == 5
    mask = np.logical_or(mask_nan, mask_eq5)
    adjusted_best_actions = np.ma.array(adjusted_best_actions, mask=mask)

    # Define the colormap for the heatmap
    colors = ["orange", "blue", "green"]
    cmap = plt.cm.colors.ListedColormap(colors)

    _set_plot_style()
    fig = plt.figure(figsize=(14, 8))
    heatmap = plt.imshow(
        adjusted_best_actions.T,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=2,
        origin="lower",
    )

    cbar = plt.colorbar(heatmap, ticks=[0.33, 1, 1.67])
    cbar.set_ticklabels(["Yellow (0)", "Blue (1)", "Wait (2)"])

    if title:
        plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Number of Draws")
    plt.ylabel("Yellow - Blue Difference")

    plt.xticks(np.arange(0, num_draws, 1))

    # Show only even numbers on the y-axis
    odd_indices = [i for i, val in enumerate(diff_range) if val % 2 != 0]
    even_diffs = diff_range[odd_indices]
    indices_step_5 = [i for i, val in enumerate(diff_range) if val % 5 == 0]

    plt.yticks(indices_step_5)
    plt.grid(which="major", color="w", linestyle="-", linewidth=0.5)
    plt.tight_layout()

    fname = (
        f"best_actions_heatmap_{label}"
        if label is not None
        else f"best_actions_heatmap_with_noise_{noise_std}"
    )
    _save_figure(fig, fname, path=path)

    return adjusted_best_actions


def plot_all_subjects_ensemble(
    ensemble_data,
    draw_bin_start: int = 1,
    valid_indices: list[int] | None = None,
    outcome_labels: list | None = None,
    horizon: str = "",
    font_size: int = 20,
    is_latex: bool = False,
    path: str = DEFAULT_FIGURE_PATH,
    title_label: str | None = None,
):
    """
    Plots ensemble average vs concatenated human data across ALL subjects.
    Human counts are summed across subjects; simulated ensemble mean is summed;
    std is propagated as sqrt(sum of variances) across independent subjects.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)
    valid_indices = valid_indices if valid_indices is not None else [0, 1, 4]
    outcome_labels = outcome_labels if outcome_labels is not None else [-2, -1, 2]

    all_human_counts = np.sum(
        [d["human_counts"] for d in ensemble_data.values()], axis=0
    )
    all_avg_sim_counts = np.sum(
        [d["avg_sim_counts"] for d in ensemble_data.values()], axis=0
    )
    all_std_sim_counts = np.sqrt(
        np.sum([d["std_sim_counts"] ** 2 for d in ensemble_data.values()], axis=0)
    )
    all_human_outcomes = np.sum(
        [d["human_outcomes"] for d in ensemble_data.values()], axis=0
    )
    all_avg_sim_outcomes = np.sum(
        [d["avg_sim_outcomes"] for d in ensemble_data.values()], axis=0
    )
    all_std_sim_outcomes = np.sqrt(
        np.sum([d["std_sim_outcomes"] ** 2 for d in ensemble_data.values()], axis=0)
    )

    n_subjects = len(ensemble_data)
    # title_label lets a caller name the panel itself. Without it the title is
    # built from the horizon string, which for a single-subject panel repeats the
    # subject id already in the prefix ("Subject 83 (long_subj83)") and appends an
    # uninformative (N=1).
    if title_label:
        title_head = title_label
        n_note = ""
    else:
        title_suffix = f" ({horizon})" if horizon else ""
        title_prefix = (f"Subject {list(ensemble_data.keys())[0]}"
                        if n_subjects == 1 else "All Subjects")
        title_head = f"{title_prefix}{title_suffix}"
        n_note = f" (N={n_subjects})"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x_draws = np.arange(len(all_human_counts))
    draw_labels = np.arange(draw_bin_start, draw_bin_start + len(all_human_counts))
    ax1.bar(x_draws, all_human_counts, alpha=0.5, color="gray", label="Human Draws")
    ax1.errorbar(
        x_draws,
        all_avg_sim_counts,
        yerr=all_std_sim_counts,
        fmt="-o",
        color="blue",
        capsize=4,
        label="Sim Draws Avg ± Std",
    )
    ax1.set_title(
        f"{title_head}: Number of Draws{n_note}",
        fontsize=font_size,
    )
    ax1.set_xlabel("Number of Draws", fontsize=font_size)
    ax1.set_ylabel("Count", fontsize=font_size)
    ax1.set_xticks(x_draws)
    ax1.set_xticklabels(draw_labels, fontsize=font_size - 2)
    ax1.grid(alpha=0.3)

    h_outcomes = all_human_outcomes[valid_indices]
    s_avg_outcomes = all_avg_sim_outcomes[valid_indices]
    s_std_outcomes = all_std_sim_outcomes[valid_indices]
    x_outcomes = np.arange(len(outcome_labels))
    ax2.bar(x_outcomes, h_outcomes, alpha=0.5, color="gray", label="Human Outcomes")
    ax2.errorbar(
        x_outcomes,
        s_avg_outcomes,
        yerr=s_std_outcomes,
        fmt="-o",
        color="orange",
        capsize=4,
        label="Sim Outcomes Avg ± Std",
    )
    ax2.set_title(
        f"{title_head}: Outcome Distribution{n_note}",
        fontsize=font_size,
    )
    ax2.set_xlabel("Outcome Value", fontsize=font_size)
    ax2.set_ylabel("Count", fontsize=font_size)
    ax2.set_xticks(x_outcomes)
    ax2.set_xticklabels(outcome_labels, fontsize=font_size)
    ax2.grid(alpha=0.3)
    legend_elements = [
        Patch(facecolor="gray", alpha=0.5, label="Human Data"),
        Line2D([0], [0], color="blue", marker="o", label="Sim Draws Avg ± Std"),
        Line2D([0], [0], color="orange", marker="o", label="Sim Outcomes Avg ± Std"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=font_size,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    suffix = f"_{horizon}" if horizon else ""
    _save_figure(fig, f"ensemble_all_subjects{suffix}", path)
    plt.close(fig)


def plot_ensemble_r2_metrics(
    metrics_df,
    horizon: str = "",
    font_size: int = 20,
    figsize: tuple[float, float] = (14, 6),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
    sort_by: str = "r2_draws",
):
    """Per-subject goodness of fit: R^2 (ensemble mean +/- SD) between human and
    simulated histograms, for number of draws and outcome.

    metrics_df : pandas.DataFrame
        One row per subject with columns "r2_draws", "r2_draws_var",
        "r2_outcome", "r2_outcome_var" -- the format produced by
        compute_subject_metrics()/build_ensemble_data() in
        notebooks/recovery_post_analysis.py (and cached by fit_data.py as
        ensemble_metrics_summary.csv).
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    df = metrics_df.copy()
    df["r2_draws_std"] = np.sqrt(df["r2_draws_var"])
    df["r2_outcome_std"] = np.sqrt(df["r2_outcome_var"])
    df = df.sort_values(sort_by).reset_index(drop=True)

    n = len(df)
    x = np.arange(n)
    color_draws = "#2166AC"
    color_outcome = "#D6604D"

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        x,
        df["r2_draws"],
        yerr=df["r2_draws_std"],
        fmt="-o",
        color=color_draws,
        ecolor=color_draws,
        alpha=0.85,
        capsize=3,
        markersize=4,
        linewidth=1.2,
        label=f"Draws (mean = {df['r2_draws'].mean():.2f})",
    )
    ax.errorbar(
        x,
        df["r2_outcome"],
        yerr=df["r2_outcome_std"],
        fmt="-o",
        color=color_outcome,
        ecolor=color_outcome,
        alpha=0.85,
        capsize=3,
        markersize=4,
        linewidth=1.2,
        label=f"Outcome (mean = {df['r2_outcome'].mean():.2f})",
    )

    ax.set_xlabel(
        f"Subject ID (sorted by R$^2$ {sort_by.split('_')[-1]})", fontsize=font_size
    )
    ax.set_ylabel(r"$R^2$", fontsize=font_size)
    ax.set_ylim(0, 1.02)
    # Label every tick with the subject's actual userID (not its sorted
    # position) -- matches plot_and_save_ensemble_metrics in
    # notebooks/recovery_post_analysis.ipynb, which this function mirrors.
    ax.set_xticks(x)
    ax.set_xticklabels(df["userID"], rotation=90, fontsize=font_size * 0.5)
    ax.set_xlim(-0.5, n - 0.5)
    title_suffix = f" ({horizon})" if horizon else ""
    ax.set_title(f"Model fit per subject{title_suffix} (N={n})", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size * 0.7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=font_size * 0.65, loc="lower right", frameon=True)
    plt.tight_layout()

    if save_fig:
        _save_figure(fig, f"ensemble_r2_metrics_{horizon}", path)

    return df


def plot_pomdp_vs_glm_fair_comparison(
    df_fair,
    w_p: float,
    font_size: int = 20,
    figsize: tuple[float, float] = (16, 5),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
):
    """Per-subject R^2 (draw-count fit to human data) for POMDP vs. GLM,
    fit under the same alignment/ensembling pipeline (see
    notebooks/Magda's_glm_fitting.ipynb, "FAIR COMPARISON" cell): scatter of
    per-subject R^2 (POMDP vs. GLM) plus a paired boxplot, annotated with the
    Wilcoxon signed-rank p-value from that same comparison.

    df_fair : pandas.DataFrame with columns "r2_pomdp", "r2_glm" (one row
        per subject).
    w_p : Wilcoxon signed-rank p-value for the paired r2_pomdp vs. r2_glm
        comparison, computed by the caller (scipy.stats.wilcoxon).
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.scatter(
        df_fair["r2_pomdp"],
        df_fair["r2_glm"],
        alpha=0.6,
        s=60,
        color="steelblue",
        edgecolors="black",
    )
    lim = [
        min(df_fair["r2_glm"].min(), df_fair["r2_pomdp"].min()),
        max(df_fair["r2_glm"].max(), df_fair["r2_pomdp"].max()),
    ]
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(r"POMDP $R^2$")
    ax.set_ylabel(r"GLM $R^2$")
    ax.set_title("Per-subject fit: POMDP vs GLM")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.boxplot([df_fair["r2_glm"], df_fair["r2_pomdp"]], tick_labels=["GLM", "POMDP"])
    ax2.set_ylabel(r"$R^2$ (draw-count fit to human data)")
    ax2.set_title(f"Paired comparison (p={w_p:.4f}, Wilcoxon)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_fig:
        _save_figure(fig, "fair_comparison_pomdp_vs_glm", path)

    return fig, axes


def plot_glm_performance(
    betas_all,
    font_size: int = 20,
    figsize: tuple[float, float] = (18, 10),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
):
    """Six-panel GLM performance overview across subjects: accuracy
    distribution, pseudo-R^2 distribution, AIC vs. accuracy, accuracy vs.
    sample size, and a boxplot of accuracy/pseudo-R^2 (see
    notebooks/Magda's_glm_fitting.ipynb, fit_glm_separate_for_human_data()
    for the betas_all record format).

    Returns the per-subject results DataFrame (or None if betas_all has no
    valid entries), matching the original notebook helper's behaviour of
    printing a summary table.
    """
    valid_results = [b for b in betas_all if b is not None]
    df_results = pd.DataFrame(valid_results)

    if len(df_results) == 0:
        print("No valid GLM results to plot")
        return None

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("GLM Performance Across Subjects", fontsize=font_size * 1.1, fontweight="bold")

    axes[0, 0].hist(df_results["accuracy"], bins=15, alpha=0.7, edgecolor="black", color="skyblue")
    axes[0, 0].axvline(
        df_results["accuracy"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_results['accuracy'].mean():.3f}",
    )
    axes[0, 0].set_xlabel("Accuracy")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Model Accuracy Distribution")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(bottom=0)

    axes[0, 1].hist(df_results["pseudo_r2"], bins=15, alpha=0.7, edgecolor="black", color="lightgreen")
    axes[0, 1].axvline(
        df_results["pseudo_r2"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_results['pseudo_r2'].mean():.3f}",
    )
    axes[0, 1].set_xlabel(r"Pseudo-$R^2$")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(r"Model Fit (Pseudo-$R^2$)")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(bottom=0)

    axes[0, 2].scatter(df_results["aic"], df_results["accuracy"], alpha=0.6, s=50)
    axes[0, 2].set_xlabel("AIC")
    axes[0, 2].set_ylabel("Accuracy")
    axes[0, 2].set_title("AIC vs Accuracy")

    axes[1, 0].scatter(df_results["n_trials"], df_results["accuracy"], alpha=0.6, s=50)
    axes[1, 0].set_xlabel("Number of Trials")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy vs Sample Size")

    metrics = ["accuracy", "pseudo_r2"]
    df_plot = df_results[metrics].melt(var_name="Metric", value_name="Score")
    sns.boxplot(data=df_plot, x="Metric", y="Score", ax=axes[1, 1])
    axes[1, 1].set_title("Distribution of Key Metrics")

    axes[1, 2].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_fig:
        _save_figure(fig, "glm_performance_summary", path)

    print("Performance Summary:")
    print(df_results[["accuracy", "pseudo_r2", "aic", "bic", "n_trials"]].describe())

    return df_results


def plot_dtd_histogram_by_horizon(
    pmat_long,
    pmat_short,
    font_size: int = 20,
    figsize: tuple[float, float] = (6, 4),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
):
    """Histogram of decision-time-to-decide (number of draws, "trial") for
    the long- vs. short-horizon GLM regressor tables (pmat_long/pmat_short
    from assemble_glm_outputs(), see notebooks/Magda's_glm_fitting.ipynb).
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(0, 15) - 0.5
    ax.hist(
        pmat_long["trial"].dropna(),
        bins=bins,
        density=True,
        edgecolor="w",
        label="long",
        color="#00a676",
    )
    ax.hist(
        pmat_short["trial"].dropna(),
        bins=bins,
        density=True,
        edgecolor="w",
        label="short",
        color="#4e4187",
    )
    ax.set_xlim([0, 14])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    ax.set_title("Histogram of DTD by horizon")
    ax.set_xlabel("trial")
    ax.set_ylabel("probability")
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, "dtd_histogram_by_horizon", path)

    return fig, ax


def plot_single_subject_glm_fit_corrected(
    ax,
    human_decisions,
    glm_truncated_games,
    games_lengths,
    target_id: int | str | None = None,
    ensemble_data: dict | None = None,
    metrics_df: pd.DataFrame | None = None,
):
    """Draws human / POMDP-ensemble / corrected-GLM draw-count histograms for
    one subject onto `ax` (a panel of a larger grid figure -- see
    plot_glm_subject_grid). "Corrected" means GLM draws use a single
    stochastic realization, with never-decided games counted at full game
    length (compute_corrected_glm_draws in data_handling.py).
    """
    # NOTE: human draw counts come from ens_data["human_counts"] below, not
    # from diffing `human_decisions`==1 positions. `human_decisions` is
    # decide[chidx] -- a filtered subset of the raw per-draw sequence
    # (draw-1-of-game excluded) -- so that reconstruction silently drops
    # every game where the subject never decided, undercounting relative to
    # the POMDP/GLM curves (which both count those games at full length).
    # `human_decisions` is only used here as a fallback when no cached
    # ensemble data exists for this subject.
    human_decisions = np.asarray(human_decisions).flatten()
    human_indices = np.where(human_decisions == 1)[0]

    ens_data = None
    if ensemble_data is not None and target_id is not None:
        ens_data = ensemble_data.get(target_id)

    if ens_data is None and len(human_indices) == 0:
        ax.text(0.5, 0.5, f"No Human Data\nUser {target_id}", ha="center", va="center")
        return

    glm_draws = compute_corrected_glm_draws(glm_truncated_games, games_lengths)

    if ens_data is not None:
        x_draws = np.arange(1, len(ens_data["human_counts"]) + 1)
        human_counts = pad_to_length(ens_data["human_counts"], len(x_draws))
    else:
        human_draws = [human_indices[0] + 1]
        if len(human_indices) > 1:
            human_draws.extend(np.diff(human_indices).tolist())
        min_draws = min(min(human_draws), min(glm_draws)) if glm_draws else min(human_draws)
        max_draws = max(max(human_draws), max(glm_draws)) if glm_draws else max(human_draws)
        x_draws = np.arange(min_draws, max_draws + 1)
        human_counts = [human_draws.count(x) for x in x_draws]

    glm_counts = [glm_draws.count(x) for x in x_draws]

    ax.bar(x_draws, human_counts, alpha=0.5, color="gray", label="Human Data")

    if ens_data is not None:
        ax.errorbar(
            x_draws,
            ens_data["avg_sim_counts"],
            yerr=ens_data["std_sim_counts"],
            fmt="-o",
            color="blue",
            capsize=3,
            markersize=4,
            label="Ensemble Sim",
        )

    if glm_draws:
        ax.plot(
            x_draws, glm_counts, "-o", color="green", linewidth=1.5, markersize=4,
            label="GLM (Corrected)",
        )

    title_text = f"User {target_id}" if target_id is not None else ""
    if metrics_df is not None and target_id is not None:
        user_metrics = metrics_df[metrics_df["userID"] == target_id]
        if not user_metrics.empty:
            r2_draws = user_metrics.iloc[0]["r2_draws"]
            title_text += rf" ($R^2$ = {r2_draws:.2f})"

    ax.set_title(title_text, fontsize=11, fontweight="bold")
    ax.set_xticks(x_draws)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_subject_glm_fit_ensemble(
    ax,
    human_decisions,
    probabilities,
    games_lengths,
    target_id: int | str | None = None,
    ensemble_data: dict | None = None,
    metrics_df: pd.DataFrame | None = None,
    n_samples: int = 200,
):
    """Draws human / POMDP-ensemble / ensemble-averaged-GLM draw-count
    histograms for one subject onto `ax` (a panel of a larger grid figure --
    see plot_glm_subject_grid). The GLM curve is a mean +/- std over
    `n_samples` stochastic realizations of `probabilities` (per-draw decide
    probabilities aligned to `games_lengths` -- see
    compute_full_per_draw_probabilities in src/glm/glm.py), instead of a
    single binomial draw.
    """
    # NOTE: human draw counts come from ens_data["human_counts"] below, not
    # from diffing `human_decisions`==1 positions. `human_decisions` is
    # decide[chidx] -- a filtered subset of the raw per-draw sequence
    # (draw-1-of-game excluded) -- so that reconstruction silently drops
    # every game where the subject never decided, undercounting relative to
    # the POMDP/GLM curves (which both count those games at full length).
    # `human_decisions` is only used here as a fallback when no cached
    # ensemble data exists for this subject.
    human_decisions = np.asarray(human_decisions).flatten()
    human_indices = np.where(human_decisions == 1)[0]

    ens_data = None
    if ensemble_data is not None and target_id is not None:
        ens_data = ensemble_data.get(target_id)

    if ens_data is None and len(human_indices) == 0:
        ax.text(0.5, 0.5, f"No Human Data\nUser {target_id}", ha="center", va="center")
        return

    if ens_data is not None:
        x_draws = np.arange(1, len(ens_data["human_counts"]) + 1)
        human_counts = pad_to_length(ens_data["human_counts"], len(x_draws))
    else:
        human_draws = [human_indices[0] + 1]
        if len(human_indices) > 1:
            human_draws.extend(np.diff(human_indices).tolist())
        max_draws = max(max(human_draws), max(games_lengths))
        x_draws = np.arange(min(human_draws), max_draws + 1)
        human_counts = [human_draws.count(x) for x in x_draws]

    glm_counts_mean, glm_counts_std = compute_ensemble_glm_counts(
        probabilities, games_lengths, x_draws, n_samples=n_samples
    )

    ax.bar(x_draws, human_counts, alpha=0.5, color="gray", label="Human Data")

    if ens_data is not None:
        ax.errorbar(
            x_draws,
            ens_data["avg_sim_counts"],
            yerr=ens_data["std_sim_counts"],
            fmt="-o",
            color="blue",
            capsize=3,
            markersize=4,
            label="Ensemble Sim",
        )

    ax.errorbar(
        x_draws,
        glm_counts_mean,
        yerr=glm_counts_std,
        fmt="-o",
        color="green",
        linewidth=1.5,
        markersize=4,
        capsize=3,
        label="GLM (Ensemble Avg)",
    )

    title_text = f"User {target_id}" if target_id is not None else ""
    if metrics_df is not None and target_id is not None:
        user_metrics = metrics_df[metrics_df["userID"] == target_id]
        if not user_metrics.empty:
            r2_draws = user_metrics.iloc[0]["r2_draws"]
            title_text += rf" ($R^2$ = {r2_draws:.2f})"

    ax.set_title(title_text, fontsize=11, fontweight="bold")
    ax.set_xticks(x_draws)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_glm_subject_grid(
    betas_all,
    ensemble_data_long,
    ensemble_data_short,
    filtered_human_data,
    n_subjects: int | None = None,
    n_cols: int = 3,
    n_samples: int = 200,
    font_size: int = 20,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
):
    """Grid of per-subject panels (plot_single_subject_glm_fit_ensemble),
    one per subject in `betas_all`, comparing human draw counts against the
    combined long+short POMDP ensemble and the ensemble-averaged GLM. See
    notebooks/Magda's_glm_fitting.ipynb for the betas_all/filtered_human_data
    record formats.
    """
    from src.glm.glm import compute_full_per_draw_probabilities

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    total_subjects = min(n_subjects, len(betas_all)) if n_subjects else len(betas_all)
    n_rows = int(np.ceil(total_subjects / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for i in range(total_subjects):
        ax = axes_flat[i]
        subject_data = betas_all[i]

        games_lengths = subject_data["games_lengths"]
        human_decisions_1d = subject_data["decide"]
        uid = subject_data["id"]

        row_data = filtered_human_data.loc[filtered_human_data["userID"] == uid, "data"].iloc[0]
        probabilities = compute_full_per_draw_probabilities(
            row_data, games_lengths,
            subject_data["mu"], subject_data["sigma"], subject_data["pdecide_beta"],
        )

        combined_ensemble_data = {uid: combine_ensemble_horizons(ensemble_data_long, ensemble_data_short, uid)}

        plot_single_subject_glm_fit_ensemble(
            ax=ax,
            human_decisions=human_decisions_1d,
            probabilities=probabilities,
            games_lengths=games_lengths,
            target_id=uid,
            ensemble_data=combined_ensemble_data,
            metrics_df=None,
            n_samples=n_samples,
        )

    for j in range(total_subjects, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    axes_flat[0].legend(loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()

    if save_fig:
        _save_figure(fig, "glm_subject_grid", path)

    return fig, axes


def plot_glm_pooled_comparison(
    betas_all,
    ensemble_data_long,
    ensemble_data_short,
    filtered_human_data,
    n_samples: int = 200,
    font_size: int = 20,
    figsize: tuple[float, float] = (10, 5),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
):
    """Pools every subject's draw-count histogram onto one shared draw-index
    x-axis (instead of one subplot per subject -- see plot_glm_subject_grid)
    to compare human vs. combined long+short POMDP ensemble vs.
    ensemble-averaged GLM in aggregate.

    Returns (fig, ax, totals) where `totals` is a dict with x_draws_all,
    total_human_counts, total_ens_mean, total_ens_std, total_glm_mean,
    total_glm_std, n_subjects_total -- reused by plot_per_draw_discrepancy
    for the companion sanity-check plot.
    """
    from src.glm.glm import compute_full_per_draw_probabilities

    n_subjects_total = len(betas_all)

    max_draws = max(max(betas_all[i]["games_lengths"]) for i in range(n_subjects_total))
    x_draws_all = np.arange(1, max_draws + 1)

    total_human_counts = np.zeros(len(x_draws_all))
    total_ens_mean = np.zeros(len(x_draws_all))
    total_ens_var = np.zeros(len(x_draws_all))
    total_glm_mean = np.zeros(len(x_draws_all))
    total_glm_var = np.zeros(len(x_draws_all))

    for i in range(n_subjects_total):
        subject_data = betas_all[i]
        games_lengths = subject_data["games_lengths"]
        uid = subject_data["id"]

        # NOTE: human draw counts must come from the cached combined
        # human_counts below, not from subject_data["decide"]. That array is
        # decide[chidx] -- a filtered subset of the raw per-draw sequence
        # (draw-1-of-game excluded) -- so diffing its decide==1 positions
        # does not recover real within-game draw counts and silently drops
        # every game where the subject never decided (undercounting the
        # human total relative to POMDP/GLM, which both count those games at
        # full length).
        combined = combine_ensemble_horizons(ensemble_data_long, ensemble_data_short, uid)
        if combined is not None:
            total_human_counts += pad_to_length(combined["human_counts"], len(x_draws_all))
            total_ens_mean += pad_to_length(combined["avg_sim_counts"], len(x_draws_all))
            total_ens_var += pad_to_length(combined["std_sim_counts"], len(x_draws_all)) ** 2

        row_data = filtered_human_data.loc[filtered_human_data["userID"] == uid, "data"].iloc[0]
        probabilities = compute_full_per_draw_probabilities(
            row_data, games_lengths,
            subject_data["mu"], subject_data["sigma"], subject_data["pdecide_beta"],
        )
        glm_mean, glm_std = compute_ensemble_glm_counts(
            probabilities, games_lengths, x_draws_all, n_samples=n_samples
        )
        total_glm_mean += glm_mean
        total_glm_var += glm_std**2

    total_ens_std = np.sqrt(total_ens_var)
    total_glm_std = np.sqrt(total_glm_var)

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_draws_all, total_human_counts, alpha=0.5, color="gray", label="Human Data")
    ax.errorbar(
        x_draws_all, total_ens_mean, yerr=total_ens_std,
        fmt="-o", color="blue", capsize=3, markersize=4, label="Ensemble Sim",
    )
    ax.errorbar(
        x_draws_all, total_glm_mean, yerr=total_glm_std,
        fmt="-o", color="green", linewidth=1.5, markersize=4, capsize=3,
        label="GLM (Ensemble Avg)",
    )

    ax.set_title(f"All Subjects Pooled (N={n_subjects_total})", fontsize=font_size * 0.8, fontweight="bold")
    ax.set_xlabel("Draw index")
    ax.set_ylabel("Count of decisions")
    ax.set_xticks(x_draws_all)
    ax.tick_params(axis="both", labelsize=font_size * 0.55)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=font_size * 0.55, frameon=False)

    fig.tight_layout()

    if save_fig:
        _save_figure(fig, "glm_pooled_comparison", path)

    totals = {
        "x_draws_all": x_draws_all,
        "total_human_counts": total_human_counts,
        "total_ens_mean": total_ens_mean,
        "total_ens_std": total_ens_std,
        "total_glm_mean": total_glm_mean,
        "total_glm_std": total_glm_std,
        "n_subjects_total": n_subjects_total,
    }
    return fig, ax, totals


def plot_per_draw_discrepancy(
    x_draws_all,
    total_glm_mean,
    total_human_counts,
    font_size: int = 20,
    figsize: tuple[float, float] = (10, 3.5),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
):
    """Per-draw (GLM count - human count) bar chart, pooled across subjects
    -- the companion sanity-check plot for plot_glm_pooled_comparison's
    totals dict, showing where in the game the GLM over/under-shoots humans.
    """
    _set_plot_style(font_size=font_size, is_latex=is_latex)

    diff_per_draw = total_glm_mean - total_human_counts

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_draws_all, diff_per_draw, color=np.where(diff_per_draw >= 0, "green", "gray"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Draw index")
    ax.set_ylabel("GLM count - Human count")
    ax.set_title("Per-draw discrepancy: GLM (ensemble avg) vs Human, pooled across subjects")
    ax.set_xticks(x_draws_all)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, "per_draw_discrepancy", path)

    return fig, ax


def plot_glmm_betas_auto(
    coefs_df,
    main_effects: list[str] | None = None,
    font_size: int = 20,
    main_ylim: tuple[float, float] = (-3, 3),
    interaction_ylim: tuple[float, float] = (-0.4, 0.4),
    figsize: tuple[float, float] = (10, 7),
    is_latex: bool = IS_LATEX,
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    fname: str = "glmm_betas_auto.pdf",
):
    """Plot GLMM coefficients from a plain coefficient table (a DataFrame
    with "Estimate"/"Std. Error"/"Pr(>|z|)" columns, e.g. a statsmodels
    `.summary2().tables[1]`-style frame), auto-detecting FA2 interaction
    terms by name (any index containing ":FA2") rather than requiring them
    to be passed explicitly -- unlike plot_glmm_betas, which takes a fitted
    GLM results object.
    """
    if main_effects is None:
        main_effects = [
            "totevminus", "deltaev", "trial", "termination",
            "FA2:totevminus", "FA2:deltaev", "FA2:trial", "FA2:termination",
        ]
    interaction_effects = [name for name in coefs_df.index if ":FA2" in name]

    est_main = coefs_df.loc[coefs_df.index.intersection(main_effects), "Estimate"]
    se_main = coefs_df.loc[coefs_df.index.intersection(main_effects), "Std. Error"]
    err_main = 1.96 * se_main
    p_main = coefs_df.loc[coefs_df.index.intersection(main_effects), "Pr(>|z|)"]

    est_inter = coefs_df.loc[coefs_df.index.intersection(interaction_effects), "Estimate"]
    se_inter = coefs_df.loc[coefs_df.index.intersection(interaction_effects), "Std. Error"]
    err_inter = 1.96 * se_inter
    p_inter = coefs_df.loc[coefs_df.index.intersection(interaction_effects), "Pr(>|z|)"]

    def p_to_stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)
    x_main = np.arange(len(est_main))
    x_inter = np.arange(len(est_main), len(est_main) + len(est_inter))

    ax.errorbar(
        x_main, est_main, yerr=err_main, fmt="o",
        color="k", elinewidth=2, capsize=5, markersize=8,
    )

    ax2 = ax.twinx()
    ax2.errorbar(
        x_inter, est_inter, yerr=err_inter, fmt="o",
        color="b", elinewidth=2, capsize=5, markersize=8,
    )

    ax.axhline(0, linestyle="--", linewidth=1.25)

    main_offset = (main_ylim[1] - main_ylim[0]) * 0.03
    inter_offset = (interaction_ylim[1] - interaction_ylim[0]) * 0.03
    for x, beta, err, p in zip(x_main, est_main, err_main, p_main):
        stars = p_to_stars(p)
        if stars:
            ax.text(x, beta + err + main_offset, stars, ha="center", va="bottom", color="k", fontsize=font_size * 0.8)
    for x, beta, err, p in zip(x_inter, est_inter, err_inter, p_inter):
        stars = p_to_stars(p)
        if stars:
            ax2.text(x, beta + err + inter_offset, stars, ha="center", va="bottom", color="b", fontsize=font_size * 0.8)

    ax.set_ylabel("Beta Estimate (Main Effects)", color="k")
    ax.set_ylim(main_ylim)
    ax.tick_params(axis="y", colors="k")
    ax2.set_ylabel("Beta Estimate (FA2 Interactions)", color="b")
    ax2.set_ylim(interaction_ylim)
    ax2.tick_params(axis="y", colors="b")

    all_names = list(est_main.index) + [x.replace("FA2:", "OC:") for x in est_inter.index]
    ax.set_xticks(np.concatenate([x_main, x_inter]))
    ax.set_xticklabels(all_names, rotation=45, ha="right")
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, (ax, ax2)


def plot_model_accuracies_decide_action(
    df_results,
    font_size: int = 20,
    figsize: tuple[float, float] = (9, 5),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
    fname: str = "decide_action_accuracy_distribution.pdf",
):
    """Histogram of the two-stage GLM's "decide vs. wait" accuracy and
    "yellow vs. blue" action-choice accuracy across subjects, against
    chance level (0.5 for a binary choice).
    """
    decide_acc = df_results["accuracy"].dropna()
    action_acc = df_results["action_accuracy"].dropna()

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(0.3, 1.05, 0.05)
    ax.hist(
        decide_acc, bins=bins, alpha=0.6,
        label=f"Decision Model\n(Mean: {decide_acc.mean():.2f})",
        color="#4e4187", edgecolor="w",
    )
    ax.hist(
        action_acc, bins=bins, alpha=0.6,
        label=f"Action Choice Model\n(Mean: {action_acc.mean():.2f})",
        color="#00a676", edgecolor="w",
    )
    ax.axvline(0.5, color="red", linestyle="dashed", linewidth=2, label="Chance Level (0.50)")

    ax.set_title("Distribution of GLM Prediction Accuracies")
    ax.set_xlabel("Prediction Accuracy")
    ax.set_ylabel("Number of Subjects")
    ax.set_xlim([0.3, 1.05])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, ax


def plot_multinomial_model_accuracy(
    df_results,
    font_size: int = 20,
    figsize: tuple[float, float] = (9, 5),
    path: str = DEFAULT_FIGURE_PATH,
    save_fig: bool = True,
    is_latex: bool = IS_LATEX,
    fname: str = "multinomial_accuracy_distribution.pdf",
):
    """Histogram of the 3-class multinomial GLM's (wait / yellow / blue)
    prediction accuracy across subjects, against chance level (1/3 for a
    3-class choice).
    """
    multi_acc = df_results["multi_accuracy"].dropna()

    _set_plot_style(font_size=font_size, is_latex=is_latex)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        multi_acc, bins=np.arange(0.2, 1.05, 0.05), alpha=0.7,
        label=f"Multinomial Model\n(Mean: {multi_acc.mean():.2f})",
        color="#ff9f1c", edgecolor="w",
    )
    ax.axvline(0.333, color="red", linestyle="dashed", linewidth=2, label="Chance Level (~0.33)")

    ax.set_title("Distribution of Multinomial Prediction Accuracy")
    ax.set_xlabel("Prediction Accuracy (Wait vs Yellow vs Blue)")
    ax.set_ylabel("Number of Subjects")
    ax.set_xlim([0.2, 1.05])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_fig:
        _save_figure(fig, fname, path)

    return fig, ax


# --- Model comparison (AIC/BIC/AICc), from model_comparison.ipynb ---


def _ensure_loglikelihoods(ll_values) -> np.ndarray:
    """Ensure log-likelihood values are negative and handle NaNs."""
    arr = np.array(ll_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([])
    return -arr if arr.mean() > 0 else arr


def compute_metrics(ll_sum: float, n_params: int, n_obs: int) -> dict:
    """Computes AIC, BIC, and AICc for a given model."""
    AIC = 2 * n_params - 2 * ll_sum
    BIC = np.log(n_obs) * n_params - 2 * ll_sum
    AICc = (
        AIC + (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
        if n_obs > n_params + 1
        else np.nan
    )
    return {
        "sum logL": ll_sum,
        "k": n_params,
        "n_obs": n_obs,
        "AIC": AIC,
        "AICc": AICc,
        "BIC": BIC,
    }


def _per_subject_n_obs(subject_data) -> int:
    """Draw-level observation count for a single subject's data_dict_of_lists
    (horizon -> DataFrame, one row per game, 'draw_yellow_blue_action_outcome'
    holding the per-draw action sequence). Single-subject counterpart to
    estimate_n_obs_for_subjects, factored out so per-subject BIC/AIC/AICc
    (compute_metrics_per_subject_summed) can use each subject's own n_obs
    instead of one pooled total.
    """
    # Forgetting-model rows store this as a pandas Series (gamma-indexed
    # lookup upstream), not a plain dict; Series.values is a property
    # (not a method), so .values() would try to call an ndarray. Iterating
    # a dict yields keys, so dicts still need .values() -- but iterating a
    # Series already yields its values, matching dict.values() semantics.
    horizon_dfs = subject_data.values() if isinstance(subject_data, dict) else subject_data
    return sum(
        len(seq)
        for horizon_df in horizon_dfs
        for seq in horizon_df["draw_yellow_blue_action_outcome"].values
    )


def estimate_n_obs_for_subjects(results_df: pd.DataFrame) -> int:
    """Returns the total number of draw-level observations across all subjects.

    data_dict_of_lists maps horizon -> DataFrame where each row is one game
    and 'draw_yellow_blue_action_outcome' holds the per-draw action sequence.
    Summing sequence lengths across all games, horizons, and subjects gives
    the exact draw count that the POMDP (and GLM) likelihood is evaluated on.
    """
    return sum(_per_subject_n_obs(subject_data) for subject_data in results_df["data_dict_of_lists"])


def compute_metrics_per_subject_summed(df: pd.DataFrame, ll_col: str = "after_lls_ga") -> dict:
    """Statistically correct aggregate AIC/AICc/BIC across N independently
    fit subjects: each subject i is its own model fit with its own k_i,
    n_obs_i, and log-likelihood, so the population-level metric is the *sum*
    of each subject's own compute_metrics(...) result -- not
    compute_metrics(ll.sum(), k, pooled_n_obs) computed once from pooled
    numbers.

    Why this matters: since subjects are fit independently, the joint
    marginal likelihood over the whole sample factors as a product across
    subjects, so -2*log(.) of it is a *sum* of per-subject Schwarz terms:
    BIC_total = sum_i(k_i*log(n_obs_i) - 2*ll_i). This equals
    compute_metrics(ll.sum(), k, pooled_n_obs) only in the special case
    where every subject contributes exactly the same n_obs_i -- which does
    not hold here (games can end after any draw, so per-subject draw counts
    vary). Matches the convention already used for personalized model
    selection (per_subject_model_selection.py).

    AICc is summed per-subject too (not recomputed from the summed AIC),
    since AICc's correction term is itself a function of each subject's own
    n_obs_i and k_i; if any subject's own AICc is undefined (n_obs_i too
    small relative to k_i), the summed AICc is NaN throughout.
    """
    n_params = len(df["fit_params_ga"].iloc[0])
    # Same sign convention as _ensure_loglikelihoods (log-likelihoods should
    # be negative; flip if the column is stored as positive), but applied
    # per-row here -- rather than calling _ensure_loglikelihoods on the whole
    # column, which also *drops* NaN rows and would desync the remaining
    # values from their matching data_dict_of_lists row during the zip below.
    raw_ll = np.asarray(df[ll_col].tolist(), dtype=float)
    sign = -1.0 if np.nanmean(raw_ll) > 0 else 1.0
    # "k" is the per-subject free-parameter count (the same for every
    # subject fitting this one model structure) -- NOT summed across
    # subjects. n_obs is deliberately not reported here: unlike k, it
    # genuinely varies subject to subject (draw counts differ), so there is
    # no single meaningful "n_obs" for the model/row as a whole to display;
    # each subject's own n_obs_i is still used internally, per subject,
    # to compute that subject's own AIC/AICc/BIC term below.
    totals = {"sum logL": 0.0, "k": n_params, "AIC": 0.0, "AICc": 0.0, "BIC": 0.0}
    aicc_nan = False
    for ll_raw, subject_data in zip(raw_ll, df["data_dict_of_lists"]):
        if np.isnan(ll_raw):
            continue
        ll_i = sign * ll_raw
        n_obs_i = _per_subject_n_obs(subject_data)
        m_i = compute_metrics(ll_i, n_params, n_obs_i)
        totals["sum logL"] += m_i["sum logL"]
        totals["AIC"] += m_i["AIC"]
        totals["BIC"] += m_i["BIC"]
        if np.isnan(m_i["AICc"]):
            aicc_nan = True
        else:
            totals["AICc"] += m_i["AICc"]
    if aicc_nan:
        totals["AICc"] = np.nan
    return totals


def get_best_model(metrics_dict: dict, criterion: str = "BIC"):
    """Determines the best model from a dict of metrics dicts, by a criterion."""
    if not metrics_dict:
        return None
    return min(metrics_dict, key=lambda model: metrics_dict[model][criterion])


def plot_compare_n_models_latex(
    model_names: list[str],
    results_df_dict: dict,
    ll_col: str = "after_lls_ga",
    export_table_path: str | None = None,
    break_at: bool | float | None = None,
    font_size: int = 20,
    is_latex: bool = IS_LATEX,
    figure_height: float = 6,
    path: str = DEFAULT_FIGURE_PATH,
    y_min: float | None = None,
    top_n: int | None = None,
):
    """Compares an arbitrary number of models, generates either a standard plot or
    a grid layout plot with a disconnected (broken) y-axis to handle extreme outliers,
    and exports a LaTeX table.

    break_at : bool or float, optional
        - False: Force a single standard plot without any Y-axis breaks.
        - True: Force a broken Y-axis plot (using auto-detected split point).
        - None: Automatically detect if a split is needed using a median-based threshold.
        - float: Manually specify the Y-axis value where the cut/break should happen.

    y_min : float, optional
        Lower bound of the (bottom, in the broken-axis case) Y-axis. Defaults
        to 0, but can be set to a value closer to the smallest plotted bar to
        zoom in on differences between closely-matched models.

    top_n : int, optional
        If given and fewer than len(model_names), only the top_n models with
        the lowest BIC are kept in the exported table and the figure (deltas
        are still computed against the best model of the *full* set passed
        in, so ΔBIC/ΔAIC/ΔAICc remain meaningful even when truncated).
    """
    metrics_dict = {}

    def to_latex_tt(name: str) -> str:
        # Real, math-mode-wrapped LaTeX -- used for the exported .tex table
        # (and caption), not matplotlib. Renders a trailing "_F"/"_NF"
        # condition tag as a subscript instead of a literal underscore.
        return f"${_texttt_with_condition_subscript(name)}$"

    def display_name_for(name: str) -> str:
        # matplotlib label/title: only valid under real usetex (mathtext
        # doesn't know \texttt and would raise a parse error).
        return to_latex_tt(name) if is_latex else _display_task_name(name)

    for model_name in model_names:
        df = results_df_dict[model_name]
        metrics_dict[model_name] = compute_metrics_per_subject_summed(df, ll_col)

    print(f"--- Comparison: {', '.join(model_names)} ---")
    best_model_aic = get_best_model(metrics_dict, "AIC")
    best_model_bic = get_best_model(metrics_dict, "BIC")
    best_model_aicc = min(
        metrics_dict,
        key=lambda m: (
            metrics_dict[m]["AICc"] if not np.isnan(metrics_dict[m]["AICc"]) else np.inf
        ),
    )
    print(f"Best model (AIC): {best_model_aic}")
    print(f"Best model (AICc): {best_model_aicc}")
    print(f"Best model (BIC): {best_model_bic}\n")
    if best_model_aic != best_model_bic or best_model_aicc != best_model_bic:
        print(f"  Note: AIC/AICc winner differs from the BIC winner; ΔAIC/ΔAICc/ΔBIC "
              f"below are all relative to the BIC winner ({best_model_bic}), not to each "
              f"criterion's own separate winner, so only the BIC winner's row is all-zero.")
    best_model = best_model_bic

    # All three deltas are relative to the single BIC winner (not each
    # criterion's own separately-optimal model): BIC is treated as primary
    # throughout this manuscript, and a bolded "best model" row whose ΔAIC
    # was computed against a *different* model would be confusing to read
    # (its own ΔAIC could be nonzero even though it's the bolded row).
    for m in model_names:
        metrics_dict[m]["ΔAIC"] = metrics_dict[m]["AIC"] - metrics_dict[best_model]["AIC"]
        metrics_dict[m]["ΔAICc"] = metrics_dict[m]["AICc"] - metrics_dict[best_model]["AICc"]
        metrics_dict[m]["ΔBIC"] = metrics_dict[m]["BIC"] - metrics_dict[best_model]["BIC"]

    # Restrict the table/figure to the top_n models with the lowest BIC
    # (deltas above were already computed against the full model_names set,
    # so they stay correct even after truncation).
    display_models = sorted(model_names, key=lambda m: metrics_dict[m]["BIC"])
    if top_n is not None and len(model_names) > top_n:
        display_models = display_models[:top_n]
        print(
            f"Showing the top {top_n} of {len(model_names)} models by BIC "
            f"in the table/figure."
        )

    if export_table_path:
        df_latex = pd.DataFrame.from_dict(metrics_dict, orient="index").reset_index()
        df_latex = df_latex.rename(columns={"index": "Model"})
        df_latex = df_latex[df_latex["Model"].isin(display_models)]
        # Order rows by ascending ΔBIC (least-to-biggest evidence against), so
        # the best-supported model (ΔBIC = 0) appears first, per PLoS CB
        # convention for model-comparison tables.
        df_latex = df_latex.sort_values("ΔBIC", ascending=True).reset_index(drop=True)
        is_best_row = np.isclose(df_latex["ΔBIC"], 0.0)
        df_latex["Model"] = df_latex["Model"].apply(to_latex_tt)
        # k is the per-subject free-parameter count (compute_metrics_per_subject_summed),
        # constant across all subjects fitting this one model. n_obs is
        # omitted -- it genuinely varies subject to subject, so there is no
        # single "n_obs" meaningful to report for the model/row as a whole.
        cols_order = ["Model", "sum logL", "k", "ΔAIC", "ΔAICc", "ΔBIC"]
        df_latex = df_latex[cols_order].rename(
            columns={
                "k": r"$N_{\text{params}}$",
                "ΔAIC": r"$\Delta$AIC",
                "ΔAICc": r"$\Delta$AICc",
                "ΔBIC": r"$\Delta$BIC",
            }
        )

        # Pre-format every cell to a string (matching the previous
        # float_format="%.2f" / na_rep="NaN" behaviour), then bold the whole
        # row of the best (ΔBIC = 0) model to highlight it in the table.
        int_cols = {r"$N_{\text{params}}$"}

        def _fmt_cell(col, val):
            if col == "Model":
                return val
            if isinstance(val, float) and np.isnan(val):
                return "NaN"
            if col in int_cols:
                return str(int(val))
            return f"{val:.2f}"

        for col in df_latex.columns:
            df_latex[col] = df_latex[col].apply(lambda v, c=col: _fmt_cell(c, v))
        for idx in df_latex.index[is_best_row]:
            for col in df_latex.columns:
                df_latex.at[idx, col] = r"\textbf{" + df_latex.at[idx, col] + "}"

        tabular = df_latex.to_latex(
            index=False,
            escape=False,
            column_format="lccccc",
        )
        truncation_note = (
            f" Showing the top {len(display_models)} of {len(model_names)} "
            "models by BIC."
            if len(display_models) < len(model_names)
            else ""
        )
        # No surrounding \begin{table}/\caption/\label here -- this file is
        # meant to be \input{} inside a \begin{table}...\end{table} that the
        # paper itself defines (with its own caption/label), so it must not
        # carry its own (a nested table environment raises "LaTeX Error: Not
        # in outer par mode").
        latex_table = (
            tabular
            + f"\\par\\footnotesize Note:{truncation_note} Rows are ordered by "
            "ascending $\\Delta$BIC; the best-supported (BIC-winning) model "
            "($\\Delta$BIC $= 0$) is shown in bold. $\\Delta$AIC, $\\Delta$AICc, "
            "and $\\Delta$BIC are all relative to this same BIC-winning model, "
            "so a negative $\\Delta$AIC/$\\Delta$AICc means that row is "
            "actually preferred over the bolded row on that criterion "
            "specifically; $k = N_{\\text{params}}$.\n"
        )
        with open(export_table_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table exported to {export_table_path}")

    metrics_names = ["ΔAIC", "ΔAICc", "ΔBIC"]
    models_to_plot = [m for m in display_models if m != best_model]
    if not models_to_plot:
        print("Only the best model is present. Skipping plot generation.")
        return {
            "metrics_table": pd.DataFrame.from_dict(metrics_dict, orient="index")
        }, best_model

    model_max_vals = {}
    all_plotted_values = []
    for m in models_to_plot:
        vals = [
            metrics_dict[m][met]
            for met in metrics_names
            if not np.isnan(metrics_dict[m][met])
        ]
        model_max_vals[m] = max(vals) if vals else 0
        all_plotted_values.extend(vals)
    all_max_values = list(model_max_vals.values())
    global_max = max(all_max_values) if all_max_values else 100
    # A criterion can favor a different model than the BIC winner these deltas
    # are computed against (e.g. AIC/AICc preferring a less-penalized
    # candidate), which shows up as a negative delta; a 0-anchored bottom
    # would silently clip that bar off rather than showing it.
    global_min = min(all_plotted_values) if all_plotted_values else 0

    should_break = False
    if break_at is False:
        should_break = False
    elif isinstance(break_at, (int, float)) and not isinstance(break_at, bool):
        should_break = True
        cutoff = break_at
    else:
        non_zero_scales = [v for v in all_max_values if v > 1e-1]
        if non_zero_scales:
            auto_cutoff = 3.5 * np.median(non_zero_scales)
            has_outliers = any(model_max_vals[m] > auto_cutoff for m in models_to_plot)
            should_break = True if break_at is True else has_outliers
            cutoff = auto_cutoff if should_break else global_max
        else:
            should_break = True if break_at is True else False
            cutoff = global_max * 0.5

    if should_break:
        normal_models = [m for m in models_to_plot if model_max_vals[m] <= cutoff]
        outlier_models = [m for m in models_to_plot if model_max_vals[m] > cutoff]
        if not normal_models:
            normal_models = sorted(models_to_plot, key=lambda m: model_max_vals[m])[:1]
            outlier_models = [m for m in models_to_plot if m not in normal_models]

        normal_vals = [
            metrics_dict[m][met]
            for m in normal_models
            for met in metrics_names
            if not np.isnan(metrics_dict[m][met])
        ]
        outlier_vals = [
            metrics_dict[m][met]
            for m in outlier_models
            for met in metrics_names
            if not np.isnan(metrics_dict[m][met])
        ]

        if isinstance(break_at, (int, float)) and not isinstance(break_at, bool):
            bot_upper_limit = break_at
            top_lower_limit = min(outlier_vals) * 0.90 if outlier_vals else break_at
            if top_lower_limit < bot_upper_limit:
                top_lower_limit = bot_upper_limit + 50
        else:
            max_normal = max(normal_vals) if normal_vals else 100
            bot_upper_limit = max_normal * 1.15
            top_lower_limit = (
                min(outlier_vals) * 0.95 if outlier_vals else bot_upper_limit
            )

        max_outlier = max(outlier_vals) if outlier_vals else bot_upper_limit
        top_upper_limit = max_outlier * 1.05

    _set_plot_style(font_size=font_size, is_latex=is_latex)
    n_models_plot = len(models_to_plot)
    fig_width = max(10, 5 * min(3, n_models_plot))
    x = np.arange(len(metrics_names))
    total_width = 0.8
    width = total_width / n_models_plot
    offsets = np.linspace(
        -total_width / 2 + width / 2, total_width / 2 - width / 2, n_models_plot
    )
    display_names = [r"$\Delta$AIC", r"$\Delta$AICc", r"$\Delta$BIC"]
    bar_colors = sns.color_palette("colorblind", n_models_plot)

    if should_break:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08},
            figsize=(fig_width, figure_height),
        )
        for i, model_name in enumerate(models_to_plot):
            vals = [metrics_dict[model_name][m] for m in metrics_names]
            display_name = display_name_for(model_name)
            ax_top.bar(
                x + offsets[i], vals, width, label=display_name, alpha=0.9,
                color=bar_colors[i],
            )
            ax_bot.bar(
                x + offsets[i], vals, width, label=display_name, alpha=0.9,
                color=bar_colors[i],
            )

        bot_lower_limit = min(0, min(normal_vals) * 1.05) if normal_vals else 0
        ax_bot.set_ylim(bot_lower_limit if y_min is None else y_min, bot_upper_limit)
        if bot_lower_limit < 0:
            ax_bot.axhline(0, color="black", linewidth=0.8, zorder=0.5)
        ax_top.set_ylim(top_lower_limit, top_upper_limit)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)
        ax_bot.xaxis.tick_bottom()
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(display_names)
        ax_top.set_title(
            f"Model Selection Metrics (Relative to Best Model)\nBest Model (BIC): {display_name_for(best_model)}",
            pad=14,
            fontweight="medium",
        )
        fig.supylabel(r"$\Delta$ Metric Value", fontsize="medium", x=0.02)
        ax_top.legend(loc="center left", bbox_to_anchor=(1.02, -0.1), fontsize="small")
        ax_top.grid(alpha=0.3, axis="y")
        ax_bot.grid(alpha=0.3, axis="y")

        d = 0.015
        kwargs = dict(
            transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2
        )
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    else:
        fig, ax = plt.subplots(figsize=(fig_width, figure_height))
        for i, model_name in enumerate(models_to_plot):
            vals = [metrics_dict[model_name][m] for m in metrics_names]
            display_name = display_name_for(model_name)
            ax.bar(
                x + offsets[i], vals, width, label=display_name, alpha=0.9,
                color=bar_colors[i],
            )
        bottom_limit = min(0, global_min * 1.05)
        ax.set_ylim(bottom_limit if y_min is None else y_min, global_max * 1.05)
        if bottom_limit < 0:
            ax.axhline(0, color="black", linewidth=0.8, zorder=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.set_title(
            f"Model Selection Metrics (Relative to Best Model)\nBest Model (BIC): {display_name_for(best_model)}",
            pad=14,
            fontweight="medium",
        )
        ax.set_ylabel(r"$\Delta$ Metric Value", fontsize="medium")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    safe_names = "_".join(models_to_plot)[:100]
    _save_figure(fig, f"compare_{safe_names}", path)
    plt.close(fig)

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    return {"metrics_table": metrics_df}, best_model


def plot_shortlong_vs_combined_latex(
    results_df_dict: dict,
    key_short: str,
    key_long: str,
    key_combined: str,
    ll_col: str = "after_lls_ga",
    export_table_path: str | None = None,
    font_size: int = 12,
    is_latex: bool = IS_LATEX,
    figure_height: float = 6,
    path: str = DEFAULT_FIGURE_PATH,
    y_min: float | None = None,
):
    """Compare separate short+long fits against a combined fit using AIC, AICc, and BIC.

    y_min : float, optional
        Lower bound of the metrics-bar-chart Y-axis (left panel). Defaults to
        matplotlib's automatic bar-anchored-at-0 behaviour; since AIC/AICc/BIC
        here are raw (not delta) values that sit far from 0, set this closer
        to the smallest bar to make the combined-vs-separate difference
        visible.
    """
    ll_short = _ensure_loglikelihoods(results_df_dict[key_short][ll_col].tolist())
    ll_long = _ensure_loglikelihoods(results_df_dict[key_long][ll_col].tolist())
    ll_comb = _ensure_loglikelihoods(results_df_dict[key_combined][ll_col].tolist())
    delta = (ll_short + ll_long) - ll_comb

    # Per-subject-summed convention (see compute_metrics_per_subject_summed):
    # each subject is an independent fit with their own n_obs, so the
    # "separate" row's metrics are the elementwise sum of the short fit's
    # own per-subject-summed metrics and the long fit's own (one BIC_i per
    # subject per horizon, summed over both horizons and all subjects).
    metrics_short = compute_metrics_per_subject_summed(results_df_dict[key_short], ll_col)
    metrics_long = compute_metrics_per_subject_summed(results_df_dict[key_long], ll_col)
    metrics_sep = {k: metrics_short[k] + metrics_long[k] for k in metrics_short}
    metrics_comb = compute_metrics_per_subject_summed(results_df_dict[key_combined], ll_col)
    metrics_table = pd.DataFrame(
        [metrics_comb, metrics_sep], index=[key_combined, f"{key_short}+{key_long}"]
    )
    # Best (lowest BIC) row first, matching the short/long/combined
    # candidate-model tables' convention.
    metrics_table = metrics_table.sort_values("BIC")

    if export_table_path:
        df_latex = metrics_table.reset_index().rename(columns={"index": "Model"})
        df_latex["Model"] = df_latex["Model"].apply(
            lambda x: f"${_texttt_with_condition_subscript(x)}$"
        )
        # k is the per-subject free-parameter count (summed across the
        # short+long fits for the separate-model row); n_obs is omitted
        # since it varies subject to subject and has no single meaningful
        # value to report for the row as a whole.
        cols_order = ["Model", "sum logL", "k", "AIC", "AICc", "BIC"]
        for col in cols_order:
            if col not in df_latex.columns:
                df_latex[col] = np.nan
        df_latex = df_latex[cols_order].rename(
            columns={"k": r"$N_{\text{params}}$"}
        )
        latex_table = df_latex.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,
            column_format="lccccc",
            na_rep="NaN",
        )
        with open(export_table_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table exported as {export_table_path}")

    _set_plot_style(font_size=font_size, is_latex=is_latex)
    fig, ax = plt.subplots(1, 1, figsize=(8, figure_height))

    metrics_names = ["AIC", "AICc", "BIC"]
    comb_vals = [metrics_comb[m] for m in metrics_names]
    sep_vals = [metrics_sep[m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(
        x - width / 2, comb_vals, width, label=_display_task_name(key_combined), color="tab:blue", alpha=0.9
    )
    ax.bar(
        x + width / 2,
        sep_vals,
        width,
        label=f"{_display_task_name(key_short)}+{_display_task_name(key_long)}",
        color="tab:orange",
        alpha=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.set_title("Model selection metrics", fontsize=14)
    ax.legend(loc="upper left")
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    plt.tight_layout()
    _save_figure(fig, f"{key_short}_{key_long}_vs_{key_combined}", path)
    plt.close(fig)

    return {"metrics_table": metrics_table, "delta": delta}


def plot_pomdp_vs_glm_bic_comparison(
    combined_row,
    separate_row,
    glm_metrics,
    combined_name: str,
    separate_name: str,
    glm_name: str = "GLM",
    export_table_path: str | None = None,
    fname: str = "pomdp_combined_separate_vs_glm_combined",
    path: str = DEFAULT_FIGURE_PATH,
    font_size: int = 12,
    is_latex: bool = IS_LATEX,
    figsize: tuple[float, float] = (9, 6),
    title: str = "Model selection metrics",
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
):
    """Three-way grouped AIC/AICc/BIC bar comparison -- POMDP combined fit
    vs. POMDP separate short+long fit vs. GLM (combined) -- plus the
    matching LaTeX metrics table.

    Shared by the full-action and commit-only (decide vs. wait) comparisons
    in notebooks/model_comparison.ipynb, which differ only in which
    log-likelihood feeds `combined_row`/`separate_row` (full 3-way
    log_likelihood vs. log_likelihood_commit) and in `title`/`fname`.

    combined_row, separate_row : Series with "sum logL", "k", "n_obs",
        "AIC", "AICc", "BIC" (as returned by `compute_metrics`).
    glm_metrics : dict/Series, same keys, for the GLM_combined fit.
    """
    bic_table = pd.DataFrame(
        [combined_row, separate_row, pd.Series(glm_metrics)],
        index=[combined_name, separate_name, glm_name],
    )
    print(bic_table.to_string())

    if export_table_path:
        df_latex = bic_table.reset_index().rename(columns={"index": "Model"})
        df_latex["Model"] = df_latex["Model"].apply(
            lambda x: f"${_texttt_with_condition_subscript(x)}$"
            if x != "GLM_combined" else r"$\texttt{GLM\_combined}$"
        )
        cols_order = ["Model", "sum logL", "k", "AIC", "AICc", "BIC"]
        df_latex = df_latex[cols_order].rename(
            columns={"k": r"$N_{\text{params}}$"}
        )
        df_latex.to_latex(
            export_table_path, index=False, float_format="%.2f", escape=False,
            column_format="lccccc", na_rep="NaN",
        )
        print(f"LaTeX table exported as {export_table_path}")

    _set_plot_style(font_size=font_size, is_latex=is_latex)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    metrics_names = ["AIC", "AICc", "BIC"]
    comb_vals = [combined_row[m] for m in metrics_names]
    sep_vals = [separate_row[m] for m in metrics_names]
    glm_vals = [glm_metrics[m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.25
    ax.bar(x - width, comb_vals, width, label=_display_task_name(combined_name), color="tab:blue", alpha=0.9)
    ax.bar(x, sep_vals, width, label=_display_task_name(separate_name), color="tab:orange", alpha=0.9)
    ax.bar(x + width, glm_vals, width, label=glm_name, color="tab:green", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=font_size)
    ax.set_title(title, fontsize=font_size + 2)
    ax.legend(loc=legend_loc )
    # Data-driven zoom (not a hardcoded value): GLM's AIC/BIC sit well below
    # POMDP's, so a fixed y_min would clip its bars off-screen entirely.
    all_vals = comb_vals + sep_vals + glm_vals
    ax.set_ylim(bottom=min(all_vals) * 0.95)

    plt.tight_layout()
    _save_figure(fig, fname, path)
    plt.close(fig)

    return bic_table, fig, ax


def plot_compare_two_models_latex(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    ll_col: str = "after_lls_ga",
    export_table_path: str | None = None,
    is_latex: bool = IS_LATEX,
    figure_height: float = 6,
    path: str = DEFAULT_FIGURE_PATH,
):
    """Compares two models, generates plots, and exports a LaTeX table."""
    ll1 = _ensure_loglikelihoods(df1[ll_col])
    ll2 = _ensure_loglikelihoods(df2[ll_col])
    n_params_1 = len(df1["fit_params_ga"].iloc[0])
    n_params_2 = len(df2["fit_params_ga"].iloc[0])
    metrics_1 = compute_metrics(ll1.sum(), n_params_1, estimate_n_obs_for_subjects(df1))
    metrics_2 = compute_metrics(ll2.sum(), n_params_2, estimate_n_obs_for_subjects(df2))

    metrics_dict = {model1_name: metrics_1, model2_name: metrics_2}
    print(f"--- Comparison: {model1_name} vs {model2_name} ---")
    print(f"Best model (AIC): {get_best_model(metrics_dict, 'AIC')}")
    print(f"Best model (BIC): {get_best_model(metrics_dict, 'BIC')}\n")
    best_model = get_best_model(metrics_dict, "BIC")

    if export_table_path:
        df_latex = pd.DataFrame([metrics_1, metrics_2])
        df_latex.insert(
            0,
            "Model",
            [
                f"${_texttt_with_condition_subscript(model1_name)}$",
                f"${_texttt_with_condition_subscript(model2_name)}$",
            ],
        )
        cols_order = ["Model", "sum logL", "k", "n_obs", "AIC", "AICc", "BIC"]
        df_latex = df_latex[cols_order].rename(
            columns={"n_obs": r"$n_{\text{obs}}$", "k": r"$N_{\text{params}}$"}
        )
        latex_table = df_latex.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,
            column_format="lcccccc",
            na_rep="NaN",
        )
        with open(export_table_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table exported to {export_table_path}")

    _set_plot_style(is_latex=is_latex)
    fig, axes = plt.subplots(1, 2, figsize=(16, figure_height))
    metrics_names = ["AIC", "AICc", "BIC"]
    m1_vals = [metrics_1[m] for m in metrics_names]
    m2_vals = [metrics_2[m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35

    axes[0].bar(
        x - width / 2, m1_vals, width, label=_display_task_name(model1_name), color="tab:blue", alpha=0.9
    )
    axes[0].bar(
        x + width / 2, m2_vals, width, label=_display_task_name(model2_name), color="tab:orange", alpha=0.9
    )
    axes[0].set_xticks(x, metrics_names)
    axes[0].legend(loc="lower left", fontsize="small")

    delta_ll = ll1 - ll2
    axes[1].hist(delta_ll, bins=20, color="gray", edgecolor="white")
    axes[1].set_xlabel(f"LL({model1_name}) - LL({model2_name}) (per subject)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-subject log-likelihood difference")

    plt.tight_layout()
    _save_figure(fig, f"{model1_name}_vs_{model2_name}", path)
    plt.close(fig)

    return delta_ll, best_model


def plot_compare_three_models_latex(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    model3_name: str = "Model 3",
    ll_col: str = "after_lls_ga",
    export_table_path: str | None = None,
    is_latex: bool = IS_LATEX,
    figure_height: float = 6,
    path: str = DEFAULT_FIGURE_PATH,
):
    """Compares three models, generates plots, and exports a LaTeX table."""
    ll1 = _ensure_loglikelihoods(df1[ll_col])
    ll2 = _ensure_loglikelihoods(df2[ll_col])
    ll3 = _ensure_loglikelihoods(df3[ll_col])
    metrics_1 = compute_metrics(
        ll1.sum(), len(df1["fit_params_ga"].iloc[0]), estimate_n_obs_for_subjects(df1)
    )
    metrics_2 = compute_metrics(
        ll2.sum(), len(df2["fit_params_ga"].iloc[0]), estimate_n_obs_for_subjects(df2)
    )
    metrics_3 = compute_metrics(
        ll3.sum(), len(df3["fit_params_ga"].iloc[0]), estimate_n_obs_for_subjects(df3)
    )

    metrics_dict = {
        model1_name: metrics_1,
        model2_name: metrics_2,
        model3_name: metrics_3,
    }
    print(f"--- Comparison: {model1_name} vs {model2_name} vs {model3_name} ---")
    print(f"Best model (AIC): {get_best_model(metrics_dict, 'AIC')}")
    print(f"Best model (BIC): {get_best_model(metrics_dict, 'BIC')}\n")
    best_model = get_best_model(metrics_dict, "BIC")

    def to_latex_tt(name: str) -> str:
        # Real, math-mode-wrapped LaTeX -- used for the exported .tex table,
        # not matplotlib. Renders a trailing "_F"/"_NF" condition tag as a
        # subscript instead of a literal underscore.
        return f"${_texttt_with_condition_subscript(name)}$"

    def display_name_for(name: str) -> str:
        # mathtext doesn't know \texttt; only valid under real usetex.
        return to_latex_tt(name) if is_latex else _display_task_name(name)

    m1_latex, m2_latex, m3_latex = (
        to_latex_tt(model1_name),
        to_latex_tt(model2_name),
        to_latex_tt(model3_name),
    )
    m1_disp, m2_disp, m3_disp = (
        display_name_for(model1_name),
        display_name_for(model2_name),
        display_name_for(model3_name),
    )

    if export_table_path:
        df_latex = pd.DataFrame([metrics_1, metrics_2, metrics_3])
        df_latex.insert(0, "Model", [m1_latex, m2_latex, m3_latex])
        cols_order = ["Model", "sum logL", "k", "n_obs", "AIC", "AICc", "BIC"]
        df_latex = df_latex[cols_order].rename(
            columns={"n_obs": r"$n_{\text{obs}}$", "k": r"$N_{\text{params}}$"}
        )
        latex_table = df_latex.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,
            column_format="lcccccc",
            na_rep="NaN",
        )
        with open(export_table_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table exported to {export_table_path}")

    fig, axes = plt.subplots(1, 4, figsize=(24, figure_height))
    metrics_names = ["AIC", "AICc", "BIC"]
    m1_vals = [metrics_1[m] for m in metrics_names]
    m2_vals = [metrics_2[m] for m in metrics_names]
    m3_vals = [metrics_3[m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.25
    axes[0].bar(x - width, m1_vals, width, label=m1_disp, color="tab:blue", alpha=0.9)
    axes[0].bar(x, m2_vals, width, label=m2_disp, color="tab:orange", alpha=0.9)
    axes[0].bar(x + width, m3_vals, width, label=m3_disp, color="tab:green", alpha=0.9)
    axes[0].set_xticks(x, metrics_names)
    axes[0].set_title("Model selection metrics")
    axes[0].legend(loc="lower left", fontsize="small")
    axes[1].hist(ll1 - ll2, bins=20, color="gray", edgecolor="white")
    axes[1].set_title(f"{m1_disp} vs {m2_disp}")
    axes[1].set_xlabel(f"LL({m1_disp}) - LL({m2_disp})")
    axes[2].hist(ll1 - ll3, bins=20, color="gray", edgecolor="white")
    axes[2].set_title(f"{m1_disp} vs {m3_disp}")
    axes[2].set_xlabel(f"LL({m1_disp}) - LL({m3_disp})")
    axes[3].hist(ll2 - ll3, bins=20, color="gray", edgecolor="white")
    axes[3].set_title(f"{m2_disp} vs {m3_disp}")
    axes[3].set_xlabel(f"LL({m2_disp}) - LL({m3_disp})")
    plt.tight_layout()

    _save_figure(fig, f"{model1_name}_{model2_name}_{model3_name}", path)
    plt.close(fig)

    return {
        "metrics_table": pd.DataFrame(
            [metrics_1, metrics_2, metrics_3],
            index=[model1_name, model2_name, model3_name],
        )
    }, best_model


def set_plot_style(font_size: int = 20, is_latex: bool = IS_LATEX) -> None:
    """Apply the project-wide matplotlib style.

    Serif with LaTeX when a LaTeX installation is available, sans-serif
    otherwise, with every text element at the same size.

    Args:
        font_size (int): Point size applied to titles, labels, ticks and legend.
        is_latex (bool): Render text through LaTeX. Defaults to whether a
            working latex binary was detected.
    """
    plt.rcParams.update(
        {
            "text.usetex": is_latex,
            "font.family": "serif" if is_latex else "sans-serif",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def save_figure(fig, base_path: str, dpi: int = 600, close: bool = True) -> None:
    """Save a figure as .pdf, .svg and .png under one base path.

    The public counterpart of _save_figure, for callers that already hold a
    full path rather than a filename plus directory.

    Args:
        fig: The matplotlib figure to write.
        base_path (str): Path without extension; three files are written.
        dpi (int): Raster resolution for the .png.
        close (bool): Close the figure afterwards. On by default because these
            are written in loops over models and subjects, where leaving them
            open exhausts matplotlib's figure limit.
    """
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    for ext, extra in ((".pdf", {}), (".svg", {}), (".png", {"dpi": dpi})):
        fig.savefig(f"{base_path}{ext}", bbox_inches="tight", pad_inches=0.03, **extra)
    if close:
        plt.close(fig)


def plot_multi_subjects_panel(
    subject_ids,
    ensemble_data_short,
    ensemble_data_long,
    metrics_df_short,
    metrics_df_long,
    short_bins=8,
    long_bins=14,
    show_simulation=True,
    show_outcomes=False,
    panel_labels=None,
    path=None,
    filename="four_subjects_panel",
):
    """
    Plot multiple subjects in a single figure, each subject in one panel.

    Subjects are arranged in a 2×2 grid (left-to-right, top-to-bottom).
    Short and long horizon distributions are overlaid in each panel.
    One shared legend is placed below all panels.

    Parameters
    ----------
    subject_ids   : list of subject IDs, length must be 4 (for 2×2) or 2 (for 1×2)
    show_simulation : bool  — overlay ensemble sim mean ± std
    show_outcomes   : bool  — add outcomes sub-panel per subject (makes layout wider)
    panel_labels  : list of str or None — e.g. ['A','B','C','D']
    path          : directory to save figure; None = show only
    filename      : base filename (without extension)
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ── colours ──────────────────────────────────────────────────────────────
    C_SHORT     = '#eb6834'
    C_LONG      = '#4a3aa7'
    C_SHORT_SIM = '#a8431a'
    C_LONG_SIM  = '#2f2470'
    BAR_ALPHA   = 0.55

    if panel_labels is None:
        panel_labels = [chr(97 + i) for i in range(len(subject_ids))]  # a,b,c,d...

    n = len(subject_ids)
    ncols_grid = 2
    nrows_grid = (n + 1) // 2
    fig_w = 16 * (2 if show_outcomes else 1)
    fig_h = 4.5 * nrows_grid + 1.2   # extra room for legend

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    def _pad(arr, n_bins):
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= n_bins:
            return arr[:n_bins]
        out = np.zeros(n_bins)
        out[:len(arr)] = arr
        return out

    x = np.arange(long_bins)
    valid_indices  = [0, 1, 4]
    outcome_labels_tick = [-2, -1, 2]
    x_out = np.arange(len(outcome_labels_tick))
    bar_w = 0.35

    for panel_idx, (sid, label) in enumerate(zip(subject_ids, panel_labels)):
        row = panel_idx // ncols_grid
        col = panel_idx %  ncols_grid
        ax  = axes[row][col]

        data_s = ensemble_data_short.get(sid)
        data_l = ensemble_data_long.get(sid)

        if data_s is None or data_l is None:
            ax.set_title(f"Subject {sid} — data missing", fontsize=12)
            ax.text(-0.09, 1.14, f"({label})", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", va="top", ha="right")
            ax.axis('off')
            continue

        hc_s = _pad(data_s['human_counts'],   long_bins)
        as_s = _pad(data_s['avg_sim_counts'],  long_bins)
        ss_s = _pad(data_s['std_sim_counts'],  long_bins)

        hc_l = _pad(data_l['human_counts'],   long_bins)
        as_l = _pad(data_l['avg_sim_counts'],  long_bins)
        ss_l = _pad(data_l['std_sim_counts'],  long_bins)

        ax.bar(x - bar_w/2, hc_s, width=bar_w, color=C_SHORT, alpha=BAR_ALPHA)
        ax.bar(x + bar_w/2, hc_l, width=bar_w, color=C_LONG,  alpha=BAR_ALPHA)

        if show_simulation:
            ax.errorbar(x - bar_w/2, as_s, yerr=ss_s, fmt='-o',
                        color=C_SHORT_SIM, capsize=3, linewidth=1.5)
            ax.errorbar(x + bar_w/2, as_l, yerr=ss_l, fmt='-s',
                        color=C_LONG_SIM,  capsize=3, linewidth=1.5)

        # Hazard onset markers
        ax.axvline(x=9,  color=C_LONG,  linestyle='--', alpha=0.7, linewidth=1.2)
        ax.axvline(x=3,  color=C_SHORT, linestyle='--', alpha=0.7, linewidth=1.2)

        ax.set_title(f"Subject {sid}", fontsize=13, fontweight='bold')
        # panel letter outside the axes, matching the convention used by the
        # other multi-panel figures rather than sitting inside the title
        ax.text(-0.09, 1.14, f"({label})", transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top", ha="right")
        ax.set_xlabel("Number of Draws", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)

    # Hide any unused axes (if n < nrows*ncols)
    for panel_idx in range(n, nrows_grid * ncols_grid):
        row = panel_idx // ncols_grid
        col = panel_idx %  ncols_grid
        axes[row][col].set_visible(False)

    # ── shared legend ─────────────────────────────────────────────────────────
    if show_simulation:
        handles = [
            Patch(facecolor=C_SHORT, alpha=BAR_ALPHA, label='Human — Short Horizon'),
            Patch(facecolor=C_LONG,  alpha=BAR_ALPHA, label='Human — Long Horizon'),
            Line2D([0],[0], color=C_SHORT_SIM, marker='o', label='Sim Short \u00b1 Std'),
            Line2D([0],[0], color=C_LONG_SIM,  marker='s', label='Sim Long \u00b1 Std'),
        ]
        ncol_leg = 4
    else:
        handles = [
            Patch(facecolor=C_SHORT, alpha=BAR_ALPHA, label='Short Horizon'),
            Patch(facecolor=C_LONG,  alpha=BAR_ALPHA, label='Long Horizon'),
            Line2D([0],[0], color=C_SHORT, linestyle='--', alpha=0.7, label='Short Hazard Onset'),
            Line2D([0],[0], color=C_LONG,  linestyle='--', alpha=0.7, label='Long Hazard Onset'),
        ]
        ncol_leg = 4

    fig.legend(handles=handles, loc='lower center', ncol=ncol_leg,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # ── save ──────────────────────────────────────────────────────────────────
    if path is not None:
        os.makedirs(path, exist_ok=True)
        base = os.path.join(path, filename)
        fig.savefig(base + ".pdf",          bbox_inches='tight')
        fig.savefig(base + ".svg",          bbox_inches='tight')
        fig.savefig(base + ".png", dpi=600, bbox_inches='tight')

    plt.show()


def plot_hazard_functions(
    short_window: tuple[int, int] = (4, 8),
    long_window: tuple[int, int] = (10, 14),
    max_draws: int = 15,
    font_size: int = 16,
    is_latex: bool = IS_LATEX,
    path: str = DEFAULT_FIGURE_PATH,
    filename: str = "hazard_functions_stacked",
):
    """Termination pmf and hazard rate for both horizon conditions.

    The task ends at a draw sampled uniformly from a window, so the pmf is flat
    across that window while the hazard rate h(t), the chance of ending now
    given it has not ended yet, rises as the window is used up. That rise is
    what the agent responds to, so the two panels show them together.

    Args:
        short_window (tuple[int, int]): First and last possible ending draw in
            the short horizon condition.
        long_window (tuple[int, int]): The same for the long horizon condition.
        max_draws (int): Extent of the shared x axis.
        font_size (int): Point size for all text.
        is_latex (bool): Render text through LaTeX.
        path (str): Directory to write into; created if absent.
        filename (str): Base name, written as .pdf, .svg and .png.

    Returns:
        matplotlib.figure.Figure: The figure, already saved.
    """
    set_plot_style(font_size=font_size, is_latex=is_latex)
    colour_short, colour_long, bar_alpha = "#2166AC", "#D6604D", 0.45

    def hazard(start, end):
        pmf = np.zeros(end + 1)
        pmf[start : end + 1] = 1.0 / (end - start + 1)
        shifted = np.zeros_like(pmf)
        shifted[1:] = pmf[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(np.cumsum(shifted) < 1,
                         pmf / (1 - np.cumsum(shifted)), 0.0)
        return np.round(h, 4), pmf

    def pad(arr):
        out = np.zeros(max_draws)
        out[: len(arr)] = arr
        return out

    h_short, pmf_short = hazard(*short_window)
    h_long, pmf_long = hazard(*long_window)
    h_short, pmf_short = pad(h_short), pad(pmf_short)
    h_long, pmf_long = pad(h_long), pad(pmf_long)

    x = np.arange(max_draws)
    width = 0.4
    y_max = max(h_short.max(), h_long.max(),
                pmf_short.max(), pmf_long.max()) + 0.08

    fig, (ax_short, ax_long) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, pmf, h, colour, marker, style, window in (
        (ax_short, pmf_short, h_short, colour_short, "o", "-", short_window),
        (ax_long, pmf_long, h_long, colour_long, "s", "--", long_window),
    ):
        offset = -width / 2 if ax is ax_short else width / 2
        ax.bar(x + offset, pmf, width=width, alpha=bar_alpha, color=colour,
               edgecolor="k", linewidth=0.5, label="PMF")
        upto = window[1] + 1 if ax is ax_short else max_draws
        ax.plot(x[:upto], h[:upto], marker=marker, linestyle=style,
                color=colour, linewidth=2, markersize=6, label=r"$h(t)$")
        ax.set_ylabel(r"Probability / $h(t)$")
        ax.set_ylim(0, y_max)
        ax.set_xlim(-0.5, max_draws - 0.5)
        label = "Short" if ax is ax_short else "Long"
        ax.set_title(f"{label} horizon (draws {window[0]} to {window[1]})",
                     fontsize=font_size)
        ax.legend(loc="upper left", frameon=True, ncol=2)
        ax.grid(alpha=0.3)

    ax_long.set_xlabel(r"Number of draws $t$")
    ax_long.set_xticks(x)
    fig.tight_layout()
    save_figure(fig, os.path.join(path, filename), close=False)
    return fig
