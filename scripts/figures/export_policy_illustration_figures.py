"""Policy heatmaps illustrating each manipulation, for the model figure.

Draws one panel per manipulation and gives each a title naming the manipulation
it shows. Without a title the panels are indistinguishable once tiled into a
multi-panel figure.

Usage (from scripts/):
    python3 export_policy_illustration_figures.py
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(R)

from src.pomdp import POMDPFactory
from src.utils.plotting import plot_hazard_functions

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

OUT = os.path.join(R, "figures", "POMDP", "illustrate")

BASE = dict(tau=1e-8, xi=0.0, subjective_cost=0, is_hazardous=False,
            horizon_condition="short", verbose=False, max_cards_per_draw=5)

# Panels are drawn at the width they are printed at, so LaTeX includes them at
# scale 1 and FONT_PT is the size that actually appears on the page. Drawn at
# the old default of 14in and shown in a 0.48\textwidth minipage, a panel was
# scaled by 0.18 and its 20pt text arrived on the page at 3.6pt.
TEXTWIDTH_IN = 5.25       # \textwidth in the PLOS template
FONT_PT = 8               # printed point size for title, labels, ticks, colourbar
ASPECT = 0.62             # panel height as a fraction of its width
YTICK_STEP = 10           # a 2.5in panel cannot carry a tick every 5 cards at 8pt

# The hazard is what the task actually does, so the hazard-on normative policy
# is the reference the other panels are read against; the no-hazard policy is
# the deterministic-deadline comparison, not the baseline.
# label -> (family, parameter overrides, title, width as a fraction of \textwidth)
PANELS = [
    ("hazard",  "vanilla", dict(is_hazardous=True),   "Normative optimum", 0.48),
    ("temp0.3", "vanilla", dict(is_hazardous=True, tau=0.3),
     r"Softmax temperature $\tau = 0.3$", 0.48),
    ("xi0.05",  "vanilla", dict(is_hazardous=True, xi=0.05),
     r"Lapse rate $\xi = 0.05$", 0.48),
    ("vanilla", "vanilla", {},                        "Deterministic deadline", 0.48),
    ("risk",    "vanilla", dict(xi=0.05, subjective_cost=-10),
     r"Subjective cost $R_{\mathrm{risk}} = -10$", 0.7),
    ("patience", "urgency", dict(xi=0.05, patience=5, urgency_coefficient=-0.6,
                                 c_max=0.5, urgency_slope=-0.3),
     r"Temporal regulation, $t_p = 5$", 0.7),
    ("patience_hazard", "urgency", dict(patience=5, urgency_coefficient=-0.6,
                                        c_max=1, urgency_slope=-0.8, is_hazardous=True),
     r"Temporal regulation with hazard", 0.7),
]


# The four policy panels are tiled into one figure rather than four files.
# Four separate panels each pay for their own colourbar, y-label and tick
# column, which at 2.5in wide left the heatmap less than half the panel; shared
# here, the same page area gives each heatmap roughly twice the width.
GRID = ["hazard", "temp0.3", "xi0.05", "vanilla"]
GRID_W = TEXTWIDTH_IN          # the composite spans the full text width
GRID_ASPECT = 0.667


def build(family, over):
    params = dict(BASE)
    params.update(over)
    pomdp = POMDPFactory(family)
    pomdp.__init__(**params)
    pomdp.value_iteration()
    return pomdp


def export_grid(specs):
    """The 2x2 policy figure, one shared colourbar, drawn at printed size."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(GRID_W, GRID_W * GRID_ASPECT),
                             sharex=True, sharey=True)
    mappable = None
    for k, (ax, key) in enumerate(zip(axes.ravel(), GRID)):
        family, over, title = specs[key]
        _, mappable = build(family, over).plot_best_actions(
            ax=ax, colorbar=False, title=title, font_size=FONT_PT,
            ytick_step=YTICK_STEP, xlabel="", ylabel="")
        ax.text(-0.11, 1.18, "abcd"[k], transform=ax.transAxes,
                fontsize=FONT_PT + 1, fontweight="bold", va="top", ha="right")
    for ax in axes[1, :]:
        ax.set_xlabel("Draws", fontsize=FONT_PT)
    for ax in axes[:, 0]:
        ax.set_ylabel("$n_y - n_b$", fontsize=FONT_PT)
    # A colourbar for three discrete actions costs a whole column of width to
    # its tick labels. A row of patches along the top says the same thing and
    # leaves the panels the full text width.
    from matplotlib.patches import Patch
    # a colourbar on the right rather than a row of patches underneath: the
    # bottom strip cost vertical space on the page, the right-hand margin is
    # already there
    fig.tight_layout(rect=(0, 0, 0.885, 1), pad=0.15, h_pad=0.6)
    cax = fig.add_axes([0.895, 0.34, 0.014, 0.32])
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["Blue", "Wait", "Yellow"])
    cbar.ax.tick_params(labelsize=FONT_PT)
    base = os.path.join(OUT, "best_actions_policy_grid")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", **({"dpi": 600} if ext == "png" else {}))
    plt.close(fig)
    print(f"  best_actions_policy_grid  {GRID_W:.2f} x {GRID_W*GRID_ASPECT:.2f} in "
          f"({' '.join(GRID)})")


# Figs 5 and 7: the normative optimum in colour (panel A of the policy grid)
# with the cells hatched where one added mechanism changes the chosen action.
# Both arms keep the generative hazard active and differ only by that
# mechanism, so the hatching isolates what the mechanism does.
# (label, (family, overrides) baseline, (family, overrides) manipulated, title)
DIFFS = [
    ("risk_diff",
     ("vanilla", {}),
     ("vanilla", dict(subjective_cost=-10)),
     "Baseline (colour): normative optimum\n"
     "Hatched: action differs when $R_{\\mathrm{risk}}=-10$"),
    ("patience_diff",
     ("vanilla", {}),
     ("urgency", dict(patience=6, urgency_coefficient=-0.6,
                      c_max=1.0, urgency_slope=-0.8)),
     "Baseline (colour): normative optimum\n"
     "Hatched: action differs with temporal regulation ($t_p=6$)"),
]
DIFF_FRAC = 0.7           # these are included at 0.7\linewidth
DIFF_ASPECT = 0.62


def _policy_array(family, over, ax=None):
    """The masked policy array, drawn into `ax` when one is supplied.

    Routed through plot_best_actions rather than reimplemented so the gap
    filling and masking cannot drift from the panels above.
    """
    import matplotlib.pyplot as plt
    pomdp = build(family, dict(over, is_hazardous=True))
    own = ax is None
    if own:
        tmp, ax = plt.subplots()
    arr, mappable = pomdp.plot_best_actions(
        ax=ax, colorbar=False, font_size=FONT_PT, ytick_step=YTICK_STEP,
        xlabel="Draws", ylabel="$n_y - n_b$")
    if own:
        plt.close(tmp)
    return arr, mappable


def export_diff(label, base_spec, alt_spec, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    w = DIFF_FRAC * TEXTWIDTH_IN
    fig, ax = plt.subplots(figsize=(w, w * DIFF_ASPECT))
    ref, mappable = _policy_array(*base_spec, ax=ax)
    alt, _ = _policy_array(*alt_spec)

    changed = (~ref.mask) & (~alt.mask) & (ref.data != alt.data)
    for d, j in zip(*np.where(changed)):
        # linewidth=0 so only the hatch strokes render; a border on every cell
        # turns a run of adjacent cells into a grid over the diagonal hatch
        ax.add_patch(Rectangle((d - .5, j - .5), 1, 1, linewidth=0,
                               edgecolor="black", facecolor="none",
                               hatch="////", zorder=3))
    ax.set_title(title, fontsize=FONT_PT)
    # colourbar on the right, as in the policy grid; the hatch cannot live on a
    # colour scale so it keeps a patch underneath
    fig.tight_layout(rect=(0, 0, 0.80, 1), pad=0.15)
    cax = fig.add_axes([0.815, 0.42, 0.022, 0.34])
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["Blue", "Wait", "Yellow"])
    cbar.ax.tick_params(labelsize=FONT_PT)
    fig.legend(handles=[Patch(facecolor="none", edgecolor="black",
                              hatch="////", label="Action\ndiffers")],
               loc="lower left", bbox_to_anchor=(0.795, 0.13), frameon=False,
               fontsize=FONT_PT, handlelength=1.1, handleheight=1.0)
    base = os.path.join(OUT, f"best_actions_heatmap_{label}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", **({"dpi": 600} if ext == "png" else {}))
    plt.close(fig)
    print(f"  best_actions_heatmap_{label:16} {w:.2f} x {w*DIFF_ASPECT:.2f} in  "
          f"({int(changed.sum())} cells differ)")


# --- alternative styling for the policy grid, following Fig 2 -----------------
# Fig 2 draws the discrete state lattice: one cell per (draw, n_y - n_b), grey
# wherever the task cannot produce the state, so the parity stripes and the
# reachable wedge are visible. export_grid above instead fills the gaps between
# reachable cells, which reads as solid blocks. Black has no counterpart here:
# in human data it marked reachable-but-never-visited states, whereas a policy
# assigns an action to every reachable state.
GRID_CELL_ASPECT = 0.667


def _raw_policy_lattice(pomdp):
    """(draws x diff) actions with unreachable states left masked, no gap filling."""
    num_draws, num_yellow, num_blue = pomdp.best_actions.shape
    max_diff = num_yellow - 1
    diff_range = np.arange(-max_diff, max_diff + 1)
    adj = np.full((num_draws, len(diff_range)), np.nan)
    for draw in range(num_draws):
        for yellow in range(num_yellow):
            blue = draw * pomdp.max_cards_per_draw - yellow
            if 0 <= blue < num_blue:
                d = yellow - blue
                if -max_diff <= d <= max_diff:
                    adj[draw, d + max_diff] = pomdp.best_actions[draw, yellow, blue]
    adj = adj[1:, :]                      # draw 0 has no cards yet
    return np.ma.array(adj, mask=np.isnan(adj) | (adj == 5)), diff_range


def export_grid_cells(specs):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    cmap = ListedColormap(["yellow", "blue", "green"])
    cmap.set_bad("0.72")                  # unreachable, as the grey in Fig 2
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(GRID_W, GRID_W * GRID_CELL_ASPECT),
                             sharex=True, sharey=True)
    for k, (ax, key) in enumerate(zip(axes.ravel(), GRID)):
        family, over, title = specs[key]
        arr, diff_range = _raw_policy_lattice(build(family, dict(over)))
        nd, ndiff = arr.shape
        mappable = ax.imshow(arr.T, cmap=cmap, norm=norm, aspect="auto",
                             interpolation="nearest", origin="lower")
        ax.set_title(title, fontsize=FONT_PT)
        ax.text(-0.11, 1.18, "abcd"[k], transform=ax.transAxes,
                fontsize=FONT_PT + 1, fontweight="bold", va="top", ha="right")
        ax.set_xticks(np.arange(nd)); ax.set_xticklabels(np.arange(1, nd + 1))
        yi = [i for i, v in enumerate(diff_range) if v % YTICK_STEP == 0]
        ax.set_yticks(yi); ax.set_yticklabels(diff_range[yi])
        ax.tick_params(labelsize=FONT_PT)
        # cell edges, drawn on minor ticks that sit on the pixel boundaries
        ax.set_xticks(np.arange(-.5, nd, 1), minor=True)
        ax.set_yticks(np.arange(-.5, ndiff, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.15)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_axisbelow(False)
    for ax in axes[1, :]:
        ax.set_xlabel("Draws", fontsize=FONT_PT)
    for ax in axes[:, 0]:
        ax.set_ylabel("$n_y - n_b$", fontsize=FONT_PT)
    # colourbar on the right; the grey has no place on a colour scale, so it
    # keeps a single patch underneath the bar
    fig.tight_layout(rect=(0, 0, 0.855, 1), pad=0.15, h_pad=0.6)
    cax = fig.add_axes([0.865, 0.40, 0.014, 0.30])
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["Blue", "Wait", "Yellow"])
    cbar.ax.tick_params(labelsize=FONT_PT)
    fig.legend(handles=[Patch(facecolor="0.72", edgecolor="0.2", linewidth=.5,
                              label="Unreachable")],
               loc="lower left", bbox_to_anchor=(0.845, 0.20), frameon=False,
               fontsize=FONT_PT, handlelength=1.1, handleheight=1.0)
    base = os.path.join(OUT, "best_actions_policy_grid_cells")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", **({"dpi": 600} if ext == "png" else {}))
    plt.close(fig)
    print(f"  best_actions_policy_grid_cells  {GRID_W:.2f} x "
          f"{GRID_W*GRID_CELL_ASPECT:.2f} in (Fig 2 styling)")


# The temporal regulation function itself (Fig 6). Its phi_max must match the
# value illustrated in the policy figure, or the sigmoid shows a different
# mechanism from the one whose effect on the policy is being drawn.
REG = dict(phi_min=-0.6, phi_max=1.0, t_p=6)
REG_KS = [(-0.3, "tab:purple", "shallow"),
          (-0.8, "black", "moderate"),
          (-2.5, "tab:orange", "steep")]


def export_regulation_sigmoid():
    """The temporal regulation function, exactly as it was drawn in
    notebooks/test_place.ipynb -- same figure size, font size, legend
    placement and annotation positions. Only phi_max and t_p change, so that
    the function drawn here is the one whose effect on the policy Fig 7 shows.
    """
    import matplotlib.pyplot as plt

    # The notebook drew this 10in wide with 20pt text and it was included at
    # 0.7\linewidth, so it printed at about 8pt. Included at the full text
    # width the same 20pt would print at 11pt, i.e. body-text size, so the
    # point size is set from the scale the figure is actually reduced by.
    font_size = round(FONT_PT * 9.55 / TEXTWIDTH_IN, 1)   # ~14.6 -> ~8pt printed
    plt.rcParams.update({
        # the notebook used text.usetex=True; there is no latex binary in the
        # container, so Computer Modern mathtext stands in for it and gives the
        # same serif look without shelling out to TeX
        # sans-serif to match every other figure in the paper, and because
        # PLOS asks for Arial/Helvetica in figure text rather than a serif face
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    })

    def temporal_regulation_function(k, t_p, phi_min, phi_max, num_draws):
        return phi_min + (phi_max - phi_min) / (1 + np.exp(-k * (num_draws - t_p)))

    phi_min = REG["phi_min"]
    phi_max = REG["phi_max"]
    t_p = REG["t_p"]

    num_draws = np.linspace(1, 8, 200)
    k_values = [-0.3, -0.8, -2.5]
    k_colors = ["tab:purple", "black", "tab:orange"]
    descriptions = ["shallow", "moderate", "steep"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.axhline(y=phi_min, color="tab:blue", linestyle="-.", linewidth=2, alpha=0.6)
    ax.axhline(y=phi_max, color="tab:green", linestyle="-.", linewidth=2, alpha=0.6)

    for k, color, desc in zip(k_values, k_colors, descriptions):
        phi = temporal_regulation_function(k, t_p, phi_min, phi_max, num_draws)
        ax.plot(num_draws, phi, color=color, linewidth=2, label=fr"$k={k}$ ({desc})")

    ax.axvline(x=t_p, color="tab:red", linestyle="--", linewidth=2)
    ax.text(t_p + 0.1, phi_max * 0.6, fr"$t_p={t_p}$", color="tab:red",
            fontsize=font_size, verticalalignment="center")

    ax.text(num_draws[-1] - 2, phi_min - 0.25, fr"$\phi_{{\min}}={phi_min}$",
            color="tab:blue", va="bottom", fontsize=font_size)
    ax.text(num_draws[-1] - 2, phi_max + 0.1, fr"$\phi_{{\max}}={phi_max}$",
            color="tab:green", va="bottom", fontsize=font_size)

    ax.set_xlabel(r"Number of Draws $(t)$")
    ax.set_ylabel(r"$\Phi(t)$")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    # the notebook's (-1, 1) was sized for phi_max = 0.5; at 1.0 the asymptote
    # sits on the frame and its label falls outside the axes
    ax.set_ylim(phi_min - 0.5, phi_max + 0.45)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=font_size * 0.9, frameon=True, facecolor="white")

    plt.tight_layout()
    base = os.path.join(OUT, "temporal_regulation_function")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight",
                    **({"dpi": 600} if ext == "png" else {}))
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    print(f"  temporal_regulation_function      10.0 x 4.0 in, font {font_size} "
          f"(phi in [{phi_min}, {phi_max}], t_p={t_p})")



def export_hazard_functions(outdir):
    """Termination pmf and hazard rate for both horizon conditions.

    Data free, like the rest of this script: the windows are a property of the
    task, not of any fit.
    """
    plot_hazard_functions(path=outdir, filename="hazard_functions_stacked")
    print(f"wrote {outdir}/hazard_functions_stacked.[pdf|png|svg]")


def main():
    os.makedirs(OUT, exist_ok=True)
    for label, family, over, title, frac in PANELS:
        params = dict(BASE)
        params.update(over)
        pomdp = POMDPFactory(family)
        pomdp.__init__(**params)
        pomdp.value_iteration()
        w = frac * TEXTWIDTH_IN
        pomdp.plot_best_actions(path=OUT, label=label, title=title,
                                figsize=(w, w * ASPECT), font_size=FONT_PT,
                                ytick_step=YTICK_STEP, exact_size=True,
                                xlabel="Draws", ylabel="$n_y - n_b$")
        name = f"best_actions_heatmap{'_' + label if label else ''}"
        print(f"  {name:38} {w:.2f}in  {title}")
    export_regulation_sigmoid()
    for lab, b, a, t in DIFFS:
        export_diff(lab, b, a, t)
    specs = {lab: (fam, ov, ti) for lab, fam, ov, ti, _ in PANELS}
    export_grid(specs)            # solid-block styling (currently in the paper)
    export_grid_cells(specs)     # Fig 2 styling, for comparison
    export_hazard_functions(OUT)
    print(f"\n  wrote {len(PANELS)} panels to {OUT}")


if __name__ == "__main__":
    main()
