"""Model recovery confusion matrices, one per horizon.

Each cell of data/POMDP/_model_recovery/gen_<G>__fit_<F>.pkl holds the fit of
candidate F to data simulated from generator G, for all 105 subjects. Recovery
is then just model selection applied to those cells: for generator G, does BIC
pick F = G?

Two views are produced, because they answer different questions:

  * subject level -- the fraction of the 105 subjects whose own BIC is lowest
    for each candidate. This is the confusion matrix proper: it says how often
    selection lands on the right model, and where it goes when it does not.
  * group level -- the argmin of the per-subject-summed BIC, the same statistic
    the paper's model comparison uses. One winner per generator, plus the BIC
    margin over the generator itself, which says how badly a miss misses.

BIC is per-subject-summed throughout (k_i*log(n_obs_i) with n_obs_i the
subject's own draw count), matching compute_metrics_per_subject_summed and the
rest of the model comparison.

Usage (from scripts/):
    python3 export_model_recovery.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plotting import (
    compute_metrics_per_subject_summed as metrics_of,
    compute_metrics,
    _per_subject_n_obs,
)

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# MODEL_RECOVERY_CELL_DIR points the export at a different set of cells, e.g.
# the resampled-parameter pass under data/POMDP_recovery_x4/.
CELL_DIR = os.environ.get(
    "MODEL_RECOVERY_CELL_DIR", os.path.join(R, "data", "POMDP", "_model_recovery")
)
FIG_OUT = os.path.join(R, "BIC", "figures")
TAB_OUT = os.path.join(R, "BIC", "tables")

# Drawn at its final printed size so LaTeX includes it unscaled; see
# export_parameter_recovery_panel.py for why that matters for legibility.
WIDTH_IN = 6.5
FONT_PT = 8
TICK_PT = 6
CELL_PT = 5.5

# Outline colour for the population-summed BIC winner. Pure red disappears
# against the dark end of viridis; this reads on both ends of the map.
OUTLINE = "#FF1F5B"

HORIZON_NAME = {"S": "short horizon", "L": "long horizon", "C": "combined horizons"}
HORIZON_ORDER = ["S", "L", "C"]
HORIZON_WORD = {"S": "short", "L": "long", "C": "combined"}

# Above this many models the three-panel overview and the per-cell numbers
# stop being legible at print size, and the per-horizon figures switch to
# indexed axes.
PANEL_MAX = 12

# MODEL_RECOVERY_LISTS names the intended model set per horizon, as a path
# template containing {h} ("short"/"long"/"combined"), e.g.
# BIC/_model_recovery_full_{h}.txt. Setting it fixes the grid to that set, so
# cells not yet computed show as gaps rather than silently shrinking the
# matrix. Unset, the grid is whatever cells exist on disk.
LIST_TMPL = os.environ.get("MODEL_RECOVERY_LISTS")

# MODEL_RECOVERY_TAG suffixes every output name, so a restricted grid (say the
# five best models per horizon) can be rendered from the same cells without
# overwriting the full grid's figures and tables.
TAG = os.environ.get("MODEL_RECOVERY_TAG", "")
SUF = f"_{TAG}" if TAG else ""


def per_subject_bic(df):
    """Each subject's own BIC under this cell's fit, in the cell's row order."""
    n_params = len(df["fit_params_ga"].iloc[0])
    raw_ll = np.asarray(df["after_lls_ga"].tolist(), dtype=float)
    sign = -1.0 if np.nanmean(raw_ll) > 0 else 1.0
    out = []
    for ll_raw, subject_data in zip(raw_ll, df["data_dict_of_lists"]):
        if np.isnan(ll_raw):
            out.append(np.nan)
            continue
        n_obs_i = _per_subject_n_obs(subject_data)
        out.append(compute_metrics(sign * ll_raw, n_params, n_obs_i)["BIC"])
    return pd.Series(out, index=df["subject_ID"].values, dtype=float)


def load_cells():
    group, subject = {}, {}
    for fname in sorted(os.listdir(CELL_DIR)):
        if not fname.startswith("gen_") or not fname.endswith(".pkl"):
            continue
        gen, fit = fname[len("gen_"):-len(".pkl")].split("__fit_")
        df = pd.read_pickle(os.path.join(CELL_DIR, fname))
        m = metrics_of(df, "after_lls_ga")
        group[(gen, fit)] = m
        subject[(gen, fit)] = per_subject_bic(df)
    return group, subject


def intended_models(horizon):
    """The model set the grid should span, if one was named."""
    if not LIST_TMPL:
        return None
    path = os.path.join(R, LIST_TMPL.format(h=HORIZON_WORD[horizon]))
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def matrices(group, subject, horizon):
    """Square generator x candidate matrices for one horizon prefix."""
    keys = [k for k in group if k[0][0] == horizon]
    listed = intended_models(horizon)
    if listed:
        models = gens = listed
    else:
        models = sorted({k[0] for k in keys} | {k[1] for k in keys})
        gens = sorted({k[0] for k in keys})
    keys = [k for k in keys if k[0] in set(gens) and k[1] in set(models)]

    bic = pd.DataFrame(np.nan, index=gens, columns=models)
    for k in keys:
        bic.loc[k[0], k[1]] = group[k]["BIC"]

    win = pd.DataFrame(np.nan, index=gens, columns=models)
    for g in gens:
        cols = [f for f in models if (g, f) in subject]
        if not cols:
            continue
        per_sub = pd.DataFrame({f: subject[(g, f)] for f in cols})
        picked = per_sub.idxmin(axis=1)
        counts = picked.value_counts()
        n = len(picked.dropna())
        for f in models:
            win.loc[g, f] = 100.0 * counts.get(f, 0) / n
    return bic, win


def render_single(h, bic, win):
    """One horizon, one figure, sized to the grid.

    Past about a dozen models the 13-character TASK strings stop fitting on the
    axes at a legible size, so the ticks become indices and the mapping moves
    to a companion table. Per-cell numbers go the same way: at 39x39 they would
    be 3 pt, and the colour already carries the value.
    """
    n = len(win.columns)
    big = n > PANEL_MAX
    side = min(WIDTH_IN, 1.6 + 0.135 * n) if big else WIDTH_IN * 0.62
    fig, ax = plt.subplots(figsize=(side + 0.9, side + 0.55))

    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="0.85")
    im = ax.imshow(win.values, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    labels = [str(i + 1) for i in range(n)] if big else list(win.columns)
    tick_pt = 4.5 if big else TICK_PT
    ax.set_xticks(range(n))
    ax.set_yticks(range(len(win.index)))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_pt, family="monospace")
    ax.set_yticklabels(labels, fontsize=tick_pt, family="monospace")
    ax.set_xlabel("fitted model" + (" (index)" if big else ""), fontsize=FONT_PT)
    ax.set_ylabel("generating model" + (" (index)" if big else ""), fontsize=FONT_PT)
    ax.set_title(f"{HORIZON_NAME[h]} ({n} models)", fontsize=FONT_PT, pad=4)

    for i, g in enumerate(win.index):
        if not big:
            for j, f in enumerate(win.columns):
                v = win.loc[g, f]
                if v >= 0.5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=CELL_PT, color="white" if v < 55 else "black")
        if bic.loc[g].notna().any():
            j = list(win.columns).index(bic.loc[g].idxmin())
            # Inset from the cell edge and drawn above the white minor grid,
            # which otherwise sits exactly on the outline and erases half of it.
            pad = 0.06
            ax.add_patch(plt.Rectangle(
                (j - 0.5 + pad, i - 0.5 + pad), 1 - 2 * pad, 1 - 2 * pad,
                fill=False, edgecolor=OUTLINE, linewidth=0.8 if big else 1.8,
                zorder=5))
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(win.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.15 if big else 0.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=1.5, pad=1)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_label("subjects selecting this model (%)", fontsize=FONT_PT)
    cbar.ax.tick_params(labelsize=TICK_PT)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_OUT, f"model_recovery_{h}{SUF}.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)

    if big:
        idx = pd.DataFrame({"index": range(1, n + 1), "model": list(win.columns)})
        idx.to_csv(os.path.join(TAB_OUT, f"model_recovery_index_{h}{SUF}.csv"), index=False)
    print(f"wrote {os.path.join(FIG_OUT, f'model_recovery_{h}{SUF}.pdf')}")


def main():
    os.makedirs(FIG_OUT, exist_ok=True)
    os.makedirs(TAB_OUT, exist_ok=True)
    group, subject = load_cells()

    per_h = {}
    for h in HORIZON_ORDER:
        if not any(k[0][0] == h for k in group):
            continue
        bic, win = matrices(group, subject, h)
        per_h[h] = (bic, win)
        have, want = int(bic.notna().sum().sum()), bic.size
        if have < want:
            print(f"\n[{HORIZON_NAME[h]}] {have}/{want} cells present "
                  f"({want - have} still missing) -- gaps are drawn grey")
        # Row-wise BIC relative to the generator's own fit: negative means the
        # candidate beat the true model.
        delta = bic.sub(pd.Series({g: bic.loc[g, g] for g in bic.index}), axis=0)
        bic.round(2).to_csv(os.path.join(TAB_OUT, f"model_recovery_bic_{h}{SUF}.csv"))
        delta.round(2).to_csv(os.path.join(TAB_OUT, f"model_recovery_dbic_{h}{SUF}.csv"))
        win.round(2).to_csv(os.path.join(TAB_OUT, f"model_recovery_winpct_{h}{SUF}.csv"))

        rows_done = [g for g in bic.index if bic.loc[g].notna().all()]
        hits = sum(bic.loc[g].idxmin() == g for g in rows_done)
        print(f"\n=== {HORIZON_NAME[h]}: {hits}/{len(rows_done)} generators recovered "
              f"(group-level summed BIC; complete rows only) ===")
        for g in rows_done:
            best = bic.loc[g].idxmin()
            mark = "OK " if best == g else "MISS"
            print(f"  {mark} gen {g} -> {best}   dBIC vs truth {delta.loc[g, best]:9.1f}"
                  f"   subject-level: truth {win.loc[g, g]:5.1f}%  best {win.loc[g].max():5.1f}%")

        # For each miss, split the BIC gap into the part the likelihood
        # explains and the part the parameter penalty explains. BIC_total =
        # k*sum_i log(n_obs_i) - 2*sum logL, so sum_i log(n_obs_i) is
        # recoverable per cell and is a property of the data, not the model.
        misses = [g for g in rows_done if bic.loc[g].idxmin() != g]
        if misses:
            print(f"  -- where the {len(misses)} miss(es) come from:")
            for g in misses:
                best = bic.loc[g].idxmin()
                mt, mb = group[(g, g)], group[(g, best)]
                logn = (mt["BIC"] + 2 * mt["sum logL"]) / mt["k"]
                d_ll = -2 * (mb["sum logL"] - mt["sum logL"])
                d_pen = (mb["k"] - mt["k"]) * logn
                print(f"     gen {g} -> {best}: dBIC {delta.loc[g, best]:9.1f}"
                      f" = fit {d_ll:9.1f} + penalty {d_pen:9.1f}"
                      f"   (k {mt['k']} -> {mb['k']})")

    if not per_h:
        raise SystemExit("no model recovery cells found")

    for h, (bic, win) in per_h.items():
        render_single(h, bic, win)

    if max(len(w.columns) for _, w in per_h.values()) > PANEL_MAX:
        # The three-panel overview only works while the grids are small enough
        # to carry their model names and per-cell numbers side by side.
        print(f"\nskipped the 3-panel overview: largest grid is "
              f"{max(len(w.columns) for _, w in per_h.values())} models "
              f"(> {PANEL_MAX}); see the per-horizon figures instead")
        return

    order = [h for h in HORIZON_ORDER if h in per_h]
    sizes = [len(per_h[h][1].columns) for h in order]
    fig, axes = plt.subplots(
        1, len(order), figsize=(WIDTH_IN, WIDTH_IN * 0.62),
        gridspec_kw={"width_ratios": sizes},
    )
    axes = np.atleast_1d(axes)

    for k, (h, ax) in enumerate(zip(order, axes)):
        bic, win = per_h[h]
        # aspect="auto", not "equal": the three matrices are 7x7, 5x5 and 6x6,
        # so equal-aspect cells give the panels three different heights, which
        # leaves their titles and top rows out of line and wastes most of the
        # figure box. Letting each fill its slot costs only square cells.
        im = ax.imshow(win.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        n = len(win.columns)
        ax.set_xticks(range(n))
        ax.set_yticks(range(len(win.index)))
        ax.set_xticklabels(win.columns, rotation=90, fontsize=TICK_PT, family="monospace")
        ax.set_yticklabels(win.index, fontsize=TICK_PT, family="monospace")
        ax.set_xlabel("fitted model", fontsize=FONT_PT)
        if k == 0:
            ax.set_ylabel("generating model", fontsize=FONT_PT)
        ax.set_title(HORIZON_NAME[h], fontsize=FONT_PT, pad=3)
        ax.text(0.0, 1.13, f"({'abc'[k]})", transform=ax.transAxes,
                fontsize=FONT_PT + 2, fontweight="bold", va="bottom", ha="left")

        for i, g in enumerate(win.index):
            for j, f in enumerate(win.columns):
                v = win.loc[g, f]
                if v < 0.5:
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=CELL_PT,
                        color="white" if v < 55 else "black")
            # Outline the group-level (summed BIC) winner for this generator.
            j = list(win.columns).index(bic.loc[g].idxmin())
            pad = 0.06
            ax.add_patch(plt.Rectangle(
                (j - 0.5 + pad, i - 0.5 + pad), 1 - 2 * pad, 1 - 2 * pad,
                fill=False, edgecolor=OUTLINE, linewidth=1.8, zorder=5))
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(win.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("subjects selecting this model (%)", fontsize=FONT_PT)
    cbar.ax.tick_params(labelsize=TICK_PT)

    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(FIG_OUT, f"model_recovery{SUF}.{ext}"),
                    dpi=600 if ext == "png" else None)
    print(f"\nwrote {os.path.join(FIG_OUT, f'model_recovery{SUF}.pdf')}")
    print(f"wrote per-horizon CSVs to {TAB_OUT}")


if __name__ == "__main__":
    main()
