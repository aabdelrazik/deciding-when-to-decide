"""Draw the task schematic: one evidence sample and the response screen.

The figure in the manuscript was made by hand, which left the one illustration
of the task as the only asset with no code behind it. This reproduces it: the
framed display holding five cards in random positions, the horizon bar, and the
two response options with the cursor over one of them.

The frame colour carries the horizon condition, green or pink, counterbalanced
across participants, so both are drawn from the same function.

Usage (from the repository root):

    python3 scripts/figures/export_task_figure.py
    python3 scripts/figures/export_task_figure.py --horizon short --seed 7
    python3 scripts/figures/export_task_figure.py --yellow 2      # 2 yellow, 3 blue

Writes figures/illustrate/MEG_task.[pdf|png|svg].
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from src.utils.plotting import save_figure, set_plot_style  # noqa: E402

# Sampled from the original illustration so the regenerated figure keeps the
# task's own colours rather than matplotlib's defaults.
YELLOW = "#F5A623"
BLUE = "#5BC0EB"
PINK = "#EE4C8C"          # one horizon condition
GREEN = "#3FB24F"         # the other
BAR = "#2E7EBB"
MARKER = "#E5322D"
INK = "#1A1A1A"

# A 5 card display, five slots the cards may occupy, in display coordinates.
# Positions are jittered within the frame rather than laid out on a grid,
# because the task presents them in random positions.
WIDTH, HEIGHT = 866.0, 1210.0
FRAME = (88, 570, 694, 593)          # x, y, w, h of the outer frame
FRAME_LW = 40
CARD = 74


def draw_display(ax, horizon, rng, n_yellow):
    """The framed evidence display holding one sample of five cards."""
    x, y, w, h = FRAME
    edge = PINK if horizon == "long" else GREEN
    # the frame is a thick border, so draw the coloured block and inset a white
    # panel rather than relying on a line width that does not scale with data
    ax.add_patch(Rectangle((x, y), w, h, facecolor=edge, edgecolor="none",
                           zorder=1))
    ax.add_patch(Rectangle((x + FRAME_LW, y + FRAME_LW),
                           w - 2 * FRAME_LW, h - 2 * FRAME_LW,
                           facecolor="white", edgecolor="none", zorder=2))

    inner = (x + FRAME_LW, y + FRAME_LW, w - 2 * FRAME_LW, h - 2 * FRAME_LW)
    colours = [YELLOW] * n_yellow + [BLUE] * (5 - n_yellow)
    rng.shuffle(colours)
    for cx, cy, colour in zip(*sample_positions(inner, rng), colours):
        ax.add_patch(Rectangle((cx, cy), CARD, CARD, facecolor=colour,
                               edgecolor="none", zorder=3))


def sample_positions(inner, rng, n=5, tries=400):
    """Five non-overlapping card positions inside the panel."""
    x0, y0, w, h = inner
    pad = 18
    placed = []
    for _ in range(tries):
        if len(placed) == n:
            break
        cx = rng.uniform(x0 + pad, x0 + w - CARD - pad)
        cy = rng.uniform(y0 + pad, y0 + h - CARD - pad)
        # allow cards to touch, as in the original, but never to overlap
        if all(abs(cx - px) >= CARD or abs(cy - py) >= CARD
               for px, py in placed):
            placed.append((cx, cy))
    while len(placed) < n:                       # pathological seed, fall back
        placed.append((x0 + pad + CARD * len(placed), y0 + pad))
    return [p[0] for p in placed], [p[1] for p in placed]


def draw_horizon_bar(ax):
    """The bar below the display, with the marker showing progress."""
    ax.add_patch(FancyBboxPatch((155, 462), 540, 30,
                                boxstyle="round,pad=0,rounding_size=15",
                                facecolor=BAR, edgecolor="none", zorder=3))
    ax.add_patch(FancyBboxPatch((398, 425), 32, 105,
                                boxstyle="round,pad=0,rounding_size=16",
                                facecolor=MARKER, edgecolor="none", zorder=4))


def draw_responses(ax):
    """The two response options the participant chooses between."""
    ax.add_patch(Rectangle((88, 90), 310, 310, facecolor=YELLOW,
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((455, 90), 327, 310, facecolor=BLUE,
                           edgecolor="none", zorder=3))


def hand_parts():
    """The cursor, as a union of rounded boxes: finger, fist, thumb."""
    return [
        # The finger has to be clearly narrower than the fist is wide, or the
        # cursor reads as a raised thumb rather than a pointing hand.
        FancyBboxPatch((268, 30), 34, 205,               # index finger
                       boxstyle="round,pad=0,rounding_size=17"),
        FancyBboxPatch((244, 8), 176, 108,               # fist
                       boxstyle="round,pad=0,rounding_size=24"),
        FancyBboxPatch((238, 52), 58, 58,                # thumb
                       boxstyle="round,pad=0,rounding_size=22"),
    ]


def draw_hand(ax):
    """Outline the union without drawing the seams between the parts.

    Each part is stroked in ink first, which merges into one silhouette, then
    filled white with no edge, which hides every internal boundary.
    """
    for patch in hand_parts():
        patch.set(facecolor=INK, edgecolor=INK, linewidth=14,
                  joinstyle="round", zorder=5)
        ax.add_patch(patch)
    for patch in hand_parts():
        patch.set(facecolor="white", edgecolor="none", zorder=6)
        ax.add_patch(patch)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", choices=("short", "long"), default="long",
                    help="which frame colour to draw (default long, pink)")
    ap.add_argument("--yellow", type=int, default=3,
                    help="how many of the five cards are yellow (default 3)")
    ap.add_argument("--seed", type=int, default=3,
                    help="card positions, fixed so the figure is reproducible")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figures", "illustrate"))
    args = ap.parse_args()
    if not 0 <= args.yellow <= 5:
        ap.error("--yellow must be between 0 and 5")

    set_plot_style()
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(4.3, 6.0))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_display(ax, args.horizon, rng, args.yellow)
    draw_horizon_bar(ax)
    draw_responses(ax)
    draw_hand(ax)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, "MEG_task")
    save_figure(fig, base)
    print(f"wrote {base}.[pdf|png|svg]  "
          f"({args.horizon} horizon, {args.yellow} yellow, seed {args.seed})")


if __name__ == "__main__":
    main()
