"""The two descriptive panels of the participants' own behaviour.

    yellow_blue_fractions_stacked   what fraction of visits to each evidence
                                    state ended in a commitment, and to which
                                    colour, against the normative policy
    four_subjects_panel             four participants' draw and outcome
                                    distributions, with the model's ensemble

Both read the preprocessed human data; the second also needs a completed fit,
and says so rather than failing if one is absent.

Usage (from the repository root):

    SIM_ALGORITHM=de python3 scripts/figures/export_human_data_figures.py
    SUBJECTS=53,6,10,88 python3 scripts/figures/export_human_data_figures.py

Writes figures/Human_Data/.
"""
import argparse
import json
import os
import sys

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "recovery"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from src.config.loader import load_config  # noqa: E402
from src.pomdp import POMDPFactory  # noqa: E402
from src.utils.utils import analyze_meg_draws_color  # noqa: E402
from src.utils.plotting import (  # noqa: E402
    plot_yellow_blue_fractions_stacked, plot_multi_subjects_panel)

OUT = os.path.join(ROOT, "figures", "Human_Data")
HUMAN = os.path.join(ROOT, "data", "TrHu_NHB_light", "data_MEG",
                     "behdat_preprocessed.pkl")


def solved_policy(horizon):
    """The normative policy for one horizon, which the panels are drawn against."""
    model = POMDPFactory("vanilla")
    model.__init__(horizon_condition=horizon, max_cards_per_draw=5,
                   is_hazardous=True, verbose=False,
                   tau=1e-8, xi=0.0, subjective_cost=0.0)
    model.value_iteration()
    return model


def fractions_panel(human):
    """Commitment fractions at every evidence state, both horizons stacked."""
    counts = analyze_meg_draws_color(human)
    short, long_ = solved_policy("short"), solved_policy("long")
    plot_yellow_blue_fractions_stacked(
        short.best_actions, long_.best_actions, short.max_cards_per_draw,
        counts["counts_dict_yellow_short"], counts["counts_dict_blue_short"],
        counts["counts_dict_yellow_long"], counts["counts_dict_blue_long"],
        path=OUT)
    print(f"wrote {os.path.join(OUT, 'yellow_blue_fractions_stacked')}"
          ".[pdf|png|svg]")


def subjects_panel(human, subjects):
    """Four participants against the ensemble simulated from their own fits."""
    from recovery_post_analysis import build_ensemble_data

    with open(os.path.join(ROOT, "BIC", "best_models.json")) as fh:
        best = json.load(fh)

    frames = {}
    for horizon in ("short", "long"):
        cfg = load_config(os.path.join(
            ROOT, "data/simulation_configs",
            f"simulation_params_{best[horizon]}.py"))
        results_path = os.path.join(cfg.DATA_PATH, "results.pkl")
        if not os.path.exists(results_path):
            print(f"skipped: no fit at {os.path.relpath(results_path, ROOT)}; "
                  "run the fitting stage first")
            return
        results = pd.read_pickle(results_path)
        rows = results[results.subject_ID.astype(str).isin(
            [str(s) for s in subjects])]
        if rows.empty:
            print(f"skipped: none of {subjects} present in the {horizon} fit")
            return
        frames[horizon] = build_ensemble_data(
            cfg, rows, human, list(rows.subject_ID))

    (metrics_short, ens_short), (metrics_long, ens_long) = (
        frames["short"], frames["long"])

    # The panel looks subjects up with a plain dict get, so the ids passed to it
    # must be the ensemble's own keys. Ours arrive as strings from the command
    # line, and a type mismatch draws an empty panel captioned "data missing"
    # rather than failing, which is worse than an error.
    resolved = []
    for wanted in subjects:
        key = next((k for k in ens_short if str(k) == str(wanted)), None)
        if key is None or key not in ens_long:
            print(f"skipped: subject {wanted} has no ensemble in both horizons")
            return
        resolved.append(key)

    plot_multi_subjects_panel(
        resolved, ens_short, ens_long, metrics_short, metrics_long,
        path=OUT, filename="four_subjects_panel")
    print(f"wrote {os.path.join(OUT, 'four_subjects_panel')}.[pdf|png|svg] "
          f"for subjects {', '.join(str(k) for k in resolved)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default=os.environ.get("SUBJECTS", "53,6,10,88"),
                    help="the four subjects the panel shows, in the caption's order")
    args = ap.parse_args()

    if not os.path.exists(HUMAN):
        # Skip rather than fail: this is the only methods target that needs the
        # dataset, and a clone without it should still build the rest.
        print(f"skipped: {os.path.relpath(HUMAN, ROOT)} not found, "
              "run the preprocessing first")
        return
    os.makedirs(OUT, exist_ok=True)
    human = pd.read_pickle(HUMAN)

    fractions_panel(human)
    subjects_panel(human, [s.strip() for s in args.subjects.split(",") if s.strip()])


if __name__ == "__main__":
    main()
