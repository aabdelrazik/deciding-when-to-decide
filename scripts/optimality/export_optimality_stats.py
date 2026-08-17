"""Every number quoted in the cost-of-departure paragraph, from one command.

The paragraph mixes three kinds of quantity, so they are derived together and
stay consistent with each other whenever the benchmark changes:

  1. session totals and shortfalls  (points earned, SD, median, IQR, Wilcoxon)
  2. sampling                       (draws taken, human vs normative, counts)
  3. where the short-horizon shortfall comes from, which needs the normative
     policy replayed against each subject's own stopping point rather than
     just its summary score

Section 3 is the reason this is a script and not a spreadsheet: it re-solves
the normative POMDP and walks every game, so it has to agree with whichever
benchmark produced the CSV. Pass INFORMED_PRIOR=1 to match a benchmark built
with the task's true prior.

Usage (from scripts/):
    SIM_ALGORITHM=de OPTIMALITY_SRC=BIC/optimality/optimality_cost_informed.csv \
    INFORMED_PRIOR=1 python3 export_optimality_stats.py
"""
import inspect
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scipy import stats

from src.config.loader import load_config
from src.pomdp import POMDPFactory
import src.pomdp.pomdp as _P
from informed_prior_optimum import build_tables, patch as _patch_prior

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.environ.get("OPTIMALITY_SRC") or os.path.join(
    R, "BIC", "optimality", "optimality_cost.csv")
OUT = os.path.join(R, "BIC", "optimality")
INFORMED_PRIOR = os.environ.get("INFORMED_PRIOR", "0") == "1"

NORMATIVE = {
    "belief_bias": 1.0, "exaggeration_factor": 1.0, "gamma": 1.0,
    "subjective_cost": 0.0, "hazard_lapse": 0.0, "patience": 0.0,
    "c_max": 0.0, "urgency_coefficient": 0.0, "urgency_slope": 0.0,
}

if INFORMED_PRIOR:
    _logZ, _logZ_hi = build_tables(14 * 5, 5)
    for _c in (_P.POMDP, _P.POMDP_Urgency, _P.POMDP_Forgetting,
               _P.POMDP_Exaggeration, _P.POMDP_exaggerate):
        _patch_prior(_c, _logZ, _logZ_hi, 5)


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}".replace("e-", r" \times 10^{-") + "}"


def totals(df, fit):
    """Section 1 and 2: session points and draws, human against the benchmark."""
    s = df[df.fit == fit]
    n = s.n_games.values
    h, o = s.human.values * n, s.optimal_full.values * n
    gap = o - h
    w = stats.wilcoxon(h, o)
    hd, od = s.human_draws.values, s.optimal_full_draws.values
    wd = stats.wilcoxon(hd, od)
    return dict(
        fit=fit, n_subjects=len(s), n_games=float(np.mean(n)),
        human_points=h.mean(), optimal_points=o.mean(),
        shortfall=gap.mean(), shortfall_sd=gap.std(ddof=1),
        shortfall_median=np.median(gap),
        shortfall_iqr_lo=np.percentile(gap, 25), shortfall_iqr_hi=np.percentile(gap, 75),
        wilcoxon_p=w.pvalue, n_subjects_above_optimum=int((h > o).sum()),
        human_correct=s.human_correct.mean() if "human_correct" in s else np.nan,
        optimal_correct=s.optimal_full_correct.mean(),
        human_draws=hd.mean(), optimal_draws=od.mean(),
        draws_diff=(hd - od).mean(), draws_p=wd.pvalue,
        n_subjects_below_optimal_draws=int((hd < od).sum()),
    )


def build_normative(cfg, horizon):
    base = dict(horizon_condition=horizon,
                max_cards_per_draw=cfg.MAX_CARDS_PER_DRAW,
                is_hazardous=True, verbose=False)
    for k, v in NORMATIVE.items():
        base.setdefault(k, getattr(cfg, k.upper(), v))
    kw = {**base, **NORMATIVE, "is_hazardous": True, "tau": 1e-8, "xi": 0.0}
    m = POMDPFactory(cfg.POMDP_TYPE)
    ok = set(inspect.signature(type(m).__init__).parameters)
    m.__init__(**{k: v for k, v in kw.items() if k in ok})
    m.value_iteration()
    return m


def misallocation(model, seqs_full, seqs_cut):
    """Section 3: where the subject's draws went, relative to the benchmark.

    Splits the shortfall into stopping at the wrong time and choosing against
    the evidence, which are the two accounts the paragraph weighs.

    The comparison is per game: the benchmark replays the same full sequence
    and its own stopping draw is compared with the subject's. Reading the
    policy array directly would be faster but its axes differ by POMDP class
    (the exaggeration models index the current draw separately, since the
    current draw is reweighted), so the simulator is the portable route.

    Rows are (draw, cumulative_yellow, cumulative_blue, action, outcome), with
    cumulative counts. Actions are 0 commit yellow, 1 commit blue, 2 wait, and
    outcome -1 marks a game that hit the deadline undecided.
    """
    WAIT = 2
    n_games = stop_early = over_sample = missed = against = 0
    # Points attributable to choosing against the evidence: what the subject
    # would have earned had they picked the majority colour at their own
    # stopping point, holding the stopping time fixed. The remainder of the
    # shortfall is then attributable to when they stopped, not what they chose.
    pts_actual = pts_if_majority = 0.0
    for full, cut in zip(seqs_full, seqs_cut):
        n_games += 1
        if int(cut[-1][4]) == -1 or int(cut[-1][3]) == WAIT:
            missed += 1
            continue
        t = len(cut)
        norm_t = float(model.simulate_cards_pomdp(
            given_sequence=True, card_sequence=full)["num_draws"])
        if t < norm_t:
            stop_early += 1
        elif t > norm_t:
            over_sample += 1
        y, b = int(cut[-1][1]), int(cut[-1][2])
        chosen_yellow = int(cut[-1][3]) == 0
        outcome = float(cut[-1][4])
        pts_actual += outcome
        if y == b:
            pts_if_majority += outcome
        else:
            correct_is_yellow = (outcome > 0) == chosen_yellow
            pts_if_majority += 2.0 if (y > b) == correct_is_yellow else -2.0
        if y != b and chosen_yellow != (y > b):
            against += 1
    return dict(n_games=n_games,
                pts_from_choice_errors=(pts_if_majority - pts_actual),
                pct_stopped_early=100 * stop_early / n_games,
                pct_over_sampled=100 * over_sample / n_games,
                pct_missed_deadline=100 * missed / n_games,
                pct_against_majority=100 * against / n_games)


def main():
    df = pd.read_csv(SRC)
    print(f"source: {SRC}")
    print(f"informed prior: {INFORMED_PRIOR}\n")

    rows = [totals(df, f) for f in ("short", "long", "combined") if f in set(df.fit)]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "optimality_stats.csv"), index=False)

    print("=" * 78)
    print("SESSION POINTS AND SHORTFALL")
    print("=" * 78)
    for r in rows:
        print(f"  {r['fit']:9} humans {r['human_points']:6.1f} of {r['optimal_points']:6.1f} "
              f"points over {r['n_games']:.0f} games")
        print(f"            shortfall {r['shortfall']:.1f} (SD {r['shortfall_sd']:.1f}, "
              f"median {r['shortfall_median']:.1f}, IQR {r['shortfall_iqr_lo']:.0f} to "
              f"{r['shortfall_iqr_hi']:.0f}); Wilcoxon p={r['wilcoxon_p']:.2e}")
        print(f"            correct {100*r['optimal_correct']:.1f}% normative; "
              f"{r['n_subjects_above_optimum']} of {r['n_subjects']} subjects beat the optimum")
        print(f"            draws {r['human_draws']:.2f} human vs {r['optimal_draws']:.2f} "
              f"normative (diff {r['draws_diff']:+.2f}, p={r['draws_p']:.2e}, "
              f"{r['n_subjects_below_optimal_draws']} of {r['n_subjects']} below)")

    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        best = json.load(fh)
    D = os.path.join(R, "data/TrHu_NHB_light/data_MEG")
    ev = pd.read_pickle(os.path.join(D, "all_subject_evidence_dicts_full_sequence.pkl"))
    ev_cut = pd.read_pickle(os.path.join(D, "all_subject_evidence_dicts.pkl"))

    print("\n" + "=" * 78)
    print("WHERE THE SHORTFALL COMES FROM (normative policy at human stopping points)")
    print("=" * 78)
    mis = []
    for f, h in (("short", "short"), ("long", "long")):
        cfg = load_config(os.path.join(R, "data/simulation_configs",
                                       f"simulation_params_{best[f]}.py"))
        model = build_normative(cfg, h)
        acc = []
        for sid in ev.index:
            acc.append(misallocation(
                model,
                list(ev.loc[sid, h]["draw_yellow_blue_action_outcome"]),
                list(ev_cut.loc[sid, h]["draw_yellow_blue_action_outcome"])))
        a = pd.DataFrame(acc).mean()
        a["fit"] = f
        mis.append(a)
        print(f"  {f:9} stopped early {a['pct_stopped_early']:.1f}%   "
              f"over-sampled {a['pct_over_sampled']:.1f}%   "
              f"missed deadline {a['pct_missed_deadline']:.1f}%   "
              f"chose against majority {a['pct_against_majority']:.1f}%")
    pd.DataFrame(mis).to_csv(os.path.join(OUT, "optimality_misallocation.csv"), index=False)
    print(f"\nwrote {os.path.join(OUT, 'optimality_stats.csv')}")
    print(f"wrote {os.path.join(OUT, 'optimality_misallocation.csv')}")


if __name__ == "__main__":
    main()
