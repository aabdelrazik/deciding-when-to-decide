"""Does the normative benchmark change if the ideal observer knows the true q?

The benchmark agent in the cost-of-departure analysis carries Beta(1,1) over
q in [0,1], i.e. it entertains generative probabilities the task never uses.
The task draws q from [0.50,0.56] or [0.60,0.66] with equal probability, so on
the yellow-probability scale the true prior is symmetric and piecewise uniform
on [0.34,0.40] U [0.44,0.50] U [0.50,0.56] U [0.60,0.66], a quarter of the mass
in each.

An agent that knows this treats each card as far weaker evidence than the
uniform-prior agent does, since it knows q can never exceed 0.66. The question
is whether that changes the policy, and if so whether it changes the reward the
benchmark achieves, which is what the reported shortfalls are measured against.

Nothing about the belief state changes. The likelihood is binomial in the
counts, so for any prior the posterior is proportional to prior(q) q^y (1-q)^b
and (y,b) remains sufficient. Only two maps change, and both reduce to weighted
sums over a grid in q:

    belief(y,b)   = P(q > 0.5 | y,b)
    trans(y,b,i)  = P(i yellows in the next draw | y,b)

Both are precomputed into lookup tables here and patched over the conjugate
versions, leaving value iteration and everything downstream untouched.

Usage (from scripts/):
    SIM_ALGORITHM=de python3 informed_prior_optimum.py
"""
import inspect
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from src.config.loader import load_config
from src.pomdp import POMDPFactory
from src.pomdp.pomdp import POMDP

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUTDIR = os.path.join(R, "BIC", "optimality")
N_GRID = int(os.environ.get("Q_GRID", "400"))      # points per interval
PILOT = int(os.environ.get("PILOT", "0"))

# True generative prior on the yellow probability. The task's p in [0.50,0.56]
# and [0.60,0.66] applies to whichever colour is correct, so the yellow scale
# carries each range and its mirror image, equally weighted.
INTERVALS = [(0.34, 0.40), (0.44, 0.50), (0.50, 0.56), (0.60, 0.66)]

NORMATIVE = {
    "belief_bias": 1.0, "exaggeration_factor": 1.0, "gamma": 1.0,
    "subjective_cost": 0.0, "hazard_lapse": 0.0, "patience": 0.0,
    "c_max": 0.0, "urgency_coefficient": 0.0, "urgency_slope": 0.0,
}


def q_grid():
    """Bin midpoints and weights. Midpoints keep q=0.5 off the grid, so the
    P(q>0.5) split is unambiguous at the shared interval endpoint."""
    qs, ws = [], []
    for lo, hi in INTERVALS:
        edges = np.linspace(lo, hi, N_GRID + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        qs.append(mid)
        ws.append(np.full(N_GRID, (1.0 / len(INTERVALS)) / N_GRID))
    q = np.concatenate(qs)
    w = np.concatenate(ws)
    return q, w / w.sum()


def build_tables(max_count, n_cards):
    """logZ[a,c] = log sum_q w(q) q^a (1-q)^c, plus the q>0.5 restriction.

    Every quantity needed is a ratio of these, because multiplying the
    likelihood by q^i (1-q)^(n-i) just shifts the exponents.
    """
    q, w = q_grid()
    lq, l1q, lw = np.log(q), np.log1p(-q), np.log(w)
    hi = q > 0.5

    A = max_count + n_cards + 1
    logZ = np.empty((A, A))
    logZ_hi = np.empty((A, A))
    for a in range(A):
        base = lw + a * lq
        for c in range(A):
            t = base + c * l1q
            logZ[a, c] = logsumexp(t)
            logZ_hi[a, c] = logsumexp(t[hi])
    return logZ, logZ_hi


def patch(cls, logZ, logZ_hi, n_cards):
    """Swap the conjugate belief and transition maps for the informed ones."""
    log_comb = np.array([gammaln(n_cards + 1) - gammaln(i + 1)
                         - gammaln(n_cards - i + 1) for i in range(n_cards + 1)])

    def calculate_transition_probability(self, i, num_yellows, num_blues):
        i = np.asarray(i, dtype=int)
        y = np.asarray(num_yellows, dtype=int)
        b = np.asarray(num_blues, dtype=int)
        i_c, y_c, b_c = np.broadcast_arrays(i, y, b)
        ok = (i_c >= 0) & (i_c <= n_cards)
        ii = np.clip(i_c, 0, n_cards)
        out = np.exp(log_comb[ii]
                     + logZ[y_c + ii, b_c + n_cards - ii]
                     - logZ[y_c, b_c])
        return np.where(ok, np.nan_to_num(out), 0.0)

    def _belief(y, b):
        y = np.asarray(y)
        b = np.asarray(b)
        yi = np.rint(y).astype(int)
        bi = np.rint(b).astype(int)
        # Non-integer counts would mean a non-unit exaggeration factor, which
        # the normative agent never has; the table is indexed by whole cards.
        assert np.allclose(y, yi) and np.allclose(b, bi), "non-integer counts"
        return np.exp(logZ_hi[yi, bi] - logZ[yi, bi])

    # Two signatures exist in this hierarchy. The base takes (num_blues,
    # num_yellows), blues first; POMDP_exaggerate takes the current draw and
    # the accumulated counts separately, yellow first, and scales the current
    # draw by the exaggeration factor before adding it.
    def belief_base(self, num_blues, num_yellows, q=0.5):
        assert q == 0.5, "informed prior tabulates the q>0.5 split only"
        return _belief(num_yellows, num_blues)

    def belief_exaggerate(self, curr_yellow, curr_blue, prev_yellow=1,
                          prev_blue=1, q=0.5):
        assert q == 0.5, "informed prior tabulates the q>0.5 split only"
        e = getattr(self, "exaggeration_factor", 1.0)
        return _belief(np.asarray(prev_yellow) + np.asarray(curr_yellow) * e,
                       np.asarray(prev_blue) + np.asarray(curr_blue) * e)

    names = list(inspect.signature(cls.calculate_belief_probability).parameters)
    cls.calculate_belief_probability = (
        belief_exaggerate if "curr_yellow" in names else belief_base)
    cls.calculate_transition_probability = calculate_transition_probability


def build(pomdp_type, kwargs):
    m = POMDPFactory(pomdp_type)
    ok = set(inspect.signature(type(m).__init__).parameters)
    m.__init__(**{k: v for k, v in kwargs.items() if k in ok})
    m.value_iteration()
    return m


def reachable_actions(model):
    """Best action at every state the agent can actually occupy.

    action_values is (draw, n_yellow, n_blue, 3) and the counts index the axes
    directly, so a state is reachable only when yellows + blues equals the
    cards dealt by that draw.
    """
    av = np.asarray(model.action_values)
    acts = np.argmax(av, axis=-1)
    n_draw, n_y, n_b = av.shape[:3]
    gy, gb = np.meshgrid(np.arange(n_y), np.arange(n_b), indexing="ij")
    mask = np.zeros(acts.shape, dtype=bool)
    for d in range(n_draw):
        mask[d] = (gy + gb) == d * model.max_cards_per_draw
    return acts, mask


def score(model, sequences):
    rew, draws, correct, missed = [], [], [], []
    for seq in sequences:
        res = model.simulate_cards_pomdp(given_sequence=True, card_sequence=seq)
        r = float(res["reward"])
        rew.append(r)
        draws.append(float(res["num_draws"]))
        correct.append(1.0 if r == 2 else 0.0)
        missed.append(1.0 if r == -1 else 0.0)
    if not rew:
        return dict(reward=np.nan, draws=np.nan, correct=np.nan, missed=np.nan)
    return dict(reward=float(np.mean(rew)), draws=float(np.mean(draws)),
                correct=float(np.mean(correct)), missed=float(np.mean(missed)))


def main():
    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        best = json.load(fh)
    D = os.path.join(R, "data/TrHu_NHB_light/data_MEG")
    ev = pd.read_pickle(os.path.join(D, "all_subject_evidence_dicts_full_sequence.pkl"))

    subjects = list(ev.index)[:PILOT] if PILOT else list(ev.index)
    print(f"{len(subjects)} subjects; q grid {N_GRID} points per interval "
          f"over {INTERVALS}")

    rows, pol = [], {}
    for f, horizons in (("short", ("short",)), ("long", ("long",)),
                        ("combined", ("short", "long"))):
        cfg = load_config(os.path.join(R, "data/simulation_configs",
                                       f"simulation_params_{best[f]}.py"))
        for h in horizons:
            base = dict(horizon_condition=h,
                        max_cards_per_draw=cfg.MAX_CARDS_PER_DRAW,
                        is_hazardous=True, verbose=False)
            for k, v in NORMATIVE.items():
                base.setdefault(k, getattr(cfg, k.upper(), v))
            kw = {**base, **NORMATIVE, "is_hazardous": True,
                  "tau": 1e-8, "xi": 0.0}

            # 1. the benchmark as published: Beta(1,1) over [0,1]
            import importlib
            import src.pomdp.pomdp as P
            importlib.reload(P)
            uni = build(cfg.POMDP_TYPE, kw)

            # 2. same agent, told the true prior
            n_cards = cfg.MAX_CARDS_PER_DRAW
            max_count = int(max(uni.max_yellows, uni.max_blues))
            logZ, logZ_hi = build_tables(max_count, n_cards)
            for c in (P.POMDP, P.POMDP_Urgency, P.POMDP_Forgetting,
                      P.POMDP_Exaggeration, P.POMDP_exaggerate):
                patch(c, logZ, logZ_hi, n_cards)
            inf = build(cfg.POMDP_TYPE, kw)

            au, mask = reachable_actions(uni)
            ai, _ = reachable_actions(inf)
            agree = float((au[mask] == ai[mask]).mean())
            pol[(f, h)] = (agree, int(mask.sum()))
            print(f"\n== {f}/{h}: policies agree on {100*agree:.1f}% of "
                  f"{int(mask.sum())} reachable states")

            for sid in subjects:
                seqs = list(ev.loc[sid, h]["draw_yellow_blue_action_outcome"])
                su, si = score(uni, seqs), score(inf, seqs)
                rows.append(dict(subject_ID=str(sid), fit=f, horizon=h,
                                 n_games=len(seqs),
                                 uniform_reward=su["reward"], informed_reward=si["reward"],
                                 uniform_draws=su["draws"], informed_draws=si["draws"],
                                 uniform_correct=su["correct"], informed_correct=si["correct"],
                                 uniform_missed=su["missed"], informed_missed=si["missed"]))
            importlib.reload(P)

    df = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(os.path.join(OUTDIR, "informed_prior_optimum.csv"), index=False)

    print("\n" + "=" * 78)
    print("BENCHMARK REWARD PER GAME: uniform prior vs correctly-informed prior")
    print("=" * 78)
    for f in df.fit.unique():
        s = df[df.fit == f]
        du = s.uniform_reward.mean()
        di = s.informed_reward.mean()
        n = s.n_games.mean()
        print(f"  {f:6}  uniform {du:+.4f}/game   informed {di:+.4f}/game   "
              f"diff {di-du:+.4f}  ({(di-du)*n*len(s)/len(s):+.2f} points over "
              f"{n:.0f} games)")
        print(f"          draws  {s.uniform_draws.mean():.2f} vs "
              f"{s.informed_draws.mean():.2f}   "
              f"correct {s.uniform_correct.mean():.3f} vs {s.informed_correct.mean():.3f}   "
              f"missed {s.uniform_missed.mean():.3f} vs {s.informed_missed.mean():.3f}")
    print(f"\nwrote {os.path.join(OUTDIR, 'informed_prior_optimum.csv')}")


if __name__ == "__main__":
    main()
