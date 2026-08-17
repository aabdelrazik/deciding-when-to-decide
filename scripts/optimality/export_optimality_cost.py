"""How many points the fitted heuristics cost, relative to the normative optimum.

The model comparison says which mechanisms the *fit* requires. It says nothing
about what they cost the subject, and the two need not agree: a mechanism can be
indispensable for likelihood and nearly free in reward, or cheap in likelihood
and expensive in points. This script measures the second thing.

Every policy is scored by the task's real payoff (+2 correct, -2 incorrect, -1
missed deadline, 0 tie), never by the subject's subjective one. That matters
because subjective_cost is a private penalty: an agent maximizing the subject's
internal utility is not the agent that earns the most, and calling a mechanism
"suboptimal" only means something against the objective payoff.

Policies are compared on each subject's own card sequences, replayed to the
realized deadline, so sequence difficulty and the deadline draw are held fixed
and only the policy differs. The sequences come from the _full_sequence variant
of the evidence dicts, not the plain one: the latter truncates each game at the
subject's decision, and an agent that wants to wait *longer* than the subject
did would run out of cards. The trailing action/outcome fields of each game
carry the subject's own choice and payoff, which gives the human arm for free.

Two reference policies are reported, because they answer different questions:

  optimal_full       everything normative and the policy near-deterministic.
                     This is the genuine ceiling on these sequences, and
                     optimal_full - human is the total cost of being human.
  normative_matched  normative belief and value settings, but the subject's own
                     fitted tau and xi. Deliberately NOT called an optimum: it
                     is not maximized subject to that noise level, it is just
                     the textbook parameter setting executed as noisily as the
                     subject executes. Whether it beats the subject's own
                     parameters is the question of interest, since if it loses
                     then the fitted distortions are compensating for noise
                     rather than merely wasting points.

Draws, accuracy and deadline misses are recorded per variant alongside reward,
so the same run also answers how many draws each mechanism is worth.

Usage (from scripts/):
    SIM_ALGORITHM=de python3 export_optimality_cost.py            # all subjects
    SIM_ALGORITHM=de PILOT=3 python3 export_optimality_cost.py    # quick check
"""
import inspect
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(R)

from src.config.loader import load_config
from src.pomdp import POMDPFactory
import src.pomdp.pomdp as _P

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from informed_prior_optimum import build_tables, patch as _patch_prior

# INFORMED_PRIOR=1 gives every agent built here the task's true prior over the
# generative probability instead of the conjugate Beta(1,1) over [0,1]. The
# belief state is unchanged (the counts stay sufficient); only the count-to-
# belief and count-to-transition maps are swapped. OPTIMAL_ONLY=1 drops the
# variants the manuscript no longer reports, leaving the normative benchmark.
INFORMED_PRIOR = os.environ.get("INFORMED_PRIOR", "0") == "1"
OPTIMAL_ONLY = os.environ.get("OPTIMAL_ONLY", "0") == "1"
if INFORMED_PRIOR:
    _MAXC = 14 * 5
    _logZ, _logZ_hi = build_tables(_MAXC, 5)
    for _c in (_P.POMDP, _P.POMDP_Urgency, _P.POMDP_Forgetting,
               _P.POMDP_Exaggeration, _P.POMDP_exaggerate):
        _patch_prior(_c, _logZ, _logZ_hi, 5)
    print("informed prior active: benchmark uses the task's generative ranges")

PILOT = int(os.environ.get("PILOT", "0"))
N_REPEATS = int(os.environ.get("N_REPEATS", "20"))
SEED = int(os.environ.get("SEED", "0"))
OUTDIR = os.path.join(R, "BIC", "optimality")
# a parallel re-run must not clobber a serial one still in flight
SUFFIX = os.environ.get("OUT_SUFFIX", "")
N_JOBS = int(os.environ.get("N_JOBS", "1"))

# Values at which each mechanism stops distorting anything. Neutral, not zero:
# belief_bias and exaggeration_factor are multiplicative and neutral at 1, while
# the regulation function is switched off by flattening the sigmoid to zero,
# matching how the BIC ablation switches it off.
NORMATIVE = {
    "belief_bias": 1.0,
    "exaggeration_factor": 1.0,
    "gamma": 1.0,
    "subjective_cost": 0.0,
    "hazard_lapse": 0.0,
    "patience": 0.0,
    "c_max": 0.0,
    "urgency_coefficient": 0.0,
    "urgency_slope": 0.0,
}
REGULATION = ("patience", "c_max", "urgency_coefficient", "urgency_slope")

# One lesion per mechanism the manuscript argues about. Each restores that
# mechanism to its normative value and leaves everything else at the subject's
# fitted value, so the recovered reward is attributable to that mechanism alone.
LESIONS = {
    "no_regulation": REGULATION,
    "no_exaggeration": ("exaggeration_factor",),
    "no_forgetting": ("gamma",),
    "no_belief_bias": ("belief_bias",),
    "no_subjective_cost": ("subjective_cost",),
    "no_hazard_lapse": ("hazard_lapse",),
}


def games(ev: pd.DataFrame, sid, horizon: str) -> list:
    """That subject's games for one horizon, as full card sequences."""
    return list(ev.loc[sid, horizon]["draw_yellow_blue_action_outcome"])


def human_reward(seqs: list) -> float:
    """What the subject actually earned per game, from the stored outcomes."""
    return float(np.mean([float(s[-1][4]) for s in seqs])) if seqs else np.nan


def build(pomdp_type: str, kwargs: dict):
    """Instantiate and solve, passing only what this POMDP class accepts.

    POMDPFactory hands back an already-constructed instance, so the real
    configuration happens by calling __init__ again on it, as the fitting code
    does. Kwargs are filtered by signature because the classes differ in which
    mechanisms they take, and passing an unknown one is a TypeError rather than
    a silently ignored argument.
    """
    m = POMDPFactory(pomdp_type)
    ok = set(inspect.signature(type(m).__init__).parameters)
    m.__init__(**{k: v for k, v in kwargs.items() if k in ok})
    m.value_iteration()
    return m


def score(model, sequences, n_repeats: int) -> dict:
    """Replay these sequences under this policy and summarize the outcome.

    Reward is the task's objective payoff, so the categories are recoverable
    from it directly: +2 correct, -2 incorrect, -1 still waiting at the
    deadline, 0 when the sequence ends tied and no answer is right.
    """
    rew, draws, correct, missed = [], [], [], []
    for seq in sequences:
        for _ in range(n_repeats):
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
    np.random.seed(SEED)

    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        best = json.load(fh)

    D = os.path.join(R, "data/TrHu_NHB_light/data_MEG")
    ev = pd.read_pickle(os.path.join(
        D, "all_subject_evidence_dicts_full_sequence.pkl"))
    # The full-sequence file runs to the deadline and parks the subject's action
    # on the last row, so it cannot say where they stopped. The plain file is
    # truncated at the decision, which makes its length the subject's draw count.
    ev_cut = pd.read_pickle(os.path.join(D, "all_subject_evidence_dicts.pkl"))

    # One fit per horizon condition. "combined" shares a single parameter vector
    # across both horizons, so it is solved once per horizon_condition and its
    # games pooled; short and long each see only their own games.
    FITS = {"short": ("short",), "long": ("long",), "combined": ("short", "long")}
    cfgs, fits = {}, {}
    for f in FITS:
        cfg = load_config(os.path.join(
            R, "data/simulation_configs", f"simulation_params_{best[f]}.py"))
        cfgs[f] = cfg
        res = pd.read_pickle(os.path.join(cfg.DATA_PATH, "results.pkl"))
        fits[f] = {str(r.subject_ID): dict(zip(cfg.PARAM_ORDER, r.fit_params_ga))
                   for r in res.itertuples()}
        print(f"{f:8} winner {best[f]}  params {list(cfg.PARAM_ORDER)}")

    subjects = list(ev.index)[:PILOT] if PILOT else list(ev.index)

    def one(sid_raw):
        """Every policy variant for one subject. Independent of all others."""
        sid = str(sid_raw)
        # Seed from the subject rather than a global stream, so the result does
        # not depend on how the work is distributed across workers. crc32 and
        # not hash(): Python salts the hash of a string per process, which made
        # this vary from run to run, which is the opposite of the intent.
        np.random.seed((zlib.crc32(sid.encode()) ^ SEED) % (2**31))
        seqs = {h: games(ev, sid_raw, h) for h in ("short", "long")}
        out = []
        for f, horizons in FITS.items():
            cfg = cfgs[f]
            if sid not in fits[f]:
                continue
            fitted = dict(fits[f][sid])
            acc = {}
            for h in horizons:
                base = dict(horizon_condition=h,
                            max_cards_per_draw=cfg.MAX_CARDS_PER_DRAW,
                            is_hazardous=True, verbose=False)
                for k, v in NORMATIVE.items():
                    base.setdefault(k, getattr(cfg, k.upper(), v))
                kw = {**base, "tau": cfg.TAU, "xi": cfg.XI, **fitted}
                norm = {**NORMATIVE, "is_hazardous": True}
                variants = {"fitted": kw,
                            "optimal_full": {**kw, **norm, "tau": 1e-8, "xi": 0.0},
                            "normative_matched": {**kw, **norm}}
                for name, keys in LESIONS.items():
                    if not any(k in fitted for k in keys):
                        continue
                    variants[name] = {**kw, **{k: NORMATIVE[k] for k in keys}}
                if OPTIMAL_ONLY:
                    variants = {"optimal_full": variants["optimal_full"]}
                for name, kwargs in variants.items():
                    reps = 1 if name == "optimal_full" else N_REPEATS
                    st = score(build(cfg.POMDP_TYPE, kwargs), seqs[h], reps)
                    acc.setdefault(name, []).append((len(seqs[h]), st))
            cut = [x for h in horizons for x in games(ev_cut, sid_raw, h)]
            allseq = [x for h in horizons for x in seqs[h]]
            row = dict(subject_ID=sid, fit=f, task=best[f],
                       human=human_reward(allseq),
                       human_draws=float(np.mean([len(x) for x in cut])),
                       n_games=len(allseq))
            for name, parts in acc.items():
                w = np.array([n for n, _ in parts], float)
                for stat in ("reward", "draws", "correct", "missed"):
                    v = np.array([st[stat] for _, st in parts], float)
                    key = name if stat == "reward" else f"{name}_{stat}"
                    row[key] = float(np.average(v, weights=w))
            out.append(row)
        return out

    rows = []
    for r in Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one)(s_) for s_ in subjects):
        rows.extend(r)

    df = pd.DataFrame(rows)
    df["gap_full"] = df["optimal_full"] - df["human"]
    # OPTIMAL_ONLY leaves only the benchmark, so the derived gaps and the
    # per-variant summary below are skipped rather than KeyError'ing.
    if "normative_matched" in df:
        df["gap_matched"] = df["normative_matched"] - df["human"]
    if "fitted" in df:
        df["model_error"] = df["fitted"] - df["human"]   # validation, should be ~0

    os.makedirs(OUTDIR, exist_ok=True)
    tag = f"_pilot{PILOT}" if PILOT else ""
    out = os.path.join(OUTDIR, f"optimality_cost{tag}{SUFFIX}.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} subject-fit rows)")

    print("\nmean points per game (and mean draws):")
    for f in FITS:
        s = df[df.fit == f]
        if s.empty:
            continue
        print(f"  --- {f} ({best[f]}) ---")
        print(f"    human              {s.human.mean():+.3f}  "
              f"draws {s.human_draws.mean():.2f}")
        if "fitted" in df:
            print(f"    fitted             {s.fitted.mean():+.3f}  "
                  f"draws {s.fitted_draws.mean():.2f}   "
                  f"[model error {s.model_error.mean():+.3f}]")
        print(f"    optimal_full       {s.optimal_full.mean():+.3f}  "
              f"draws {s.optimal_full_draws.mean():.2f}   "
              f"GAP {s.gap_full.mean():+.3f}")
        if OPTIMAL_ONLY:
            continue
        print(f"    normative_matched  {s.normative_matched.mean():+.3f}  "
              f"draws {s.normative_matched_draws.mean():.2f}")
        for name in LESIONS:
            if name in s and s[name].notna().any():
                print(f"      {name:20} {s[name].mean():+.3f}  "
                      f"draws {s[name+'_draws'].mean():.2f}   "
                      f"(vs fitted {s[name].mean() - s.fitted.mean():+.3f} pts, "
                      f"{s[name+'_draws'].mean() - s.fitted_draws.mean():+.2f} draws)")

    if "gap_full" in df:
        symptom_scan(df)


def symptom_scan(df: pd.DataFrame):
    """Does the gap to optimality widen with obsessive-compulsive symptoms?"""
    from scipy.stats import spearmanr

    fa = pd.read_csv(os.path.join(R, "data/TrHu_NHB_light/data_MEG/fa_scores.csv"))
    fa = fa.rename(columns={"ML1": "FA1", "ML2": "FA2", "ML3": "FA3"})
    fa["subject_ID"] = fa["userID"].astype(str)
    m = df.merge(fa[["subject_ID", "FA2", "OCIR_total"]], on="subject_ID", how="left")

    print("\ncost of suboptimality vs symptoms (Spearman):")
    for f in m.fit.unique():
        s = m[m.fit == f]
        for measure in ("FA2", "OCIR_total"):
            ok = s[[measure, "gap_full"]].dropna()
            if len(ok) < 10:
                continue
            r, p = spearmanr(ok[measure], ok["gap_full"])
            star = "*" if p < 0.05 else " "
            print(f"  {f:8} gap_full vs {measure:11} "
                  f"rho={r:+.3f} p={p:.3f} N={len(ok)} {star}")
    m.to_csv(os.path.join(OUTDIR, "optimality_cost_with_symptoms.csv"), index=False)


if __name__ == "__main__":
    main()
