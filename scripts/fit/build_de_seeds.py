"""Build per-subject starting points for the DE refit.

For each model, find every model whose configuration its own search space can
reach (a nested neighbour), and seed each subject at whichever neighbour fits
that subject best. Because the seed is a point the larger model can occupy, a
converged fit cannot end up worse than the neighbour it contains, which is what
removes the nesting violations left by the unseeded run.

Set POMDP_SEEDS_COMMIT=1 to do the same for the commit-likelihood fits instead:
the commit configs carry the same TASK names as the full-fit ones, so the two
families cannot be loaded together without colliding, and their results live
under data/POMDP_commit/ rather than data/POMDP/.
"""
import glob, itertools, os, pickle, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
from src.config.loader import load_config

COMMIT = os.environ.get("POMDP_SEEDS_COMMIT", "") == "1"
SUBDIR = "POMDP_commit" if COMMIT else "POMDP"
OUT_NAME = "_de_commit_seeds.pkl" if COMMIT else "_de_seeds.pkl"

FIELD = {"belief_bias": "BELIEF_BIAS", "exaggeration_factor": "EXAGGERATION_FACTOR",
         "xi": "XI", "tau": "TAU", "gamma": "GAMMA",
         "subjective_cost": "SUBJECTIVE_COST", "patience": "PATIENCE",
         "is_hazardous": "IS_HAZARDOUS", "c_max": "C_MAX",
         "hazard_lapse": "HAZARD_LAPSE", "urgency_coefficient": "URGENCY_COEFFICIENT",
         "urgency_slope": "URGENCY_SLOPE"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def de_results(task):
    for p in (f"{ROOT}/data/{SUBDIR}/{task}/de/results.pkl",
              f"{ROOT}/data/{SUBDIR}/{task}/de/short/results.pkl",
              f"{ROOT}/data/{SUBDIR}/{task}/de/long/results.pkl"):
        if os.path.exists(p):
            return pd.read_pickle(p)
    return None


def nests(A, B):
    """True if A's search space contains B's configuration."""
    if A.TASK[0] != B.TASK[0]:
        return False
    fa, fb = set(A.PARAM_RANGES), set(B.PARAM_RANGES)
    if not fb < fa:
        return False
    for k, fld in FIELD.items():
        va, vb = getattr(A, fld), getattr(B, fld)
        if k in fa and k in fb:
            la, ha = A.PARAM_RANGES[k]
            lb, hb = B.PARAM_RANGES[k]
            if not (la <= lb and hb <= ha):
                return False
        elif k in fa:
            lo, hi = A.PARAM_RANGES[k]
            if not (lo <= float(vb) <= hi):
                return False
        elif k not in fb:
            if float(va) != float(vb):
                return False
        else:
            return False
    return True


cfg, fits = {}, {}
for f in sorted(glob.glob(f"{ROOT}/data/simulation_configs/simulation_params_*.py")):
    if f.endswith("_commit.py") != COMMIT:
        continue
    c = load_config(f)
    cfg[c.TASK] = c
    d = de_results(c.TASK)
    if d is not None:
        fits[c.TASK] = d

seeds, stats = {}, []
for a in sorted(cfg):
    A = cfg[a]
    neighbours = [b for b in fits if b != a and nests(A, cfg[b])]
    if not neighbours:
        stats.append((a, 0, 0))
        continue
    keys = list(A.PARAM_RANGES.keys())
    per_subject = {}
    for b in neighbours:
        B, d = cfg[b], fits[b]
        ll = dict(zip(d["subject_ID"].astype(str), np.asarray(d["after_lls_ga"], dtype=float)))
        pv = {str(r.subject_ID): dict(zip(B.PARAM_ORDER, r.fit_params_ga)) for r in d.itertuples()}
        for sid, val in ll.items():
            if sid not in per_subject or val > per_subject[sid][0]:
                # parameters B does not fit are held at B's own fixed values
                vec = [float(pv[sid][k]) if k in pv[sid] else float(getattr(B, FIELD[k]))
                       for k in keys]
                vec = [min(max(v, A.PARAM_RANGES[k][0]), A.PARAM_RANGES[k][1])
                       for k, v in zip(keys, vec)]
                per_subject[sid] = (val, b, vec)
    seeds[a] = {sid: v[2] for sid, v in per_subject.items()}
    stats.append((a, len(neighbours), len(per_subject)))

out = f"{ROOT}/BIC/{OUT_NAME}"
with open(out, "wb") as fh:
    pickle.dump(seeds, fh)
print(f"models: {len(cfg)}   with DE fits: {len(fits)}   seeded: {len(seeds)}")
print(f"{'model':16}{'neighbours':>11}{'subjects':>10}")
for t, n, s in stats:
    if n:
        print(f"  {t:16}{n:>9}{s:>10}")
print(f"\nwrote {out}")
