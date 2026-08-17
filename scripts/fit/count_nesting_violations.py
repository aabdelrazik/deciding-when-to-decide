"""Count nesting violations across every checkable model pair.

If model A's search space contains model B's configuration, a converged fit
satisfies logL(A) >= logL(B) for every subject. A violation therefore proves at
least one of the two fits is not at its optimum. POMDP_SEEDS_COMMIT=1 checks the
commit-likelihood family instead.
"""
import glob, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd
from src.config.loader import load_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_de_seeds import nests, de_results, COMMIT, SUBDIR

cfg, fits = {}, {}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for f in sorted(glob.glob(f"{ROOT}/data/simulation_configs/simulation_params_*.py")):
    if f.endswith("_commit.py") != COMMIT:
        continue
    c = load_config(f)
    cfg[c.TASK] = c
    d = de_results(c.TASK)
    if d is not None:
        fits[c.TASK] = d

pairs = [(a, b) for a in fits for b in fits if a != b and nests(cfg[a], cfg[b])]
bad, worst = [], []
for a, b in pairs:
    la = fits[a].set_index("subject_ID")["after_lls_ga"].astype(float)
    lb = fits[b].set_index("subject_ID")["after_lls_ga"].astype(float)
    idx = la.index.intersection(lb.index)
    d = la[idx] - lb[idx]
    n = int((d < -1e-6).sum())
    if n:
        bad.append((a, b, n, float(d.min())))

print(f"family={'commit' if COMMIT else 'full fit'}   models with fits={len(fits)}   checkable pairs={len(pairs)}")
print(f"pairs containing at least one violated subject: {len(bad)}")
for a, b, n, w in sorted(bad, key=lambda r: r[3])[:15]:
    print(f"   {a} > {b}: {n} subject(s), worst {w:+.3f} logL")
