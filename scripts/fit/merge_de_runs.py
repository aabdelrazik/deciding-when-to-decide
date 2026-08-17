"""Merge the unseeded and seeded DE fits, keeping each subject's better fit.

Seeding guarantees a fit no worse than the model nested inside this one, which
is what removes nesting violations. It can also converge prematurely around the
seed, so it is not uniformly better than an unseeded search. Taking the best per
subject across both starts is a multi-start estimate: it keeps the seeded run's
guarantee while protecting against its failure mode, and can only improve on
either run alone.

Writes the merged fit to the live de/ path and preserves the seeded-only run.
"""
import os, shutil, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# POMDP_SEEDS_COMMIT=1 merges the commit-likelihood family instead, which lives
# under data/POMDP_commit/ and keeps its own pair of run backups.
COMMIT = os.environ.get("POMDP_SEEDS_COMMIT", "") == "1"
SUBDIR = "POMDP_commit" if COMMIT else "POMDP"
UNSEEDED = f"{ROOT}/BIC/_de_commit_unseeded_results" if COMMIT else f"{ROOT}/BIC/_de_unseeded_results"
SEEDED_BAK = f"{ROOT}/BIC/_de_commit_seeded_results" if COMMIT else f"{ROOT}/BIC/_de_seeded_results"

# optional horizon filter: `python merge_de_runs.py S L` merges only those
# prefixes, so a partially finished batch can be merged without touching models
# whose seeded fit is still being written.
WANT = set(sys.argv[1:]) or None

pairs, rows = [], []
for dirpath, _, files in os.walk(UNSEEDED):
    for fn in files:
        # only the fit to real data. results_recovered.pkl is a fit to data
        # each run simulated from its own parameters, so the two runs' recovery
        # files describe different datasets and taking the better logL across
        # them compares nothing meaningful.
        if fn != "results.pkl":
            continue
        old = os.path.join(dirpath, fn)
        rel = os.path.relpath(old, UNSEEDED)          # <TASK>/de/[horizon/]results.pkl
        live = os.path.join(ROOT, "data", SUBDIR, rel)
        if WANT and rel.split(os.sep)[0][0] not in WANT:
            continue
        if os.path.exists(live):
            pairs.append((rel, old, live))

print(f"result files with both runs: {len(pairs)}")
for rel, old, live in sorted(pairs):
    a = pd.read_pickle(old)                            # unseeded
    b = pd.read_pickle(live)                           # seeded
    # keep the seeded run before overwriting the live path
    bak = os.path.join(SEEDED_BAK, rel)
    os.makedirs(os.path.dirname(bak), exist_ok=True)
    if not os.path.exists(bak):
        shutil.copy2(live, bak)

    la = dict(zip(a["subject_ID"].astype(str), np.asarray(a["after_lls_ga"], dtype=float)))
    lb = dict(zip(b["subject_ID"].astype(str), np.asarray(b["after_lls_ga"], dtype=float)))
    keep, n_seed, n_unseed, n_tie, gain = [], 0, 0, 0, 0.0
    for i, sid in enumerate(b["subject_ID"].astype(str)):
        if sid not in la:
            keep.append(b.iloc[i]); n_seed += 1; continue
        if lb[sid] > la[sid] + 1e-9:
            keep.append(b.iloc[i]); n_seed += 1
        elif la[sid] > lb[sid] + 1e-9:
            j = list(a["subject_ID"].astype(str)).index(sid)
            keep.append(a.iloc[j]); n_unseed += 1; gain += la[sid] - lb[sid]
        else:
            keep.append(b.iloc[i]); n_tie += 1
    merged = pd.DataFrame(keep).reset_index(drop=True)
    merged.to_pickle(live)
    task = rel.split(os.sep)[0]
    rows.append((task, n_seed, n_unseed, n_tie, gain))

print(f"\n{'model':16}{'seeded won':>11}{'unseeded won':>14}{'tie':>6}{'logL rescued':>14}")
tot_s = tot_u = tot_t = 0; tot_g = 0.0
for t, s, u, ti, g in sorted(rows):
    tot_s += s; tot_u += u; tot_t += ti; tot_g += g
    if u:
        print(f"  {t:16}{s:>9}{u:>14}{ti:>6}{g:>14.1f}")
print(f"\n  TOTAL   seeded won {tot_s}, unseeded won {tot_u}, tied {tot_t}")
print(f"  logL rescued by keeping the unseeded run: {tot_g:.1f}")
print(f"  merged fits written to data/POMDP/<TASK>/de/ ; seeded-only preserved in {SEEDED_BAK}")
