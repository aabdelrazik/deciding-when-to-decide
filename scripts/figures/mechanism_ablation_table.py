"""Controlled exaggeration-by-temporal-regulation comparison, one quartet per horizon.

Each quartet is anchored on that horizon's winning model and varies only the two
mechanisms: the exaggeration factor E, and the temporal regulation function
Phi(t) (switched off by setting c_max, urgency_coefficient and urgency_slope to
zero, which makes Phi(t) = 0 at every draw). Every other free parameter is free
in all four cells, and the hazard state is identical across them, so a dBIC
between cells is attributable to the mechanism alone.

BIC uses the per-subject-summed convention used throughout: sum_i k*log(n_obs_i)
- 2*ll_i, with each subject's own observation count.
"""
import os, sys, json
import numpy as np, pandas as pd

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
project_root = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(project_root)
from src.utils.plotting import compute_metrics

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

CONFIG_DIR = os.path.join(project_root, "data/simulation_configs")

QUARTETS = {
    "Short": [
        ("Exaggeration + temporal regulation", "SBEXT-RPh----"),
        ("Temporal regulation only",           "SB-XT-RPh----"),
        ("Exaggeration only",                  "SBEXT-R-h----"),
        ("Neither",                            "SB-XT-R-h----"),
    ],
    "Long": [
        ("Exaggeration + temporal regulation", "LBE-T-RPhCL--"),
        ("Temporal regulation only",           "LB--T-RPhCL--"),
        ("Exaggeration only",                  "LBE-T-R-h-L--"),
        ("Neither",                            "LB--T-R-h-L--"),
    ],
    # Anchored on the combined winner C-EXT-RPHC-UK. Its background (xi, tau,
    # subjective_cost, is_hazardous) is free in all four cells, so as in the
    # other two quartets a dBIC between cells isolates the mechanism.
    "Combined": [
        ("Exaggeration + temporal regulation", "C-EXT-RPHC-UK"),
        ("Temporal regulation only",           "C--XT-RPHC-UK"),
        ("Exaggeration only",                  "C-EXT-R-H----"),
        ("Neither",                            "C--XT-R-H----"),
    ],
}


def load_overrides(task):
    ns = {}
    exec(open(os.path.join(CONFIG_DIR, f"simulation_params_{task}.py")).read(), ns)
    return ns["OVERRIDES"]


def summed_bic(task, overrides):
    fh = overrides.get("FIT_HORIZON", ["short"])
    parts = [fh[0]] if len(fh) == 1 else []
    path = os.path.join(project_root, "data", "POMDP", task, "de", *parts, "results.pkl")
    if not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    k = len(overrides["PARAM_RANGES"])
    tot, n = 0.0, 0
    for _, row in df.iterrows():
        dd = row["data_dict_of_lists"]
        hs = dd.values() if isinstance(dd, dict) else dd
        n_obs = sum(len(s) for h in hs for s in h["draw_yellow_blue_action_outcome"].values)
        if n_obs == 0:
            continue
        tot += compute_metrics(row["after_lls_ga"], k, n_obs)["BIC"]
        n += 1
    return dict(BIC=tot, k=k, n=n)


rows, missing = [], []
for horizon, cells in QUARTETS.items():
    got = []
    for label, task in cells:
        o = load_overrides(task)
        m = summed_bic(task, o)
        if m is None:
            missing.append(task)
            continue
        got.append(dict(horizon=horizon, label=label, task=task, **m))
    if not got:
        continue
    # the both-mechanisms cell is the reference for that horizon
    ref = next(r for r in got if r["label"].startswith("Exaggeration + "))
    for r in got:
        r["dBIC"] = r["BIC"] - ref["BIC"]
    rows.extend(sorted(got, key=lambda r: r["dBIC"]))

out = pd.DataFrame(rows)
if len(out):
    print(out.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
if missing:
    print("\nnot yet fitted: " + ", ".join(missing))
out.to_json(os.path.join(project_root, "BIC", "mechanism_ablation.json"),
            orient="records", indent=1)

# Emit the table body too, so the rows always describe the fits on disk rather
# than a copy maintained separately in the .tex.
if len(out):
    tex = [r"\begin{tabular}{llccr}", r"\hline",
           r"\textbf{Horizon} & \textbf{Mechanisms present} & \textbf{Model} & "
           r"\textbf{$N_{\text{params}}$} & \textbf{$\Delta$BIC} \\", r"\hline"]
    for horizon in QUARTETS:
        blk = out[out.horizon == horizon]
        if blk.empty:
            continue
        # report dBIC against the best cell of the quartet, not the anchor cell,
        # so the winning row always reads 0.0
        base = blk.BIC.min()
        for _, r in blk.sort_values("BIC").iterrows():
            d = r.BIC - base
            val = r"\textbf{0.0}" if d < 1e-9 else f"{d:.1f}"
            tex.append(f"{horizon} & {r.label} & \\texttt{{{r.task}}} & {r.k} & {val} \\\\")
        tex.append(r"\hline")
    tex += [r"\end{tabular}", ""]
    p = os.path.join(project_root, "BIC", "tables", "mechanism_ablation_table.tex")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("\n".join(tex))
    print(f"\nwrote {p}")
