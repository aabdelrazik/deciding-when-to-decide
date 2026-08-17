import os
"""Candidate-model table: one row per model, one column per parameter.

Emits a longtable so it breaks across pages instead of overflowing, and reads
every value from the configs themselves so ranges and fixed values cannot drift
out of step with what was actually fitted.
"""
import glob, json, os, sys

sys.path.append(os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.config.loader import load_config

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
COMMIT = os.environ.get("PARAMS_TABLE_COMMIT", "") == "1"
OUT = os.path.join(R, "BIC_commit" if COMMIT else "BIC", "tables",
                   "simulation_params_table_full.tex")

# column -> (config key, schema default when the parameter is not fitted)
COLS = [(r"$\beta$", "belief_bias", 1), (r"$E$", "exaggeration_factor", 1),
        (r"$\xi$", "xi", 0), (r"$\tau$", "tau", 0.001), (r"$\gamma$", "gamma", 1),
        (r"$R_{\text{risk}}$", "subjective_cost", 0), (r"$t_p$", "patience", 0),
        (r"$H$", "is_hazardous", None), (r"$\phi_{\max}$", "c_max", 0.2),
        (r"$L$", "hazard_lapse", 0), (r"$\phi_{\min}$", "urgency_coefficient", -10),
        (r"$k$", "urgency_slope", -2)]

def fmt(v):
    if isinstance(v, bool): return r"\textrm{True}" if v else r"\textrm{False}"
    if isinstance(v, tuple): return f"$({v[0]:g}, {v[1]:g})$"
    if isinstance(v, float) and v == int(v): return f"${int(v):g}$"
    return f"${v:g}$" if isinstance(v, (int, float)) else str(v)

rows = []
for f in sorted(glob.glob(f"{R}/data/simulation_configs/simulation_params_*.py")):
    base = os.path.basename(f)[len("simulation_params_"):-3]
    if base.endswith("_commit") != COMMIT: continue
    task = base[:-7] if COMMIT else base
    ns = {"__file__": f}; exec(open(f).read(), ns); o = ns["OVERRIDES"]
    # Resolve through the loader: the schema rewrites some ranges (patience is
    # (0,8) for short-only fits, (0,14) otherwise), so the raw OVERRIDES do not
    # always show the range the model was actually fitted over.
    cfg = load_config(f)
    pr = cfg.PARAM_RANGES
    cells = []
    for _, key, default in COLS:
        if key in pr:
            cells.append(fmt(tuple(pr[key])))
        elif key == "is_hazardous":
            cells.append(fmt(bool(o.get("IS_HAZARDOUS", True))))
        else:
            cells.append(fmt(o.get(key.upper(), default)))
    rows.append((task, len(pr), cells))

rows.sort(key=lambda r: (r[0][0], r[0]))
hdr = " & ".join([r"\textbf{Model}", r"$N_{\text{params}}$"] + [c for c, _, _ in COLS])

def block(sel):
    body = "\n".join(f"\\texttt{{{t}}} & {k} & " + " & ".join(c) + r" \\" for t, k, c in sel)
    return (r"\resizebox{\textwidth}{!}{%" + "\n"
            r"\begin{tabular}{l c " + "c " * len(COLS) + "}\n"
            r"\toprule" + "\n" + hdr + r" \\" + "\n" + r"\midrule" + "\n"
            + body + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}}" + "\n")

# Split by horizon so each part fits a page and the set continues on the next.
short = [r for r in rows if r[0][0] == "S"]
rest  = [r for r in rows if r[0][0] != "S"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(block(short))
OUT2 = OUT.replace("_full.tex", "_full_part2.tex")
open(OUT2, "w").write(block(rest))
print(f"wrote {OUT}: {len(short)} short-horizon models")
print(f"wrote {OUT2}: {len(rest)} long/combined models")

# The three-winner table the main text cites. Winners come from best_models.json
# rather than a fixed list, so this cannot fall out of step with the comparison.
with open(os.path.join(R, "BIC_commit" if COMMIT else "BIC", "best_models.json")) as fh:
    best = json.load(fh)
wanted = [best[h] for h in ("short", "long", "combined")]
sel = [r for w in wanted for r in rows if r[0] == w]
missing = [w for w in wanted if not any(r[0] == w for r in rows)]
if missing:
    raise SystemExit(f"no config for winning model(s) {missing}; cannot build the best-model table")
OUT3 = OUT.replace("_full.tex", "_best.tex")
open(OUT3, "w").write(block(sel))
print(f"wrote {OUT3}: {', '.join(w for w in wanted)}")
