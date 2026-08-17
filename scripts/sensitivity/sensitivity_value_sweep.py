"""Value-sweep sensitivity analysis for the fixed temporal-regulation parameters.

For a winning model, holds every subject's fitted parameters constant and
evaluates the population-summed log-likelihood over a grid of candidate values
for each regulation parameter that the model fixes rather than fits. This says
how much fit quality the fixed choice leaves on the table.

Everything is derived from BIC/best_models.json and the model's own config: the
model family, the parameter order used to unpack fit_params_ga, and crucially
*which* parameters are fixed. The superseded version of this script hardcoded
CB--TGRPhCL--, its parameter order, and the ga fits, so it silently described a
model that is no longer the combined-horizon winner. A parameter the model fits
per subject is skipped, since there is no fixed value to be sensitive about.

Usage (one horizon per invocation, from scripts/):
    SIM_ALGORITHM=de SWEEP_HORIZON=short python3 sensitivity_value_sweep.py
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.config.loader import load_config

ALGO = os.environ.get("SIM_ALGORITHM", "de")
HORIZON = os.environ.get("SWEEP_HORIZON", "short")

# grids at least as wide as the range used when the parameter was fitted freely
GRIDS = {
    "c_max": np.linspace(0, 80, 41),
    "urgency_coefficient": np.linspace(-30, 0, 61),
    "urgency_slope": np.linspace(-20, 0, 41),
}
DEFAULTS = {"c_max": 50, "urgency_coefficient": -10, "urgency_slope": -2}

with open(os.path.join(project_root, "BIC", "best_models.json")) as fh:
    TASK = json.load(fh)[HORIZON]
CFG_PATH = os.path.join(project_root, "data/simulation_configs",
                        f"simulation_params_{TASK}.py")
CFG = load_config(CFG_PATH)
ORDER = list(CFG.PARAM_RANGES)
HORIZONS = list(CFG.FIT_HORIZON)
# the config names the class; inferring it from the parameter list gets it wrong
# (a model with patience is an urgency model, not vanilla)
FAMILY = CFG.POMDP_TYPE

_ns = {"__file__": CFG_PATH}
exec(open(CFG_PATH).read(), _ns)
OVR = _ns["OVERRIDES"]

# fixed here means: not fitted per subject, so a single value was chosen for it
FIXED = {k: OVR.get(k.upper(), DEFAULTS[k]) for k in GRIDS if k not in ORDER}
OUTPUT_PATH = os.path.join(project_root, "BIC", "tables",
                           f"sensitivity_{HORIZON}_sweep.npz")


def subject_ll(args):
    row, over = args
    import inspect
    from src.pomdp import POMDPFactory

    p = row["fit_params_ga"]
    base = {name: p[i] for i, name in enumerate(ORDER)}
    for k, v in FIXED.items():
        base.setdefault(k, v)
    base.update(over)
    base.setdefault("is_hazardous", bool(OVR.get("IS_HAZARDOUS", True)))
    base.setdefault("gamma", OVR.get("GAMMA", 1))
    base.update(verbose=False, max_cards_per_draw=5)

    total = 0.0
    for hz in HORIZONS:
        params = dict(base)
        params["horizon_condition"] = hz
        pomdp = POMDPFactory(FAMILY)
        # each family accepts a different subset of these; passing one it does
        # not take raises rather than being ignored
        ok = set(inspect.signature(type(pomdp).__init__).parameters)
        pomdp.__init__(**{k: v for k, v in params.items() if k in ok})
        pomdp.value_iteration()
        d = row["data_dict_of_lists"]
        total += pomdp.log_likelihood(d[hz] if isinstance(d, dict) and hz in d else d)
    return total


def main():
    if not FIXED:
        print(f"{HORIZON}: {TASK} fits every temporal-regulation parameter per subject "
              f"({', '.join(GRIDS)}); nothing is fixed, so no sweep applies.")
        np.savez(OUTPUT_PATH, task=TASK, nothing_fixed=True)
        return

    sub = HORIZONS[0] if len(HORIZONS) == 1 else None
    rp = os.path.join(project_root, "data/POMDP", TASK, ALGO,
                      *([sub] if sub else []), "results.pkl")
    rows = pd.read_pickle(rp).to_dict("records")
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))
    print(f"{HORIZON}: {TASK} ({FAMILY}), fixed = {FIXED}", flush=True)
    print(f"  {len(rows)} subjects, {n_workers} workers, horizons {HORIZONS}", flush=True)

    out = {}
    with Pool(n_workers) as pool:
        for key, chosen in FIXED.items():
            t0, grid, lls = time.time(), GRIDS[key], []
            for v in grid:
                over = dict(FIXED); over[key] = v
                lls.append(sum(pool.map(subject_ll, [(r, over) for r in rows])))
                print(f"  {key} v={v:.2f} ll={lls[-1]:.2f} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
            lls = np.array(lls)
            cur_ll = sum(pool.map(subject_ll, [(r, dict(FIXED)) for r in rows]))
            b = int(np.argmax(lls))
            print(f"{key}: chosen={chosen} cur_ll={cur_ll:.2f} | optimum={grid[b]:.2f} "
                  f"ll={lls[b]:.2f} | dLL={lls[b]-cur_ll:.2f} | {time.time()-t0:.1f}s",
                  flush=True)
            out[key] = dict(grid=grid, lls=lls, cur_ll=cur_ll, chosen=chosen)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, task=TASK, horizon=HORIZON,
             **{f"{k}_grid": v["grid"] for k, v in out.items()},
             **{f"{k}_lls": v["lls"] for k, v in out.items()},
             **{f"{k}_cur_ll": v["cur_ll"] for k, v in out.items()},
             **{f"{k}_chosen": v["chosen"] for k, v in out.items()})
    print(f"Saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
