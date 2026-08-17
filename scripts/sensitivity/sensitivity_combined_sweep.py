"""Value-sweep sensitivity analysis for the combined-horizon winning model's
fixed temporal-regulation parameters (urgency_coefficient / phi_min and
urgency_slope / k; c_max / phi_max is fitted per subject for this model, so
it is not swept).

For each swept parameter, holds every subject's other already-fitted
parameters constant and evaluates the population-summed log-likelihood
(short + long, matching how this model was originally fit) at a grid of
candidate values for that one parameter, to check whether the value it was
actually fixed to (the schema default, -10 and -2 respectively) is close to
the value that would maximize log-likelihood given everyone's other
parameters. No refitting/optimization happens here -- this is a pure
forward evaluation (construct POMDP -> value_iteration -> log_likelihood)
at each candidate value, parallelized across subjects with multiprocessing.

Usage (submitted via slurm/run_sensitivity_combined_sweep.sh):
    python scripts/sensitivity/sensitivity_combined_sweep.py
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

# The combined winner and the optimizer both come from the run rather than
# from a literal, so this follows a refit instead of reading a stale tree.
ALGORITHM = os.environ.get("SIM_ALGORITHM", "de")
with open(os.path.join(project_root, "BIC", "best_models.json")) as _fh:
    COMBINED_TASK = os.environ.get("SWEEP_TASK") or json.load(_fh)["combined"]
RESULTS_PATH = os.path.join(project_root, "data/POMDP", COMBINED_TASK,
                            ALGORITHM, "results.pkl")
OUTPUT_PATH = os.path.join(project_root, "BIC/tables/sensitivity_combined_sweep.npz")

CURRENT = {"urgency_coefficient": -10, "urgency_slope": -2}
SWEEPS = {
    "urgency_coefficient": np.linspace(-30, 0, 61),
    "urgency_slope": np.linspace(-20, 0, 41),
}


def subject_ll(args):
    row, uc, us = args
    from src.pomdp import POMDPFactory

    p = row["fit_params_ga"]
    base = dict(
        tau=p[0],
        gamma=p[1],
        subjective_cost=p[2],
        patience=p[3],
        c_max=p[4],
        hazard_lapse=p[5],
        belief_bias=p[6],
        urgency_coefficient=uc,
        urgency_slope=us,
        is_hazardous=True,
        verbose=False,
        max_cards_per_draw=5,
    )
    total = 0.0
    for hz in ["short", "long"]:
        params = dict(base)
        params["horizon_condition"] = hz
        pomdp = POMDPFactory("forgetting")
        pomdp.__init__(**params)
        pomdp.value_iteration()
        total += pomdp.log_likelihood(row["data_dict_of_lists"][hz])
    return total


def main():
    df = pd.read_pickle(RESULTS_PATH)
    rows = df.to_dict("records")
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))
    print(f"n subjects: {len(rows)}, n_workers: {n_workers}", flush=True)

    out = {}
    with Pool(n_workers) as pool:
        for key, grid in SWEEPS.items():
            t0 = time.time()
            lls = []
            for v in grid:
                uc = v if key == "urgency_coefficient" else CURRENT["urgency_coefficient"]
                us = v if key == "urgency_slope" else CURRENT["urgency_slope"]
                tasks = [(row, uc, us) for row in rows]
                total = sum(pool.map(subject_ll, tasks))
                lls.append(total)
                print(f"  {key} v={v:.2f} ll={total:.2f} elapsed={time.time()-t0:.1f}s", flush=True)
            lls = np.array(lls)
            tasks = [
                (row, CURRENT["urgency_coefficient"], CURRENT["urgency_slope"])
                for row in rows
            ]
            cur_ll = sum(pool.map(subject_ll, tasks))
            best_idx = np.argmax(lls)
            print(
                f"{key}: current={CURRENT[key]} cur_ll={cur_ll:.2f} | "
                f"optimum={grid[best_idx]:.2f} ll={lls[best_idx]:.2f} | "
                f"dLL={lls[best_idx]-cur_ll:.2f} | total {time.time()-t0:.1f}s",
                flush=True,
            )
            out[key] = dict(grid=grid, lls=lls, cur_ll=cur_ll)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        **{f"{k}_grid": v["grid"] for k, v in out.items()},
        **{f"{k}_lls": v["lls"] for k, v in out.items()},
        **{f"{k}_cur_ll": v["cur_ll"] for k, v in out.items()},
    )
    print(f"Saved to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
