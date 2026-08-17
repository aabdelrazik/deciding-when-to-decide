"""Write the fitted parameters out as CSV, for deposit alongside the paper.

results.pkl is a pandas pickle, which is fragile across pandas versions and
unreadable outside Python. Every number in the manuscript is a function of the
per subject parameter estimates, and refitting them costs hundreds of node
hours, so they are deposited as plain CSV instead.

One file per model and horizon, with a column per free parameter plus the
log likelihood and the number of observations the BIC uses.

Usage (from the repository root):
    python3 scripts/optimality/export_fitted_parameters_csv.py
    python3 scripts/optimality/export_fitted_parameters_csv.py --tasks SB-XT-RPh---- LBE-T-RPhCL--
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from src.config.loader import load_config  # noqa: E402
from src.utils.plotting import _per_subject_n_obs  # noqa: E402

OUT = os.path.join(ROOT, "results", "fits")


def winners():
    p = os.path.join(ROOT, "BIC", "best_models.json")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(json.load(fh).values())


def export(task):
    cfg_path = os.path.join(ROOT, "data", "simulation_configs",
                            f"simulation_params_{task}.py")
    if not os.path.exists(cfg_path):
        print(f"  no config for {task}, skipped")
        return None
    cfg = load_config(cfg_path)
    results_path = cfg.RESULTS_PATH
    if not os.path.exists(results_path):
        print(f"  no fit for {task} at {results_path}, skipped")
        return None

    df = pd.read_pickle(results_path)
    params = np.vstack(df["fit_params_ga"].values)
    out = pd.DataFrame(params, columns=list(cfg.PARAM_ORDER))
    out.insert(0, "subject_ID", df["subject_ID"].values)
    out["log_likelihood"] = df["after_lls_ga"].values
    out["n_obs"] = [_per_subject_n_obs(d) for d in df["data_dict_of_lists"]]

    horizon = "-".join(cfg.FIT_HORIZON)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{task}_{horizon}.csv")
    out.to_csv(dest, index=False)
    print(f"  {task:16s} {len(out):3d} subjects, {len(cfg.PARAM_ORDER)} parameters "
          f"-> {os.path.relpath(dest, ROOT)}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="TASK names; defaults to the winners in BIC/best_models.json")
    args = ap.parse_args()

    tasks = args.tasks if args.tasks else winners()
    if not tasks:
        sys.exit("no tasks given and BIC/best_models.json not found")

    print(f"exporting {len(tasks)} model(s)")
    written = [export(t) for t in tasks]
    print(f"wrote {len([w for w in written if w])} file(s) to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
