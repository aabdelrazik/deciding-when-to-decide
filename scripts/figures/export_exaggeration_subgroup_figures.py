"""Per-subject ensemble fit for the subjects with the highest fitted exaggeration factor.

Produces the S1 Fig panels. The subjects are found from the current fits rather
than named in the script: the previous version of this figure had been generated
by hand and still showed the two highest-E short-horizon subjects, which the
best-supported short-horizon model cannot have because it does not fit an
exaggeration factor at all.

Only horizons whose winning model actually fits exaggeration_factor get panels,
so if a winner changes to one without the mechanism its panels disappear rather
than silently describing a model that never had it.

Usage:
    SIM_ALGORITHM=de python3 export_exaggeration_subgroup_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(R)
sys.path.append(os.path.join(R, "scripts"))

from src.config.loader import load_config
from src.utils.plotting import plot_all_subjects_ensemble
# export_exaggeration_subgroup_figures imports a sibling that lives in
# scripts/recovery/, which is not on the path when this runs
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recovery"))
from recovery_post_analysis import build_ensemble_data

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ALGO = os.environ.get("SIM_ALGORITHM", "de")
HORIZON_NAME = {"short": "short horizon", "long": "long horizon",
                "combined": "combined horizons"}
OUT = os.path.join(R, "figures", "POMDP", "exaggeration_subgroups")
N_TOP = int(os.environ.get("EXAG_N_TOP", "2"))


def main():
    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        best = json.load(fh)

    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for horizon in ("short", "long", "combined"):
        task = best[horizon]
        cfg_path = os.path.join(R, "data/simulation_configs", f"simulation_params_{task}.py")
        cfg = load_config(cfg_path)
        if "exaggeration_factor" not in cfg.PARAM_RANGES:
            print(f"{horizon}: {task} does not fit an exaggeration factor; no panels")
            continue

        i = list(cfg.PARAM_RANGES).index("exaggeration_factor")
        res = pd.read_pickle(os.path.join(cfg.DATA_PATH, "results.pkl"))
        res["E"] = [r[i] for r in res["fit_params_ga"]]
        top = res.nlargest(N_TOP, "E")

        human = pd.read_pickle(cfg.HUMAN_DATA_PATH)
        print(f"{horizon}: {task}, top {N_TOP} by E")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            sid = int(row.subject_ID)
            _, ens = build_ensemble_data(cfg, res[res.subject_ID == row.subject_ID],
                                         human, [row.subject_ID])
            if not ens:
                print(f"    subj {sid}: no ensemble data; skipped")
                continue
            # build_ensemble_data reuses the cached whole-sample ensemble and ignores
            # the subject list, so restrict to this subject before plotting or the
            # panel shows all 105 while claiming to show one
            k = next((kk for kk in ens if str(kk) == str(sid)), None)
            if k is None:
                print(f"    subject {sid} absent from the ensemble; skipped")
                continue
            ens = {k: ens[k]}
            plot_all_subjects_ensemble(
                ens, horizon=f"{horizon}_subj{sid}", path=OUT,
                title_label=f"Subject {sid}, {HORIZON_NAME[horizon]}")
            print(f"    subj {sid:>3}  E={row.E:.3f}  (rank {rank}) -> "
                  f"ensemble_all_subjects_{horizon}_subj{sid}.pdf")
            manifest.append(dict(horizon=horizon, task=task, subject=sid,
                                 E=float(row.E), rank=rank))

    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nwrote {OUT}/manifest.json ({len(manifest)} panels)")


if __name__ == "__main__":
    main()
