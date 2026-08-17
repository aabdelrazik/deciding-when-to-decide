"""Human vs model-simulated behaviour for one subject, under all three winners.

Same convention as the exaggeration-subgroup panels: for the named subject, plot
the human draw-count and outcome distributions against the model-simulated
ensemble, once per horizon condition using that horizon's winning model.

Written for the worst-fitting subject, to show whether a low draws-R^2 reflects a
failure of the model or a participant whose own behaviour is close to
unstructured. Set SUBJECT to look at any other.

Usage (from scripts/):
    SIM_ALGORITHM=de SUBJECT=17 python3 export_subject_fit_panel.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

# Repository root. Resolved from this file so the scripts run from any
# working directory and outside a container; override with POMDP_ROOT.
R = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(R)
sys.path.append(os.path.join(R, "scripts"))

from src.config.loader import load_config
from src.utils.plotting import plot_all_subjects_ensemble
# export_subject_fit_panel imports a sibling that lives in
# scripts/recovery/, which is not on the path when this runs
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recovery"))
from recovery_post_analysis import build_ensemble_data

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ALGO = os.environ.get("SIM_ALGORITHM", "de")
SUBJECT = int(os.environ.get("SUBJECT", "17"))
HORIZON_NAME = {"short": "short horizon", "long": "long horizon",
                "combined": "combined horizons"}
OUT = os.path.join(R, "figures", "POMDP", "subject_fit_panel")


def main():
    with open(os.path.join(R, "BIC", "best_models.json")) as fh:
        best = json.load(fh)

    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for horizon in ("short", "long", "combined"):
        task = best[horizon]
        cfg = load_config(os.path.join(R, "data/simulation_configs",
                                       f"simulation_params_{task}.py"))
        res = pd.read_pickle(os.path.join(cfg.DATA_PATH, "results.pkl"))
        row = res[res.subject_ID.astype(int) == SUBJECT]
        if row.empty:
            print(f"  {horizon}: subject {SUBJECT} not in {task}; skipped")
            continue

        human = pd.read_pickle(cfg.HUMAN_DATA_PATH)
        metrics, ens = build_ensemble_data(cfg, row, human, [row.subject_ID.iloc[0]])
        if not ens:
            print(f"  {horizon}: no ensemble data for subject {SUBJECT}; skipped")
            continue

        r2 = None
        if metrics is not None and len(metrics):
            m = metrics[metrics.iloc[:, 0].astype(str) == str(row.subject_ID.iloc[0])]
            if len(m) and "r2_draws" in m:
                r2 = float(m.r2_draws.iloc[0])

        # build_ensemble_data reuses the cached whole-sample ensemble and ignores
        # the subject list, so restrict to this subject before plotting or the
        # panel shows all 105 while claiming to show one
        k = next((kk for kk in ens if str(kk) == str(SUBJECT)), None)
        if k is None:
            print(f"    subject {SUBJECT} absent from the ensemble; skipped")
            continue
        ens = {k: ens[k]}
        plot_all_subjects_ensemble(
                ens, horizon=f"{horizon}_subj{SUBJECT}", path=OUT,
                title_label=f"Subject {SUBJECT}, {HORIZON_NAME[horizon]}")
        print(f"  {horizon:9} {task:14} subject {SUBJECT}"
              + (f"  draws R2={r2:.3f}" if r2 is not None else "")
              + f"  -> ensemble_all_subjects_{horizon}_subj{SUBJECT}.pdf")
        manifest.append(dict(horizon=horizon, task=task, subject=SUBJECT, r2_draws=r2))

    with open(os.path.join(OUT, f"manifest_subj{SUBJECT}.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n  wrote {OUT}/manifest_subj{SUBJECT}.json ({len(manifest)} panels)")


if __name__ == "__main__":
    main()
