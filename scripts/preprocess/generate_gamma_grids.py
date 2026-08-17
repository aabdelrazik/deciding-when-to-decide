"""Regenerate the per-gamma discounted evidence grids.

Each gamma is discounted from the RAW sequences. The notebook version mutated
DataFrames shared with the source (copy.deepcopy on a DataFrame does not copy
DataFrames held in object cells), so every gamma after the first was discounted
on top of the previous one's output. Only the first grid point was ever correct.
"""
import os, sys

_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import pandas as pd, numpy as np
from src.config import *
# generate_gamma_grids imports a sibling that lives in
# scripts/fit/, which is not on the path when this runs
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fit"))
from fit_data_forgetting import apply_discounting

ROOT = os.path.join(_ROOT, "data/TrHu_NHB_light/data_MEG")
COL = "draw_yellow_blue_action_outcome"
horizons = [f"_{FIT_HORIZON[0]}"] if len(FIT_HORIZON) == 1 else [""]

for horizon in horizons:
    for stem in (f"all_subject_evidence_dicts{horizon}",
                 f"all_subject_evidence_dicts_full_sequence{horizon}"):
        src = f"{ROOT}/{stem}.pkl"
        base = pd.read_pickle(src)
        print(f"\n{stem}: {base.shape}", flush=True)
        for gamma in gamma_values:
            out = {}
            for col in base.columns:
                for idx in base.index:
                    cell = base.loc[idx, col]
                    if isinstance(cell, pd.DataFrame) and COL in cell.columns:
                        # build a genuinely independent frame from the RAW cell
                        newdf = cell.copy(deep=True)
                        newdf[COL] = cell[COL].apply(
                            lambda x: apply_discounting(x, gamma=gamma))
                    else:
                        newdf = cell
                    out.setdefault(idx, {})[col] = newdf
            pd.DataFrame(out).T.to_pickle(f"{ROOT}/{stem}_{gamma}.pkl")
            print(f"  gamma={gamma} written", flush=True)
        # the _combined file stacks every gamma per subject; rebuild for the new grid
        per = {g: pd.read_pickle(f"{ROOT}/{stem}_{g}.pkl") for g in gamma_values}
        combined = {sid: {g: per[g].loc[sid] for g in gamma_values}
                    for sid in per[gamma_values[0]].index}
        pd.DataFrame(combined).T.to_pickle(f"{ROOT}/{stem}_combined.pkl")
        print(f"  rebuilt {stem}_combined.pkl over {len(gamma_values)} gammas", flush=True)
print("\nGRIDS DONE")
