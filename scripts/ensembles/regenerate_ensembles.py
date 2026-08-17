"""Re-simulate the ensemble (raw_simulations/) from whatever parameters are
currently stored in results.pkl, without refitting.

Needed after merging two optimizer runs: the ensembles on disk were simulated
during the run that produced them, so for every subject whose merged fit came
from the other run they no longer correspond to the stored parameters. The GLM
and GLMM scripts read both files, so they would otherwise describe a fit that
was rejected. Running fit_data.py again is not an option, since it refits and
would overwrite the merged results.
"""
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from src.config import *  # noqa: F401,F403  (TASK, DATA_PATH, FIT_HORIZON, ...)
from src.params_fitting.data_simulation import simulate_data

NUM_RUNS = int(os.environ.get("ENSEMBLE_RUNS", "300"))
N_JOBS = int(os.environ.get("SIM_N_JOBS", "-1"))
horizon_str = FIT_HORIZON[0] if len(FIT_HORIZON) == 1 else "combined"
raw_sim_dir = os.path.join(DATA_PATH, "raw_simulations")


def one_run(run_idx, results_df):
    """Simulate one ensemble replicate and write it where the GLM scripts look."""
    simulated, _ = simulate_data(results_df, sim_same_data=True)
    simulated.to_pickle(os.path.join(raw_sim_dir, f"sim_run_{run_idx}_{horizon_str}.pkl"))
    return run_idx


if __name__ == "__main__":
    results_df = pd.read_pickle(RESULTS_PATH)
    os.makedirs(raw_sim_dir, exist_ok=True)
    stale = glob.glob(os.path.join(raw_sim_dir, f"sim_run_*_{horizon_str}.pkl"))
    for f in stale:
        os.remove(f)
    print(f"TASK={TASK} horizon={horizon_str} subjects={len(results_df)}")
    print(f"removed {len(stale)} stale replicates; simulating {NUM_RUNS} fresh ones")
    with tqdm_joblib(tqdm(desc="runs", total=NUM_RUNS)):
        Parallel(n_jobs=N_JOBS)(delayed(one_run)(i, results_df) for i in range(NUM_RUNS))
    n = len(glob.glob(os.path.join(raw_sim_dir, f"sim_run_*_{horizon_str}.pkl")))
    print(f"ENSEMBLE DONE: {n} replicates in {raw_sim_dir}")
