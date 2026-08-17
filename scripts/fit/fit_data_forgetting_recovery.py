"""Parameter recovery for a forgetting model, on top of an existing fit.

Every non-forgetting model gets its recovery pass inside the fitting run, so
results.pkl and results_recovered.pkl appear together. The forgetting models are
fitted through scripts/fit/fit_data_forgetting.py, which can do the same but
needs the gamma-keyed evidence rebuilt first, and that step was not run when
those models were originally fitted. This is the entry point for adding it
afterwards.

The recovery simulates each subject's data from their own fitted parameters and
refits it blind, which answers whether the optimiser can recover parameters it
generated itself. It reads the existing fit and never rewrites it: the original
results.pkl is loaded, not recomputed, so a backfilled recovery cannot perturb a
published fit.

Usage (from the repository root), one forgetting model at a time:

    SIM_CONFIG_PATH=$PWD/data/simulation_configs/simulation_params_LB--TGRPhCL--.py \\
    SIM_ALGORITHM=de python3 scripts/fit/fit_data_forgetting_recovery.py

Writes results_recovered.pkl and the recovered simulation frames beside the fit,
in data/POMDP/<TASK>/<algorithm>/[<horizon>/].

Requires the gamma grids, so run scripts/preprocess/generate_gamma_grids.py
first if they are not already built.
"""
import os
import runpy
import sys

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

if not os.environ.get("SIM_CONFIG_PATH"):
    sys.exit("SIM_CONFIG_PATH is not set: name the forgetting model to recover, "
             "for example data/simulation_configs/simulation_params_LB--TGRPhCL--.py")

# The recovery pass lives in the fitting script behind this flag, so there is one
# implementation rather than two that can drift apart. Setting it here makes the
# recovery a command of its own.
os.environ["RECOVERY_ONLY"] = "1"

print("recovery only: the existing fit is loaded, not refitted")
runpy.run_path(os.path.join(HERE, "fit_data_forgetting.py"), run_name="__main__")
