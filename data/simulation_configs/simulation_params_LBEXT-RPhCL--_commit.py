# Commit-only variant of simulation_params_LBEXT-RPhCL--.py
# Fits using log_likelihood_commit (decide vs wait) instead of the full
# 3-way action LL. Results are saved under data/POMDP_commit/ so they
# never overwrite the original data/POMDP/ outputs.
import importlib.util as _ilu, os as _os

def _load(path):
    spec = _ilu.spec_from_file_location("_orig", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OVERRIDES

_here = _os.path.dirname(_os.path.abspath(__file__))
OVERRIDES = {**_load(_os.path.join(_here, "simulation_params_LBEXT-RPhCL--.py")), "POMDP_COMMIT": True}
