import importlib.util
import os
from typing import Optional

from src.config.schema import SimulationConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Used when no override is named explicitly (e.g. interactive notebook use).
DEFAULT_OVERRIDE_PATH = os.path.join(_THIS_DIR, "simulation_params.py")


def _load_overrides(path: str) -> dict:
    """Import the Python file at `path` as a module and return its OVERRIDES dict.

    Args:
        path (str): Path to a Python file defining a module-level OVERRIDES dict
            (see simulation_params.py for the expected format).

    Returns:
        dict: The module's OVERRIDES dict, passed to SimulationConfig(**overrides).

    Raises:
        AttributeError: If the module at `path` has no OVERRIDES attribute.
    """
    spec = importlib.util.spec_from_file_location("sim_config_override", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "OVERRIDES"):
        raise AttributeError(f"{path} must define an OVERRIDES dict")
    return module.OVERRIDES


def load_config(override_path: Optional[str] = None) -> SimulationConfig:
    """Build a SimulationConfig from defaults + the named override file.

    Resolution order: explicit `override_path` argument, then the
    SIM_CONFIG_PATH env var (set per Slurm job/array task), then the
    bundled default override.

    SIM_ALGORITHM, if set, overrides the optimizer for every config. Because
    ALGORITHM is part of DATA_PATH/FIGURE_PATH, switching it sends results to a
    sibling directory (".../de/..." rather than ".../ga/...") instead of
    overwriting the existing fits, so an alternative optimizer can be run over
    the whole candidate set without touching the incumbent results.
    """
    path = override_path or os.environ.get("SIM_CONFIG_PATH") or DEFAULT_OVERRIDE_PATH
    overrides = _load_overrides(path)
    algorithm = os.environ.get("SIM_ALGORITHM")
    if algorithm:
        overrides = {**overrides, "ALGORITHM": algorithm}
    # N_JOBS defaults to -1 ("every core joblib can see"). Under Slurm that is
    # inferred from the cpuset, which is right for a whole-node allocation but
    # over-subscribes when the task is packed onto a node shared with other
    # users. SIM_N_JOBS lets the batch script state --cpus-per-task explicitly.
    n_jobs = os.environ.get("SIM_N_JOBS")
    if n_jobs:
        overrides = {**overrides, "N_JOBS": int(n_jobs)}
    return SimulationConfig(**overrides)
