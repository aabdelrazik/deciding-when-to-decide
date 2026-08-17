from src.config.loader import load_config
from src.config.schema import SimulationConfig

# Built once per process, from whichever override SIM_CONFIG_PATH names
# (or the bundled default if unset). Each Slurm job/array task gets its
# own process and its own SIM_CONFIG_PATH, so concurrent jobs never share
# this mutable state.
CONFIG = load_config()

# Backward compatibility: existing modules do `from src.config import *`
# and expect bare names like TASK, DATA_PATH, PARAM_RANGES, etc.
globals().update(CONFIG.as_globals())
