from .pomdp import POMDPFactory
from .pomdp_simulate import *
from .validation import accepted_param_keys, validate_param_ranges

from src.config import CONFIG

# Fail fast and clearly: if the active config's PARAM_RANGES asks to fit a
# parameter its POMDP_TYPE doesn't accept, every GA/DE evaluation would
# otherwise fail silently (make_cost_function catches the TypeError and
# returns a flat 1e10 penalty) and produce a meaningless-but-complete fit.
validate_param_ranges(CONFIG.POMDP_TYPE, CONFIG.PARAM_RANGES)
