import inspect

from src.config.schema import TASK_ORDER

# PARAM_RANGES keys the fitting/task-naming system recognizes at all (see
# src/config/schema.py's TASK_ORDER). A model type may accept a subset of these.
FITTABLE_PARAM_KEYS = {key for key, _letter in TASK_ORDER}


def _type_classes():
    # Imported lazily: src.pomdp.pomdp imports `from src.config import *`, so importing
    # it at module load time here would be circular if this module were imported from
    # src.config. It isn't, but keep this lazy anyway in case that ever changes.
    from src.pomdp.pomdp import (
        POMDP,
        POMDP_Urgency,
        POMDP_exaggerate,
        POMDP_Exaggeration,
        POMDP_Forgetting,
    )

    return {
        "vanilla": POMDP,
        "urgency": POMDP_Urgency,
        "exaggerate": POMDP_exaggerate,
        "exaggerate_data": POMDP_Exaggeration,
        "forgetting": POMDP_Forgetting,
    }


def accepted_param_keys(pomdp_type: str) -> set:
    """The PARAM_RANGES keys that POMDP_TYPE's constructor actually accepts."""
    classes = _type_classes()
    if pomdp_type not in classes:
        raise ValueError(
            f"Unknown POMDP_TYPE {pomdp_type!r}; expected one of {sorted(classes)}"
        )
    cls = classes[pomdp_type]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return FITTABLE_PARAM_KEYS & accepted


def validate_param_ranges(pomdp_type: str, param_ranges: dict) -> None:
    """Raise a clear error if PARAM_RANGES asks to fit a parameter POMDP_TYPE's
    constructor doesn't accept (it would otherwise fail silently: make_cost_function
    catches the resulting TypeError per-evaluation and returns a flat 1e10 penalty,
    so the optimizer runs to completion but every result is meaningless)."""
    allowed = accepted_param_keys(pomdp_type)
    invalid = set(param_ranges) - allowed
    if invalid:
        raise ValueError(
            f"PARAM_RANGES key(s) {sorted(invalid)} are not accepted by "
            f"POMDP_TYPE={pomdp_type!r}'s constructor. Valid PARAM_RANGES keys "
            f"for {pomdp_type!r}: {sorted(allowed)}"
        )
