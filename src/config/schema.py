import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# Maps a PARAM_RANGES key to the letter it contributes to TASK when present.
TASK_ORDER = [
    ("belief_bias", "B"),
    ("exaggeration_factor", "E"),
    ("xi", "X"),
    ("tau", "T"),
    ("gamma", "G"),
    ("subjective_cost", "R"),
    ("patience", "P"),
    ("is_hazardous", "H"),
    ("c_max", "C"),
    ("hazard_lapse", "L"),
    ("urgency_coefficient", "U"),
    ("urgency_slope", "K"),
]


def _horizon_prefix(fit_horizon: list) -> str:
    """Map FIT_HORIZON to the leading letter of the TASK name.

    Args:
        fit_horizon (list): Must be a list containing "short", "long", or both.

    Returns:
        str: "C" (combined) if both are present, "S" for short-only, "L" for
            long-only.
    """
    if (
        isinstance(fit_horizon, list)
        and "long" in fit_horizon
        and "short" in fit_horizon
    ):
        return "C"
    elif isinstance(fit_horizon, list) and "short" in fit_horizon:
        return "S"
    elif isinstance(fit_horizon, list) and "long" in fit_horizon:
        return "L"
    raise ValueError(f"Unrecognized FIT_HORIZON: {fit_horizon!r}")


def build_task_name(
    fit_horizon: list, param_ranges: dict, is_hazardous_fixed: bool | None = None
) -> str:
    """Build the TASK name: a horizon-prefix letter followed by one letter per
    TASK_ORDER entry (the entry's letter if its key is in param_ranges, else "-").

    Args:
        fit_horizon (list): Passed through to _horizon_prefix.
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple;
            only membership of the keys is used here.
        is_hazardous_fixed (bool | None): Only consulted when "is_hazardous"
            is NOT in param_ranges (i.e. it isn't fit), and then it is the
            *resolved* hazard state rather than whether the config set it
            explicitly. True renders lowercase "h" (hazard on but not fit),
            False renders "-" (hazard off). None is treated as False.

    Returns:
        str: The task name, e.g. "LB-XT-RPHCLUK".

    Notes:
        The hazard slot distinguishes three states unambiguously: "H" when the
        hazard is a free parameter, "h" when it is on but fixed, and "-" when
        it is off, so the dash always means off and never on-by-default.
    """
    prefix = _horizon_prefix(fit_horizon)
    letters = []
    for key, letter in TASK_ORDER:
        if key in param_ranges:
            letters.append(letter)
        elif key == "is_hazardous" and is_hazardous_fixed is True:
            letters.append(letter.lower())
        else:
            letters.append("-")
    return prefix + "".join(letters)


# Sentinel distinguishing "IS_HAZARDOUS left at its class default" from "a
# config file's OVERRIDES explicitly set IS_HAZARDOUS=True" -- both resolve to
# the same bool (True) for every other purpose, but build_task_name() needs to
# tell them apart (see __post_init__) to render the "fixed on, not fit"
# lowercase "h" only for the latter. Resolved to a real bool by the end of
# __post_init__, so nothing outside this module ever observes the sentinel.
_UNSET_IS_HAZARDOUS = object()


@dataclass
class SimulationConfig:
    # --- Simulation control ---
    POMDP_COMMIT: bool = False  # if True, fit using log_likelihood_commit (commit/wait only) instead of full 3-way LL; saves to data/POMDP_commit/
    # 'de' = differential evolution, 'ga' = genetic algorithm. All reported
    # results use 'de'. The algorithm is part of the output path, so the two
    # never mix.
    ALGORITHM: str = "de"
    TAU: float = 0.001
    XI: float = 0
    DEADLINE: int = 14
    HORIZON_CONDITION: str = "long"
    SUBJECTIVE_COST: float = 0
    VERBOSE: bool = False
    PATIENCE: int = 0
    URGENCY_COEFFICIENT: float = -10
    URGENCY_SLOPE: float = -2
    C_MAX: float = 0.2
    IS_HAZARDOUS: bool = _UNSET_IS_HAZARDOUS  # type: ignore[assignment]  # resolved in __post_init__; see _UNSET_IS_HAZARDOUS
    MAX_CARDS_PER_DRAW: int = 5
    POMDP_TYPE: str = "vanilla"
    EXAGGERATION_FACTOR: float = 1
    gamma_values: List[float] = field(
        default_factory=lambda: [1.0, 1.05, 1.1, 1.15, 1.2]
    )
    SWEETSPOT_TAU: float = 0
    SWEETSPOT_XI: float = 0
    BELIEF_BIAS: float = 1
    HAZARD_LAPSE: float = 0
    GAMMA: float = 1

    # --- Parameter fitting ranges ---
    FIT_HORIZON: List[str] = field(default_factory=lambda: ["short"])
    PARAM_RANGES: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # --- Data / project settings ---
    N_SUBJECTS: int = 105
    N_JOBS: int = -1

    # --- Computed fields (do not set directly; derived in __post_init__) ---
    PROJECT_ROOT: str = field(init=False)
    TASK: str = field(init=False)
    PARAM_ORDER: list = field(init=False)
    variable_type: list = field(init=False)
    FIGURE_PATH: str = field(init=False)
    DATA_PATH: str = field(init=False)
    HUMAN_DATA_PATH: str = field(init=False)
    ALL_EVIDENCE_PATH: str = field(init=False)
    FULL_SIM_DF_PATH: str = field(init=False)
    FULL_SIM_DF_PATH_compressed: str = field(init=False)
    RESULTS_PATH: str = field(init=False)
    FULL_SIM_DF_RECOVERED_PATH: str = field(init=False)
    FULL_SIM_DF_RECOVERED_PATH_compressed: str = field(init=False)
    RESULTS_RECOVERED_PATH: str = field(init=False)

    def __post_init__(self) -> None:
        """Derive TASK, PARAM_ORDER, variable_type, and the various file-system
        paths from the fields set by __init__, and validate PARAM_RANGES
        (e.g. hazard_lapse is incompatible with a short-only FIT_HORIZON).
        """
        # Resolved from this file, not from the working directory, so scripts
        # run from anywhere. POMDP_ROOT overrides it.
        self.PROJECT_ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

        is_short_only = len(self.FIT_HORIZON) == 1 and "short" in self.FIT_HORIZON

        # IS_HAZARDOUS explicitness only matters (for TASK naming and the
        # hazard_lapse check below) when it's fixed rather than fit -- resolve
        # the sentinel to the real default (True) either way so every other
        # reader of self.IS_HAZARDOUS just sees a plain bool, as before.
        is_hazardous_explicit = self.IS_HAZARDOUS is not _UNSET_IS_HAZARDOUS
        if not is_hazardous_explicit:
            self.IS_HAZARDOUS = True
        # The TASK hazard slot reflects the resolved state, not whether the
        # config happened to set it explicitly: "h" whenever the hazard is on
        # but not fit, "-" only when it is genuinely off.
        is_hazardous_fixed = (
            self.IS_HAZARDOUS
            if "is_hazardous" not in self.PARAM_RANGES
            else None
        )
        hazard_fixed_off = is_hazardous_fixed is False

        assert not (
            "hazard_lapse" in self.PARAM_RANGES and (is_short_only or hazard_fixed_off)
        ), (
            f"hazard_lapse cannot be fit under a short-only FIT_HORIZON, nor "
            f"alongside IS_HAZARDOUS explicitly fixed to False (the hazard "
            f"mechanism is off, so a lapse rate for it is meaningless) "
            f"(FIT_HORIZON={self.FIT_HORIZON!r}, IS_HAZARDOUS={self.IS_HAZARDOUS!r})"
        )

        if "patience" in self.PARAM_RANGES:
            self.PARAM_RANGES["patience"] = (0, 8 if is_short_only else 14)

        self.TASK = build_task_name(self.FIT_HORIZON, self.PARAM_RANGES, is_hazardous_fixed)
        self.PARAM_ORDER = list(self.PARAM_RANGES.keys())
        # Mixed-type spec for the GA optimizer (geneticalgorithm2). Matches the
        # original per-config boilerplate: is_hazardous is fit as a 0/1 int,
        # everything else as real, and is_hazardous is always last in PARAM_ORDER.
        if "is_hazardous" in self.PARAM_RANGES:
            self.variable_type = ["real"] * (len(self.PARAM_ORDER) - 1)
            self.variable_type.append("int")
        else:
            self.variable_type = ["real"] * len(self.PARAM_ORDER)

        # POMDP_SUBDIR redirects both data/ and figures/ to a separate tree, so a
        # run that changes the model itself cannot overwrite the current fits.
        _subdir = os.environ.get("POMDP_SUBDIR") or (
            "POMDP_commit" if self.POMDP_COMMIT else "POMDP"
        )
        if len(self.FIT_HORIZON) == 1:
            self.FIGURE_PATH = os.path.join(
                self.PROJECT_ROOT,
                f"figures/{_subdir}/{self.TASK}/{self.ALGORITHM}/{self.FIT_HORIZON[0]}",
            )
            self.DATA_PATH = os.path.join(
                self.PROJECT_ROOT,
                f"data/{_subdir}/{self.TASK}/{self.ALGORITHM}/{self.FIT_HORIZON[0]}",
            )
        else:
            self.FIGURE_PATH = os.path.join(
                self.PROJECT_ROOT, f"figures/{_subdir}/{self.TASK}/{self.ALGORITHM}"
            )
            self.DATA_PATH = os.path.join(
                self.PROJECT_ROOT, f"data/{_subdir}/{self.TASK}/{self.ALGORITHM}"
            )

        self.HUMAN_DATA_PATH = os.path.join(
            self.PROJECT_ROOT, "data/TrHu_NHB_light/data_MEG/behdat_preprocessed.pkl"
        )
        self.ALL_EVIDENCE_PATH = os.path.join(
            self.PROJECT_ROOT,
            "data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts.pkl",
        )
        self.FULL_SIM_DF_PATH = os.path.join(self.DATA_PATH, "all_simulated_data.pkl")
        self.FULL_SIM_DF_PATH_compressed = os.path.join(
            self.DATA_PATH, "all_simulated_data_compressed.pkl"
        )
        self.RESULTS_PATH = os.path.join(self.DATA_PATH, "results.pkl")
        self.FULL_SIM_DF_RECOVERED_PATH = os.path.join(
            self.DATA_PATH, "all_simulated_data_recovered.pkl"
        )
        self.FULL_SIM_DF_RECOVERED_PATH_compressed = os.path.join(
            self.DATA_PATH, "all_simulated_data_recovered_compressed.pkl"
        )
        self.RESULTS_RECOVERED_PATH = os.path.join(
            self.DATA_PATH, "results_recovered.pkl"
        )

    def as_globals(self) -> dict:
        """Flat dict of every field, for modules still doing `from src.config import *`."""
        return asdict(self)
