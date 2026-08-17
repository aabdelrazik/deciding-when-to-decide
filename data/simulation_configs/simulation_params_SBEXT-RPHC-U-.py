# Auto-converted from the original standalone config file.
# Only fields that differ from src/config/schema.py's defaults are listed.
OVERRIDES = dict(
    POMDP_TYPE="exaggerate",
    FIT_HORIZON=["short"],
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "urgency_coefficient": (-30, 0),
        "subjective_cost": (-300, 0),
        "patience": (0, 8),
        "c_max": (0, 80),
        "belief_bias": (0.01, 5),
        "exaggeration_factor": (0.1, 4),
        "is_hazardous": (0, 1),
    },
)
