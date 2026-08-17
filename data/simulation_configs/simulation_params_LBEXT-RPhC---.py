# Auto-converted from the original standalone config file.
# Only fields that differ from src/config/schema.py's defaults are listed.
OVERRIDES = dict(
    POMDP_TYPE="exaggerate",
    FIT_HORIZON=["long"],
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "patience": (0, 14),
        "c_max": (0, 80),
        "belief_bias": (0.01, 5),
        "exaggeration_factor": (0.1, 4),
    },
)
