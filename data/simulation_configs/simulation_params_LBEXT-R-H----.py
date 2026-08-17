# Auto-converted from the original standalone config file.
# Only fields that differ from src/config/schema.py's defaults are listed.
OVERRIDES = dict(
    POMDP_TYPE="exaggerate",
    PATIENCE=0,
    URGENCY_COEFFICIENT=0,
    URGENCY_SLOPE=0,
    C_MAX=0,
    FIT_HORIZON=["long"],
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "belief_bias": (0.01, 5),
        "exaggeration_factor": (0.1, 4),
        "is_hazardous": (0, 1),
    },
)
