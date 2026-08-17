# Forgetting three-way model, long horizon: drops xi, and fixes
# is_hazardous=True explicitly (not fit) -- renders lowercase "h" in TASK
# per the fixed-hazard naming convention (src/config/schema.py).
OVERRIDES = dict(
    POMDP_TYPE="forgetting",
    FIT_HORIZON=["long"],
    IS_HAZARDOUS=True,
    gamma_values=[0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
    PARAM_RANGES={
        "tau": (0, 100),
        "gamma": (0.7, 1.0),
        "subjective_cost": (-300, 0),
        "patience": (0, 8),
        "c_max": (0, 80),
        "hazard_lapse": (0, 1),
        "belief_bias": (0.01, 5),
    },
)
