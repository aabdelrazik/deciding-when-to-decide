# Forgetting three-way model, combined (short+long) horizon: identical
# parameter set to the fullest long-horizon rung (LB-XTGRPHCLUK), just
# FIT_HORIZON=["short","long"] instead of ["long"] alone.
OVERRIDES = dict(
    POMDP_TYPE="forgetting",
    FIT_HORIZON=["short", "long"],
    gamma_values=[0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "gamma": (0.7, 1.0),
        "subjective_cost": (-300, 0),
        "patience": (0, 8),
        "is_hazardous": (0, 1),
        "c_max": (0, 80),
        "hazard_lapse": (0, 1),
        "urgency_coefficient": (-30, 0),
        "urgency_slope": (-20, 0),
        "belief_bias": (0.01, 5),
    },
)
