# Forgetting three-way model, short horizon: drops urgency_coefficient/
# urgency_slope from the fullest rung (SB-XTGRPHC-UK).
OVERRIDES = dict(
    POMDP_TYPE="forgetting",
    FIT_HORIZON=["short"],
    gamma_values=[0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 30),
        "gamma": (0.8, 1.0),
        "subjective_cost": (-300, 0),
        "patience": (0, 8),
        "is_hazardous": (0, 1),
        "c_max": (0, 80),
        "belief_bias": (0, 5),
    },
)
