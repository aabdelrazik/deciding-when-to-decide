# Temporal-regulation-off cell of the controlled long-horizon comparison:
# identical to the best long-horizon model LBE-T-RPhCL-- except that Phi(t) = 0
# at every draw, so neither patience nor c_max is fitted. Exaggeration and every
# non-mechanism parameter (tau, subjective_cost, belief_bias, hazard_lapse) stay
# free, and the hazard stays on-but-fixed.
OVERRIDES = dict(
    POMDP_TYPE="exaggerate",
    FIT_HORIZON=["long"],
    PATIENCE=0,
    URGENCY_COEFFICIENT=0,
    URGENCY_SLOPE=0,
    C_MAX=0,
    PARAM_RANGES={
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "belief_bias": (0.01, 5),
        "exaggeration_factor": (0.1, 4),
        "hazard_lapse": (0, 1),
    },
)
