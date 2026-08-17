# Exaggeration-only cell of the controlled combined-horizon mechanism comparison:
# identical to the combined winner C-EXT-RPHC-UK except that Phi(t) = 0 at every
# draw (PATIENCE, URGENCY_COEFFICIENT, URGENCY_SLOPE and C_MAX all zero), so
# neither patience nor the sigmoid's bounds are fitted. The exaggeration factor
# and every non-mechanism parameter (xi, tau, subjective_cost, is_hazardous) stay
# free, matching the other three cells of the quartet.
OVERRIDES = dict(
    POMDP_TYPE="exaggerate",
    FIT_HORIZON=["short", "long"],
    PATIENCE=0,
    URGENCY_COEFFICIENT=0,
    URGENCY_SLOPE=0,
    C_MAX=0,
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "exaggeration_factor": (0.1, 4),
        "is_hazardous": (0, 1),
    },
)
