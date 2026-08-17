# Neither-mechanism cell of the controlled short-horizon comparison: no
# exaggeration factor and Phi(t) = 0 at every draw (PATIENCE, URGENCY_COEFFICIENT,
# URGENCY_SLOPE and C_MAX all zero). The urgency class is used rather than the
# vanilla one only because vanilla's constructor does not accept belief_bias;
# with the sigmoid's bounds both zero the two are mathematically identical, and
# this keeps belief_bias free as in the other three cells.
OVERRIDES = dict(
    POMDP_TYPE="urgency",
    FIT_HORIZON=["short"],
    PATIENCE=0,
    URGENCY_COEFFICIENT=0,
    URGENCY_SLOPE=0,
    C_MAX=0,
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "belief_bias": (0.01, 5),
    },
)
