# Neither-mechanism cell of the controlled long-horizon comparison: no
# exaggeration factor and Phi(t) = 0 at every draw. As in SB-XT-R-h----, the
# urgency class is used with the sigmoid's bounds zeroed rather than the vanilla
# class, because vanilla's constructor does not accept belief_bias.
OVERRIDES = dict(
    POMDP_TYPE="urgency",
    FIT_HORIZON=["long"],
    PATIENCE=0,
    URGENCY_COEFFICIENT=0,
    URGENCY_SLOPE=0,
    C_MAX=0,
    PARAM_RANGES={
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "belief_bias": (0.01, 5),
        "hazard_lapse": (0, 1),
    },
)
