# Exaggeration-off cell of the controlled long-horizon comparison: identical to
# LBE-T-RPhCL-- (temporal regulation active, patience and c_max fitted) except
# that the evidence-exaggeration factor is absent.
OVERRIDES = dict(
    POMDP_TYPE="urgency",
    FIT_HORIZON=["long"],
    PARAM_RANGES={
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "patience": (0, 14),
        "c_max": (0, 80),
        "belief_bias": (0.01, 5),
        "hazard_lapse": (0, 1),
    },
)
