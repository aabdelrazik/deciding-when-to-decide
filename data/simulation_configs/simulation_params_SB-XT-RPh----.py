# Exaggeration-off cell of the controlled short-horizon mechanism comparison:
# identical to SBEXT-RPh---- (temporal regulation active, C_MAX fixed at 50 with
# patience fitted) except that the evidence-exaggeration factor is absent, so the
# urgency class is used in place of the exaggerate class. Every non-mechanism
# parameter (xi, tau, subjective_cost, belief_bias) stays free, and the hazard
# stays on-but-fixed, so the only difference from the anchor is exaggeration.
OVERRIDES = dict(
    POMDP_TYPE="urgency",
    FIT_HORIZON=["short"],
    C_MAX=50,
    PARAM_RANGES={
        "xi": (0, 1),
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "patience": (0, 14),
        "belief_bias": (0.01, 5),
    },
)
