# Nested ablation of the best short-horizon model SBEXT-RPh----: identical in
# every respect except that patience is no longer fitted and the temporal
# regulation function is switched off entirely (PATIENCE, URGENCY_COEFFICIENT,
# URGENCY_SLOPE and C_MAX all zero, so Phi(t) = 0 at every draw). This supplies
# the "recency without temporal regulation" cell of the exaggeration-by-urgency
# comparison, which the short-horizon candidate set otherwise lacks; the long
# horizon already has it as LBEXT-R-H----.
OVERRIDES = dict(
    POMDP_TYPE="exaggerate",
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
        "exaggeration_factor": (0.1, 4),
    },
)
