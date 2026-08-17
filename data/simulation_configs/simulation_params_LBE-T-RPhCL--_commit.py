# Same model as LBE-T-RPhCL-- but fit using commit-only log-likelihood
# (decide vs wait, collapsing yellow/blue). Results saved under data/POMDP_commit/.
OVERRIDES = dict(
    POMDP_COMMIT=True,
    POMDP_TYPE="exaggerate",
    FIT_HORIZON=["long"],
    PARAM_RANGES={
        "tau": (0, 100),
        "subjective_cost": (-300, 0),
        "patience": (0, 14),
        "c_max": (0, 80),
        "belief_bias": (0.01, 5),
        "exaggeration_factor": (0.1, 4),
        "hazard_lapse": (0, 1),
    },
)
