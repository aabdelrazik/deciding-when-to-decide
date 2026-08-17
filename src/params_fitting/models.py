import numpy as np

from src.pomdp import POMDPFactory


def model_simulation(params: dict) -> np.ndarray:
    """Build a POMDP of the config's default type, initialize it with `params`,
    run value iteration, and return the resulting policy.

    Args:
        params (dict): Keyword args for the POMDP class's constructor.

    Returns:
        np.ndarray: self.policy, shape (l_draws, n_yellows, m_blues, 3)
            (softmax action probabilities per state).
    """
    pomdp = POMDPFactory()
    pomdp.__init__(**params)
    pomdp.value_iteration()
    return pomdp.policy
