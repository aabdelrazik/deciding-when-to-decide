import numpy as np
from scipy.stats import beta, binom
import scipy.integrate as integrate
from math import ceil
from scipy.special import comb, betaln, gammaln
import json
import os
from src.config import *
from src.utils import (
    calculate_hazard_cum,
)

import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import matplotlib.colors as mcolors
import matplotlib as mpl
from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import dual_annealing, differential_evolution
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator as _RGI

# from src.params_fitting import get_horizon_keys

IS_LATEX = True


class POMDP:
    """
    Base card-drawing POMDP: solves for the optimal (softmax) Yellow/Blue/Wait
    policy via backward-induction value iteration over (draw, num_yellow, num_blue)
    states, optionally under a hazard (deadline) function. Also provides fitting
    (log-likelihood, GA/differential-evolution optimizers) and plotting utilities.
    Subclasses add urgency costs, non-integer/interpolated grids, forgetting, and
    evidence exaggeration on top of this base.
    """

    def __init__(
        self,
        tau: float = TAU,
        xi: float = XI,
        hazard_lapse: float = HAZARD_LAPSE,
        subjective_cost: float = SUBJECTIVE_COST,
        horizon_condition: str = HORIZON_CONDITION,
        max_cards_per_draw: int = MAX_CARDS_PER_DRAW,
        is_hazardous: bool = IS_HAZARDOUS,
        verbose: bool = VERBOSE,
        start_hazard: int | None = None,
    ):
        """
        Initialize the vanilla POMDP (no urgency/forgetting/exaggeration) and
        allocate the (draws, yellows, blues, actions) tables used by value
        iteration.

        Args:
            tau (float): Softmax temperature for the action policy.
            xi (float): Lapse rate mixed into the Yellow/Blue (non-Wait) softmax probabilities.
            hazard_lapse (float): Mixing weight between the short- and long-horizon
                cumulative hazard curves when horizon_condition == "long".
            subjective_cost (float): Added to the incorrect-choice reward (reward_incorrect = subjective_cost - 2).
            horizon_condition (str): "short" or "long"; sets max_draws and the default start_hazard.
            max_cards_per_draw (int): Number of cards revealed per draw.
            is_hazardous (bool): Whether a hazard (deadline) function is applied during planning.
            verbose (bool): Whether to print progress information.
            start_hazard (int, optional): Draw at which the hazard function starts. Defaults to 4
                (short horizon) or 10 (long horizon) when None.
        """
        # the temperature parameter for the softmax policy
        self.tau = tau
        # the lapse rate parameter for the softmax policy
        self.xi = xi
        # This is whether there is a hazard function implemented during the planning or not
        self.is_hazardous = bool(is_hazardous)
        # if the answer is yes, then the hazard condition is either short or long
        self.horizon_condition = horizon_condition
        # Here we set the horizons as described in the actual task,
        # i.e., short horizon: hazard starts at the number of draws 4 and ends at 8
        # long horizon: starts at 10 and ends at 14
        # Here if the is_hazardous is False, we just ignore the start_hazard and the deadline is just the max_number of draws which is needed anyways
        # otherwise, everything is used so no need to do if condition here before initialization and the if condition for either to put the hazard or not is already implemented below during the value iteration
        if start_hazard is None:
            self.start_hazard = 4 if horizon_condition == "short" else 10
        else:
            self.start_hazard = start_hazard
        if horizon_condition == "long":
            self.max_draws = 14

        elif horizon_condition == "short":
            self.max_draws = 8
        self.hazard_lapse = hazard_lapse
        if is_hazardous:
            if horizon_condition == "long":
                # Get the hazard arrays
                hazard_10_14 = calculate_hazard_cum(10, 14)
                hazard_4_8 = calculate_hazard_cum(4, 8)

                # Create a common array size (max length needed)
                max_len = max(len(hazard_10_14), len(hazard_4_8))
                self.cum_hazard = np.zeros(max_len)

                # Add weighted hazards at their respective positions
                self.cum_hazard[: len(hazard_10_14)] += (
                    1 - self.hazard_lapse
                ) * hazard_10_14
                self.cum_hazard[: len(hazard_4_8)] += self.hazard_lapse * hazard_4_8
            else:
                self.cum_hazard = calculate_hazard_cum(4, 8)
        else:
            self.cum_hazard = np.zeros((self.max_draws + 1))
        self.reward_incorrect = subjective_cost - 2
        self.reward_correct = 2
        self.deadline_cost = -0.5 * self.reward_correct
        # The following alpha and beta are the priors of the beta distribution
        self.alpha = 1
        self.beta = 1
        # I redefined the same variable here just for easier readability
        # so the max_draws are the same as the deadline: But the naming of the deadline in the hazard
        # is indicative and here the max_draws is indicative when initializing or looping the following variables
        self.max_cards_per_draw = max_cards_per_draw
        self.verbose = verbose

        # the next_yellow is used for the transition probability function below
        self.next_yellow = np.arange(0, max_cards_per_draw + 1)

        # The following are the initialization of the arrays that will be used to store the
        # action values, value function, belief, transition probability and softmaxpolicy
        self.max_yellows = self.max_cards_per_draw * self.max_draws
        self.max_blues = self.max_cards_per_draw * self.max_draws

        self.action_values = np.zeros(
            (self.max_draws + 1, self.max_yellows + 1, self.max_blues + 1, 3)
        )
        self.value_function = np.zeros(
            (self.max_draws + 1, self.max_yellows + 1, self.max_blues + 1)
        )
        self.belief = np.zeros(
            (self.max_draws + 1, self.max_yellows + 1, self.max_blues + 1, 2)
        )
        self.transition_probability = np.zeros(
            (self.max_draws + 1, self.max_yellows + 1, self.max_blues + 1)
        )
        self.policy = np.zeros(
            (self.max_draws + 1, self.max_yellows + 1, self.max_blues + 1, 3)
        )

        self.best_actions = (
            np.ones((self.max_draws + 1, self.max_yellows + 1, self.max_blues + 1)) * 5
        )

    def softmax_policy(
        self, action_values: np.ndarray, draw: int = 0, axis: int = -1
    ) -> np.ndarray:
        """
        Numerically stable softmax with lapse applied ONLY to first two actions.
        The last action is NOT affected at all (its probability is set by pure softmax).
        The total sum is always 1; total of first two actions is preserved.

        Args:
            action_values (np.ndarray): Q-values (..., n_actions), with Wait as the last action.
            draw (int): Current draw index. Accepted for a draw-dependent softmax but currently
                unused in this implementation (self.tau/self.xi are applied uniformly across draws).
            axis (int): Axis over which to apply the softmax (must index the actions dimension).

        Returns:
            np.ndarray: Action-probability array with the same shape as action_values.
        """
        # Standard, stable softmax
        max_val = np.max(action_values, axis=axis, keepdims=True)
        all_impossible_mask = np.isneginf(max_val)
        safe_max_val = np.where(all_impossible_mask, 0, max_val)
        # tau is a softmax denominator -- as it approaches 0 the division
        # blows up (divide-by-zero/overflow), so floor it for this division
        # only; self.tau itself (the fitted/reported parameter) is untouched.
        safe_tau = max(self.tau, 1e-3)
        scaled_values = (action_values - safe_max_val) / safe_tau

        exps = np.exp(scaled_values)
        denominator = np.sum(exps, axis=axis, keepdims=True)
        softmax_probs = np.nan_to_num(exps / denominator)

        # Save original last action
        probs_last = softmax_probs[..., -1]

        # Get total probability of first two before mixing
        probs_first_two_sum = np.sum(softmax_probs[..., :2], axis=axis, keepdims=True)

        # Mix the first two: new = (1-xi)*orig + xi/2, but RESCALE so their total is unchanged
        # Calculate the sum after lapse would have been applied
        mixed_first_two = softmax_probs[..., :2] * (1 - self.xi) + self.xi / 2
        mixed_sum = np.sum(mixed_first_two, axis=axis, keepdims=True)

        # Scale factor so the sum of the first two actions remains the same as before
        scaling = np.divide(
            probs_first_two_sum,
            mixed_sum,
            out=np.ones_like(mixed_sum),
            where=(mixed_sum != 0),
        )
        mixed_first_two = mixed_first_two * scaling

        # Stack all together: first two mixed, last untouched
        final_policy = np.concatenate(
            [mixed_first_two, probs_last[..., None]], axis=axis
        )

        return final_policy

    def calculate_reward(
        self,
        is_correct: bool,
    ) -> float:
        """
        Return the fixed reward for a Yellow/Blue choice, depending on whether it was correct.

        Args:
            is_correct (bool): Whether the choice matched the generative majority color.

        Returns:
            float: self.reward_correct if is_correct else self.reward_incorrect.
        """

        if is_correct:
            return self.reward_correct
        else:
            return self.reward_incorrect

    def calculate_belief_probability(
        self, num_blues: int, num_yellows: int, q: float = 0.5
    ) -> float:
        """
        Calculate the probability that the underlying generative probability q_y > q,
        given num_yellows and num_blues, using a Beta posterior.

        Args:
            num_blues (int): Number of blue cards drawn.
            num_yellows (int): Number of yellow cards drawn.
            q (float): Threshold probability to compare against (default is 0.5).

        Returns:
            float: Posterior probability that the generative probability q_y > q.
        """
        Alpha = self.alpha + num_yellows
        Beta_ = self.beta + num_blues

        posterior_prob = 1 - beta.cdf(q, Alpha, Beta_)
        return posterior_prob

    def calculate_transition_probability(
        self, i: np.ndarray, num_yellows: np.ndarray, num_blues: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the transition probability using the analytical beta-binomial formula.
        This version is fully vectorized to handle array inputs.

        Args:
            i (np.ndarray): Array of the number of yellow cards in the next draw (k).
            num_yellows (np.ndarray): Array of the total number of yellow cards observed so far.
            num_blues (np.ndarray): Array of the total number of blue cards observed so far.

        Returns:
            np.ndarray: An array of transition probability values.
        """
        # Parameters for the posterior beta distribution. These are now arrays.
        alpha_post = self.alpha + num_yellows
        beta_post = self.beta + num_blues
        n = self.max_cards_per_draw

        # Suppress warnings for invalid values (e.g., log(0)) since we handle them.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_comb = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)

            log_prob = (
                log_comb
                + betaln(alpha_post + i, beta_post + n - i)
                - betaln(alpha_post, beta_post)
            )

        # Convert back from log-probability to probability
        probabilities = np.exp(log_prob)

        valid_mask = (i >= 0) & (i <= self.max_cards_per_draw)

        return np.where(valid_mask, np.nan_to_num(probabilities), 0.0)

    def extract_actions(self) -> None:
        """
        Extract best actions with custom tie-breaking:
        If multiple actions are tied within tolerance, prefer index 2 over 0 or 1.

        Returns:
            None. Sets self.best_actions in place.
        """

        policy = self.policy

        max_vals = np.max(policy, axis=-1, keepdims=True)

        # tolerance-based tie detection
        is_max = np.abs(policy - max_vals) <= 1e-6

        best_actions = np.full(policy.shape[:-1], -1, dtype=int)

        # Priority: 2 first, then 0, then 1
        priority = [2, 0, 1]

        for a in priority:
            candidate = is_max[..., a]
            best_actions[(best_actions == -1) & candidate] = a

        self.best_actions = best_actions

    def calculate_action_values_for_Y_and_B(self):
        """
        Calculate the action values for all states using vectorized operations.
        """
        # Create a grid of all possible draw and yellow counts
        # Note: We create a full grid and will use a mask to select valid states.
        max_total_yellows = self.max_draws * self.max_cards_per_draw
        draws_grid, yellows_grid = np.meshgrid(
            np.arange(self.max_draws + 1),
            np.arange(max_total_yellows + 1),
            indexing="ij",  # Use matrix indexing
        )

        # Calculate the corresponding number of blue cards
        blues_grid = draws_grid * self.max_cards_per_draw - yellows_grid

        # Create a mask to identify the valid states
        # A state is valid if blue_count is not negative.
        valid_mask = blues_grid >= 0

        # Use the mask to get only the valid states
        valid_draws = draws_grid[valid_mask]
        valid_yellows = yellows_grid[valid_mask]
        valid_blues = blues_grid[valid_mask]

        # --- Belief Calculation ---
        # Assuming self.calculate_belief_probability is vectorized
        belief_yellow = self.calculate_belief_probability(valid_blues, valid_yellows)
        belief_blue = 1 - belief_yellow

        # Update the belief array using advanced indexing
        self.belief[valid_draws, valid_yellows, valid_blues, 0] = belief_yellow
        self.belief[valid_draws, valid_yellows, valid_blues, 1] = belief_blue

        # --- Reward Calculation ---
        # Assuming self.reward is vectorized
        reward_correct = self.calculate_reward(is_correct=True)
        reward_incorrect = self.calculate_reward(is_correct=False)

        # --- Action Value Calculation ---
        # Calculate action values for all valid states at once
        yellow_action_value = (
            reward_correct * belief_yellow + reward_incorrect * belief_blue
        )
        blue_action_value = (
            reward_correct * belief_blue + reward_incorrect * belief_yellow
        )

        # Update the action_values array using advanced indexing
        self.action_values[valid_draws, valid_yellows, valid_blues, 0] = (
            yellow_action_value
        )
        self.action_values[valid_draws, valid_yellows, valid_blues, 1] = (
            blue_action_value
        )

    def value_iteration(self) -> None:
        """
        Perform value iteration using vectorized operations for the inner loops.

        Returns:
            None. Populates self.action_values, self.value_function, self.policy,
            self.transition_probability, and (via extract_actions) self.best_actions.
        """
        # Initialize terminal state values
        self.calculate_action_values_for_Y_and_B()

        self.action_values[-1, :, :, 2] = self.deadline_cost

        # The policy at the terminal state is the softmax over the three Q-values.
        policy_probs = self.softmax_policy(
            self.action_values[-1, :, :, :], self.max_draws
        )
        self.policy[-1, :, :, :] = policy_probs

        self.value_function[-1, :, :] = np.sum(
            policy_probs * self.action_values[-1, :, :, :], axis=-1
        )

        for draw in range(self.max_draws - 1, -1, -1):
            # --- 1. Define the state space for the current draw ---
            max_possible_yellows = draw * self.max_cards_per_draw
            current_yellows = np.arange(max_possible_yellows + 1)
            current_blues = draw * self.max_cards_per_draw - current_yellows

            # --- 2. Calculate the expected value of the next state ---

            # Create broadcastable arrays for outcomes of the next draw
            # next_yellows has shape (N_outcomes,)
            next_yellows_arr = np.array(self.next_yellow)
            next_blues_arr = self.max_cards_per_draw - next_yellows_arr

            # Use broadcasting to find all future states from all current states
            future_yellows = (
                current_yellows[:, np.newaxis] + next_yellows_arr[np.newaxis, :]
            )
            future_blues = current_blues[:, np.newaxis] + next_blues_arr[np.newaxis, :]

            # Get the action values at all possible future states
            # Result has shape (N_states, N_outcomes, N_actions)
            action_values_next = self.action_values[
                draw + 1, future_yellows, future_blues
            ]

            # Apply softmax policy and calculate the expected value for each future state
            # Assuming self.softmax_policy is vectorized or can be replaced by a vectorized version
            policy_probs = self.softmax_policy(
                action_values_next, draw + 1, axis=-1
            )  # axis=-1 applies softmax over actions
            expected_future_value = np.sum(
                policy_probs * action_values_next, axis=-1
            )  # Shape: (N_states, N_outcomes)

            # Store the calculated value function and policy
            self.value_function[draw + 1, future_yellows, future_blues] = (
                expected_future_value
            )
            self.policy[draw + 1, future_yellows, future_blues] = policy_probs

            # --- 3. Calculate transition probabilities ---
            transitions = self.calculate_transition_probability(
                i=next_yellows_arr[np.newaxis, :],
                num_yellows=current_yellows[:, np.newaxis],
                num_blues=current_blues[:, np.newaxis],
            )  # Shape: (N_states, N_outcomes)
            self.transition_probability[draw, future_yellows, future_blues] = (
                transitions
            )
            # --- 4. Calculate the "wait" action value ---
            if self.is_hazardous:
                discount_factor = 1 - self.cum_hazard[draw]
                # Sum over the "outcomes" axis (axis=1)
                wait_value_update = discount_factor * np.sum(
                    transitions * expected_future_value, axis=1
                )
                total_wait_value = (
                    wait_value_update + self.cum_hazard[draw] * self.deadline_cost
                )
            else:

                # hazard plus urgency.
                wait_value_update = np.sum(
                    transitions * expected_future_value, axis=1
                )  #
                total_wait_value = wait_value_update

            # --- 5. Update the action_values table for the current draw ---
            self.action_values[draw, current_yellows, current_blues, 2] = (
                total_wait_value
            )
            current_q_values = self.action_values[draw, current_yellows, current_blues]

            self.policy[draw, current_yellows, current_blues, :] = self.softmax_policy(
                current_q_values, draw
            )

        # Finally, extract the best actions based on the fully populated table
        self.extract_actions()

    # Fitting Functions
    def log_likelihood(self, data: "pd.DataFrame") -> float:
        """
        Sum the log-probability the fitted policy assigns to each subject's observed
        sequence of Wait actions followed by a final Yellow/Blue choice.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column
                of per-trial sequences, each an iterable of (draw, num_yellow, num_blue,
                action, outcome) tuples with action in {0: Yellow, 1: Blue, 2: Wait}.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        eps = 1e-10
        for action_seq in actions_col:

            for draw, y, b, act, outcome in action_seq:
                p_y, p_b, p_w = self.policy[draw, y, b, :3]
                if act == 2:  # wait
                    total_ll += np.log(p_w + eps)
                elif act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    total_ll += np.log((p_action) + eps)
                    break  # I need this break, because I have the full sequence and subjects can decide but still see the pre-assigned sequence yet.

        return total_ll

    # Fitting Functions
    def log_likelihood_commit(self, data: "pd.DataFrame") -> float:
        """
        Same as log_likelihood, but collapses Yellow/Blue into a single "commit/go"
        probability (p_y + p_b) so the fit only distinguishes Wait from committing,
        not which color was chosen.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column,
                as in log_likelihood.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        eps = 1e-10
        for action_seq in actions_col:

            for draw, y, b, act, outcome in action_seq:
                p_y, p_b, p_w = self.policy[draw, y, b, :3]
                p_go=p_y + p_b
                if act == 2:  # wait
                    total_ll += np.log(p_w + eps)
                elif act == 0 or act == 1:  # choose yellow or blue
                    total_ll += np.log((p_go) + eps)
                    break  # I need this break, because I have the full sequence and subjects can decide but still see the pre-assigned sequence yet.

        return total_ll

    # Fitting Functions
    def log_likelihood_extended(self, data: "pd.DataFrame") -> list:
        """
        Same walk as log_likelihood, but instead of a single summed value, returns
        one record per trial with the full policy breakdown, for diagnostics/plotting.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column,
                as in log_likelihood.

        Returns:
            list: One [draw, y, b, p_y, p_b, p_w, p_chosen, action, ll, cum_ll] row per
                trial, where cum_ll is the running log-likelihood within that trial.
        """
        total_ll = []
        ll = 0
        cum_ll = 0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        eps = 1e-10
        n_draw = 0
        for action_seq in actions_col:

            for draw, y, b, act, outcome in action_seq:
                p_y, p_b, p_w = self.policy[draw, y, b, :3]
                if act == 2:  # wait
                    ll = np.log(p_w + eps)
                    cum_ll = cum_ll + ll

                    total_ll.append([draw, y, b, p_y, p_b, p_w, p_w, act, ll, cum_ll])
                elif act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    ll = np.log(p_action + eps)
                    cum_ll = cum_ll + ll

                    total_ll.append(
                        [draw, y, b, p_y, p_b, p_w, p_action, act, ll, cum_ll]
                    )
                    break  # I need this break, because I have the full sequence and subjects can decide but still see the pre-assigned sequence yet.

        return total_ll

    def make_cost_function(self, data: dict) -> "Callable[[list], float]":
        """
        Build a cost function suitable for the optimizers below (ga_fit /
        fit_differential_evolution), which re-initializes this POMDP with the
        candidate params for each horizon key in data and returns -log-likelihood.

        Args:
            data (dict): Maps horizon_condition ("short"/"long") to the DataFrame of
                trials for that horizon (see log_likelihood for the expected schema).

        Returns:
            Callable[[list], float]: cost_function(params), where params is a flat
                list ordered per PARAM_ORDER; returns 1e10 on any exception.
        """

        def cost_function(params):
            # separate the dataset into two, and return each according to their separation. Assume that the data is separated with label long and short
            # I can have a disctionary of the two data data_dict['short']= evidence_to_fit and the other one and then
            try:
                # extract the keys from the evidence_to_fit_dict
                keys = data.keys()
                # loop over the keys
                # the key is either 'long' or 'short'
                ll = 0
                params = {k: v for k, v in zip(PARAM_ORDER, params)}
                # check if start_hazard is one of the keys of the Param_order, if so, make corresponding value to be intger.
                if "start_hazard" in params:
                    params["start_hazard"] = int(np.floor(params["start_hazard"]))

                params.update(
                    {
                        "verbose": VERBOSE,
                        "max_cards_per_draw": MAX_CARDS_PER_DRAW,
                    }
                )
                # loop over the keys and calculate the log likelihood for each key
                _ll_fn = self.log_likelihood_commit if POMDP_COMMIT else self.log_likelihood
                for key in keys:
                    params.update({"horizon_condition": key})
                    # re-initialize the class here.
                    self.__init__(**params)
                    self.value_iteration()
                    data_single_horizon = data[key]
                    ll += _ll_fn(
                        data_single_horizon
                    )  # instead of that one return, I can return the sum of two
                cost = -ll

                return cost
            except Exception as e:
                print(f"[Cost Error] {e}")
                return 1e10

        return cost_function

    @staticmethod
    def ga_fit(param_ranges: dict, cost_function: "Callable[[list], float]") -> tuple:
        """
        Fit parameters by minimizing cost_function with a genetic algorithm
        (geneticalgorithm.geneticalgorithm).

        Args:
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
                Order determines the parameter vector order passed to cost_function.
            cost_function (Callable[[list], float]): Function to minimize, e.g. one
                built by make_cost_function.

        Returns:
            tuple: (best_params_dict, best_fitness, hessian_placeholder), where
                hessian_placeholder is a zero matrix (GA does not estimate a Hessian).
                start_hazard, if present, is floored to an int.
        """
        varbound = np.array([param_ranges[k] for k in param_ranges.keys()])
        algorithm_param = get_ga_params()
        # variable_type=["real"]*(len(PARAM_ORDER)-1)
        # variable_type=["real"]*(len(PARAM_ORDER))
        # variable_type.append("int")

        model = ga(
            function=cost_function,
            dimension=len(param_ranges.keys()),
            convergence_curve=False,
            progress_bar=False,
            variable_type_mixed=np.array(variable_type),
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param,
        )

        model.run()

        best_params = model.output_dict["variable"]
        best_fitness = model.output_dict["function"]
        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        if "start_hazard" in best_params_dict.keys():
            best_params_dict["start_hazard"] = int(
                np.floor(best_params_dict["start_hazard"])
            )
        return (
            best_params_dict,
            best_fitness,
            np.zeros((len(param_ranges.keys()), len(param_ranges.keys()))),
        )

    @staticmethod
    def fit_differential_evolution(
        param_ranges: dict, cost_function: "Callable[[list], float]", x0=None
    ) -> tuple:
        """
        Fit parameters by minimizing cost_function with scipy's differential evolution.

        Args:
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
            cost_function (Callable[[list], float]): Function to minimize, e.g. one
                built by make_cost_function.

        Returns:
            tuple: (best_params_dict, best_fitness, hessian_placeholder), where
                hessian_placeholder is a 5x5 zero matrix (not an actual Hessian estimate).
        """
        bounds = [param_ranges[k] for k in param_ranges.keys()]

        result = run_differential_evolution(param_ranges, cost_function, x0=x0)
        best_params = result.x
        best_fitness = result.fun
        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        return best_params_dict, best_fitness, np.zeros((5, 5))

    def fit_subject(
        self,
        df_ev_simulated: dict,
        param_ranges: dict,
        subject_ID,
        algorithm: str,
    ) -> tuple:
        """
        Fit this POMDP's free parameters to one subject's data using the requested
        optimizer, then store the result on self (best_params, log_likelihood_values,
        subject_ID) in addition to returning it.

        Args:
            df_ev_simulated (dict): Per-horizon trial data, passed through to
                make_cost_function (see log_likelihood for the expected schema).
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
            subject_ID: Identifier for the subject/dataset being fit, echoed back in
                the return value.
            algorithm (str): "ga" for genetic algorithm or "de" for differential evolution.

        Returns:
            tuple: (best_params, log_likelihood_values, df_ev_simulated, subject_ID,
                hessian_matrix).
        """
        cost_function = self.make_cost_function(df_ev_simulated)
        if algorithm == "ga":
            best_params, best_cost, hessian_matrix = self.ga_fit(
                param_ranges, cost_function
            )
        if algorithm == "de":
            best_params, best_cost, hessian_matrix = self.fit_differential_evolution(
                param_ranges, cost_function, x0=de_seed_for(subject_ID, param_ranges)
            )
        log_likelihood_values = -best_cost
        self.best_params = best_params
        self.log_likelihood_values = log_likelihood_values
        self.subject_ID = subject_ID

        return (
            best_params,
            log_likelihood_values,
            df_ev_simulated,
            subject_ID,
            hessian_matrix,
        )

        # Plotting Functions:

    def plot_best_actions(
        self,
        label: str | None = None,
        path: str = "figures/illustrate",
        title: str | None = None,
        figsize: tuple[float, float] = (14, 8),
        font_size: float | None = None,
        ytick_step: int = 5,
        exact_size: bool = False,
        xlabel: str = "Number of Draws",
        ylabel: str = "Yellow - Blue Difference",
        ax=None,
        colorbar: bool = True,
    ) -> None:
        """
        Plots the best actions heatmap, filling internal gaps for a solid look.

        Args:
            label (str, optional): Label for the figure.
            path (str, optional): Path to save the figures.
            figsize (tuple, optional): Figure size in inches. Set this to the
                size the panel is actually printed at, so LaTeX includes it at
                scale 1; otherwise the panel is scaled on the page and its text
                shrinks by the same factor.
            font_size (float, optional): Point size for title, axis labels,
                ticks and colourbar. Meaningful together with a printed-size
                `figsize`, where it is the size that appears on the page.
                Defaults to the historical large sizes.
            ytick_step (int, optional): Spacing of the y-axis ticks in cards.
                A small panel cannot carry a tick every 5 cards without the
                labels colliding, so widen this as the printed size shrinks.
            exact_size (bool, optional): Save the figure at exactly `figsize`
                rather than cropped to its content. Cropping shrinks the file
                below the requested width, so LaTeX scales it back up to
                \linewidth and the text no longer prints at `font_size`.
            xlabel, ylabel (str, optional): Axis labels. At a small printed size
                the default wording costs more of the panel than the heatmap can
                spare, so a tiled panel should pass compact forms.
            ax (matplotlib Axes, optional): Draw into this axes instead of
                creating a figure. Nothing is saved, and the masked array is
                returned so a caller tiling several policies into one figure can
                share a single colourbar between them.
            colorbar (bool, optional): Draw the per-panel colourbar. Turn this
                off when the panels share one.
            title (str, optional): Panel title naming the manipulation. Passed
                through to plot_best_actions_symm; without it the panels of a
                multi-panel figure cannot be told apart.

        Returns:
            None
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        num_draws, num_yellow, num_blue = self.best_actions.shape
        max_diff = num_yellow - 1
        min_diff = -max_diff

        diff_range = np.arange(min_diff, max_diff + 1)
        adjusted_best_actions = np.full((num_draws, len(diff_range)), np.nan)

        for draw in range(num_draws):
            for yellow in range(num_yellow):
                blue = draw * self.max_cards_per_draw - yellow
                if blue < 0 or blue >= num_blue:
                    continue
                diff = yellow - blue
                if min_diff <= diff <= max_diff:
                    diff_index = diff - min_diff
                    adjusted_best_actions[draw, diff_index] = self.best_actions[
                        draw, yellow, blue
                    ]

        # Skip draw 0 (no cards drawn yet)
        adjusted_best_actions = adjusted_best_actions[1:, :]

        # --- NEW LOGIC: Fill internal gaps using nearest neighbor ---
        for i in range(adjusted_best_actions.shape[0]):
            row = adjusted_best_actions[i, :]

            # Identify valid elements (not NaN and not 5)
            valid_mask = ~np.isnan(row) & (row != 5)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            # Find boundaries to preserve the triangular empty space outside the data
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]

            # Extract the active vertical segment for this draw
            segment = row[first_valid : last_valid + 1]
            seg_valid_mask = ~np.isnan(segment) & (segment != 5)

            seg_valid_idx = np.where(seg_valid_mask)[0]
            seg_invalid_idx = np.where(~seg_valid_mask)[0]

            if len(seg_invalid_idx) > 0 and len(seg_valid_idx) > 0:
                distances = np.abs(seg_invalid_idx[:, None] - seg_valid_idx)
                for k_pos, k in enumerate(seg_invalid_idx):
                    min_d = distances[k_pos].min()
                    tied = seg_valid_idx[distances[k_pos] == min_d]
                    tied_actions = segment[tied]
                    if np.all(tied_actions == tied_actions[0]):
                        # Neighbours agree — fill normally
                        segment[k] = tied_actions[0]
                    elif 2 in tied_actions:
                        # Gap sits on a wait/decide boundary — extend wait
                        # symmetrically rather than biasing toward lower index
                        segment[k] = 2
                    # else: gap between two different decide actions (e.g. the
                    # very last draw with no wait states); handled below

                # Second pass: fill any still-NaN cells (edge cases where neither
                # equidistant neighbour was wait) with plain nearest-neighbour so
                # no reachable cell inside the valid range is left blank.
                still_nan = np.where(np.isnan(segment))[0]
                if len(still_nan) > 0:
                    filled_valid = np.where(~np.isnan(segment))[0]
                    if len(filled_valid) > 0:
                        dist2 = np.abs(still_nan[:, None] - filled_valid)
                        seg2 = filled_valid[np.argmin(dist2, axis=1)]
                        segment[still_nan] = segment[seg2]

            # Reinsert the cleanly filled segment
            adjusted_best_actions[i, first_valid : last_valid + 1] = segment
        # ------------------------------------------------------------

        # Mask out remaining invalid entries (the outer regions)
        mask_nan = np.isnan(adjusted_best_actions)
        mask_eq5 = adjusted_best_actions == 5
        mask = np.logical_or(mask_nan, mask_eq5)
        adjusted_best_actions = np.ma.array(adjusted_best_actions, mask=mask)

        # Assumes _set_plot_style() is defined elsewhere in your class/file
        # self._set_plot_style()

        # Custom colormap: Yellow=0, Blue=1, Green=2
        action_cmap = ListedColormap(["yellow", "blue", "green"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = BoundaryNorm(bounds, action_cmap.N)

        standalone = ax is None
        if standalone:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig = ax.figure

        heatmap = ax.imshow(
            adjusted_best_actions.T,
            cmap=action_cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )
        # Colorbar legend displayed in a different order (Blue, Wait, Yellow
        # from bottom to top) than the actual action codes used in the heatmap.
        legend_cmap = ListedColormap(["blue", "green", "yellow"])
        legend_mappable = plt.cm.ScalarMappable(cmap=legend_cmap, norm=norm)
        if colorbar:
            cbar = fig.colorbar(legend_mappable, ax=ax, ticks=[0, 1, 2])
            cbar.set_ticklabels(["Blue", "Wait", "Yellow"])
            if font_size is not None:
                cbar.ax.tick_params(labelsize=font_size)

        _fs = {"fontsize": font_size} if font_size is not None else {}
        if title:
            ax.set_title(title, fontsize=font_size if font_size is not None else 20)
        ax.set_xlabel(xlabel, **_fs)
        ax.set_ylabel(ylabel, **_fs)
        if font_size is not None:
            ax.tick_params(labelsize=font_size)

        num_draws_plot = adjusted_best_actions.shape[0]
        ax.set_xticks(np.arange(0, num_draws_plot, 1))
        ax.set_xticklabels(np.arange(1, num_draws_plot + 1, 1))

        # Match y-axis style: multiples of 5, centered at 0
        ytick_idx = [i for i, val in enumerate(diff_range) if val % ytick_step == 0]
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels(diff_range[ytick_idx])

        # ax.grid(which="major", color="w", linestyle="-", linewidth=0.5)
        # Turn off the major grid
        ax.grid(False)

        # Create minor ticks shifted by -0.5 to land exactly on the pixel edges
        ax.set_xticks(np.arange(-0.5, num_draws_plot, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(diff_range), 1), minor=True)

        # Draw the grid only on these minor ticks
        # ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

        # Hide the actual minor tick marks so they don't clutter the axes
        ax.tick_params(which="minor", bottom=False, left=False)
        if not standalone:
            return adjusted_best_actions, legend_mappable
        # tight_layout fits the content inside the canvas; the tight *bbox* at
        # save time instead shrinks the canvas to the content, which is what
        # breaks the printed size.
        fig.tight_layout()

        base = f"{path}/best_actions_heatmap"

        if label is not None:
            base += f"_{label}"

        save_kw = {} if exact_size else dict(bbox_inches="tight", pad_inches=0.03)
        fig.savefig(base + ".pdf", **save_kw)
        fig.savefig(base + ".svg", **save_kw)
        fig.savefig(base + ".png", dpi=600, **save_kw)

    def plot_policy(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
        continuous_policy: bool = False,
    ) -> None:
        """
        Plot self.policy as one heatmap per action (Yellow/Blue/Wait), on the
        symmetric (yellow-blue difference) grid used by plot_best_actions.

        Args:
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).
            continuous_policy (bool, optional): If True, fill internal NaN gaps
                (checkerboard cells from the diagonal grid) with the nearest
                valid neighbour(s), the same way plot_best_actions does, so
                there are no white cells in between valid ones. Defaults to
                False (original behaviour, gaps left blank).

        Returns:
            None. Figures are shown (and optionally saved); the mapping of
            action_label -> masked array is currently not returned (see commented-out
            return at the end of this method).
        """
        if self.policy.ndim != 4:
            raise ValueError(
                "policy must have shape (num_draws, num_yellow, num_blue, n_actions)"
            )

        num_draws, num_yellow, num_blue, n_actions = self.policy.shape
        # action_labels = [f"Action {i}" for i in range(n_actions)]
        action_labels = {0: "Yellow", 1: "Blue", 2: "Wait"}

        max_diff = num_yellow - 1
        min_diff = -max_diff
        diff_range = np.arange(min_diff, max_diff + 1)
        n_diffs = len(diff_range)

        # build adjusted probability arrays per action
        adjusted_probs = np.full((n_actions, num_draws, n_diffs), np.nan, dtype=float)

        for draw in range(num_draws):
            for yellow in range(num_yellow):
                blue = draw * self.max_cards_per_draw - yellow
                if blue < 0 or blue >= num_blue:
                    continue
                diff = yellow - blue
                if min_diff <= diff <= max_diff:
                    diff_index = diff - min_diff
                    for a in range(n_actions):
                        adjusted_probs[a, draw, diff_index] = self.policy[
                            draw, yellow, blue, a
                        ]

        if continuous_policy:
            # Fill internal gaps (checkerboard NaNs from the diagonal grid)
            # the same way plot_best_actions does, so there are no white
            # cells in between valid ones.
            for a in range(n_actions):
                _fill_diagonal_gaps(adjusted_probs[a])

        # mask invalid entries (NaNs) outside the active triangular region
        masks = np.isnan(adjusted_probs)
        masked = np.ma.array(adjusted_probs, mask=masks)

        # custom colormap
        _set_plot_style()

        # one figure per action
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))  # same size for all figures

            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Probability")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Yellow - Blue Difference")
            ax.set_xlabel("Number of Draws")

            ax.set_xticks(np.arange(0, num_draws, max(1, num_draws // 10)))
            # show only diffs that are multiples of 5 to reduce clutter
            indices_step_5 = [i for i, val in enumerate(diff_range) if val % 5 == 0]
            ax.set_yticks(indices_step_5)
            ax.set_yticklabels(diff_range[indices_step_5])

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)

            plt.tight_layout()

            if save_fig:
                os.makedirs(path, exist_ok=True)

                base = f"{path}/policy_heatmap_{action_labels[a]}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        # return {action_labels[i]: masked[i] for i in range(n_actions)}

    def plot_action_values(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot self.action_values (Q-values, not probabilities) as one heatmap per
        action (Yellow/Blue/Wait), on the symmetric (yellow-blue difference) grid
        used by plot_best_actions.

        Args:
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            None. Figures are shown (and optionally saved); the mapping of
            action_label -> masked array is currently not returned (see commented-out
            return at the end of this method).
        """

        if self.action_values.ndim != 4:
            raise ValueError(
                "policy must have shape (num_draws, num_yellow, num_blue, n_actions)"
            )

        num_draws, num_yellow, num_blue, n_actions = self.action_values.shape
        # action_labels = [f"Action {i}" for i in range(n_actions)]
        action_labels = {0: "Yellow", 1: "Blue", 2: "Wait"}

        max_diff = num_yellow - 1
        min_diff = -max_diff
        diff_range = np.arange(min_diff, max_diff + 1)
        n_diffs = len(diff_range)

        # build adjusted probability arrays per action
        adjusted_probs = np.full((n_actions, num_draws, n_diffs), np.nan, dtype=float)

        for draw in range(num_draws):
            for yellow in range(num_yellow):
                blue = draw * self.max_cards_per_draw - yellow
                if blue < 0 or blue >= num_blue:
                    continue
                diff = yellow - blue
                if min_diff <= diff <= max_diff:
                    diff_index = diff - min_diff
                    for a in range(n_actions):
                        adjusted_probs[a, draw, diff_index] = self.action_values[
                            draw, yellow, blue, a
                        ]

        # mask invalid entries (NaNs)
        masks = np.isnan(adjusted_probs)
        masked = np.ma.array(adjusted_probs, mask=masks)
        _set_plot_style()
        # custom colormap
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))  # same size for all figures

            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=np.min(self.action_values),
                vmax=np.max(self.action_values),
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Action Values")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Yellow - Blue Difference")
            ax.set_xlabel("Number of Draws")

            ax.set_xticks(np.arange(0, num_draws, max(1, num_draws // 10)))
            # show only diffs that are multiples of 5 to reduce clutter
            indices_step_5 = [i for i, val in enumerate(diff_range) if val % 5 == 0]
            ax.set_yticks(indices_step_5)
            ax.set_yticklabels(diff_range[indices_step_5])

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)

            plt.tight_layout()

            if save_fig:
                os.makedirs(path, exist_ok=True)

                base = f"{path}/action_values_heatmap_{action_labels[a]}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        # return {action_labels[i]: masked[i] for i in range(n_actions)}

    @staticmethod
    def get_stochastic_action_depending_on_policy(
        policy_at_current_draw: np.ndarray,
    ) -> int:
        """
        Sample an action from the policy distribution at the current state.

        Args:
            policy_at_current_draw (np.ndarray): Length-3 vector of action
                probabilities [P(Yellow), P(Blue), P(Wait)] summing to 1.

        Returns:
            int: Sampled action, one of {0: Yellow, 1: Blue, 2: Wait}.
        """
        actions = [0, 1, 2]
        chosen_action = np.random.choice(actions, p=policy_at_current_draw)
        return chosen_action

    def simulate_cards_pomdp(
        self,
        given_sequence: bool = False,
        card_sequence: list | None = None,
        start_hazard: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """
        Simulate one trial of the card task: generate (or take a given) sequence of
        yellow/blue draws, then step through it sampling actions from self.policy
        (or self.belief for the interpolated subclasses) until a Yellow/Blue choice
        is made, the deadline is reached, or the hazard-truncated sequence runs out.

        Args:
            given_sequence (bool): If True, replay card_sequence instead of generating
                a random one.
            card_sequence (list, optional): Required when given_sequence is True; a
                sequence of (draw, cum_yellow, cum_blue, action, outcome) tuples as
                recorded from the real experiment.
            start_hazard (int, optional): Draw at which the hazard function starts,
                for the randomly-generated case. Defaults to self.start_hazard.
            seed (int, optional): Seed for np.random, for reproducible simulation.

        Returns:
            dict: Belief trajectory, actions taken, reward, decision, num_draws,
                num_draws_list, max_draws (the realized deadline), and the observed
                yellow/blue counts (both up to the decision and for the full sequence).
        """
        # initalize the lists, and variables
        num_yellows = []
        num_blues = []
        yellow_trace = []
        blue_trace = []
        belief_trajectory = []
        actions = []
        num_draws = 1
        num_draws_list = []
        num_yellow_observed = []

        # Initializing the cards sequence for the case that no sequence is given. This is intialized randomly
        correct_decision = None

        np.random.seed(seed)
        if not given_sequence:
            # Draw a number to determine the number of draws
            u = np.random.random()

            # Here I need to recalculate the hazard even though it could've been calculated in the class.
            # the reason is that, I need the simulate_cards always to be hazardous and random at each draw.
            # but the one in the class is fixed. So that's why I should separate the generation of cards here from
            # what was previously calculated in the class.
            # also no matter whether is_hazardous is true or false, I need to have a hazardous generation of cards.
            # to match the real experiment.

            # first, I could have the start hazard as a free parameter in the future to be fitted.
            if start_hazard is None:
                start_hazard = self.start_hazard

            # but the end_hazard is always the same, i.e., the max draws depending on the horizon condition, long or short
            hazard = calculate_hazard_cum(int(start_hazard), self.max_draws)

            # now this is to calculate the actual deadline that could be smaller than the max_draws
            deadline = np.searchsorted(hazard, u, side="right")
            # I'm generating the trace now like tobias's one

            p_yellow = np.random.uniform(0.4, 0.6)
            while p_yellow == 0.5:
                p_yellow = np.random.uniform(0.4, 0.6)

            yellow_trace = np.random.binomial(
                n=self.max_cards_per_draw, p=p_yellow, size=deadline
            )
            blue_trace = self.max_cards_per_draw - yellow_trace

            # according to the max number of draws, i.e., the deadline, I generate the yellow and blue cards here.

            # yellow_trace = np.array(
            #     [
            #         np.random.randint(0, self.max_cards_per_draw + 1)
            #         for _ in range(deadline)
            #     ]
            # )
            # blue_trace = self.max_cards_per_draw - yellow_trace

            # I just insert zeros at the beginning, so that when I calculate the cumsum, I have a full list of cards.
            yellow_trace = np.insert(yellow_trace, 0, 0)
            blue_trace = np.insert(blue_trace, 0, 0)

            # the belief and policy is always w.r.t. total evidence. So I need the num_yellows, num_blues, total
            num_yellows = np.cumsum(yellow_trace)
            num_blues = np.cumsum(blue_trace)
            correct_decision = 0 if p_yellow > 0.5 else 1

        elif given_sequence:
            # this randomize seed, because if I called the first if within the same code run,
            # the seed will be fixed always, so I just need to re randomize it again here.
            cum_y = np.array([trial[1] for trial in card_sequence], dtype=int)
            cum_b = np.array([trial[2] for trial in card_sequence], dtype=int)

            # add initial 0 to match the other branch (so traces start at draw 0)
            num_yellows = np.insert(cum_y, 0, 0)
            num_blues = np.insert(cum_b, 0, 0)

            # number of draws
            deadline = len(num_blues) - 1
            # extract the reward from the last trial in the sequence, which is the outcome of the decision
            real_reward = card_sequence[-1][-1]
            real_action = card_sequence[-1][-2]
            if real_reward == 2:
                correct_decision = real_action  # the action taken in the last trial
            elif real_reward == -2:
                correct_decision = 1 if real_action == 0 else 0
            else:
                # in case of missing here, the correct action is calculated based on the total number of yellow and total number of blue because I don't know the generative probability.
                correct_decision = 0 if num_yellows[-1] > num_blues[-1] else 1

        num_yellow_observed.append(num_yellows[num_draws])
        num_draws_list.append(num_draws)
        belief_trajectory.append(
            [self.belief[num_draws, num_yellows[num_draws], num_blues[num_draws], 0]]
        )  # belief of yellow
        action = self.get_stochastic_action_depending_on_policy(
            self.policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],
        )
        # print(policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],num_draws,num_yellows[num_draws], num_blues[num_draws],action)
        actions.append(int(action))

        # ensure that the first action is always 2 (wait), otherwise raise an error with zero draws
        while action == 2 and num_draws < (len(num_yellows) - 1):
            num_draws += 1
            num_yellow_observed.append(num_yellows[num_draws])
            num_draws_list.append(num_draws)
            belief_trajectory.append(
                [
                    self.belief[
                        num_draws, num_yellows[num_draws], num_blues[num_draws], 0
                    ]
                ]
            )  # belief of yellow
            action = self.get_stochastic_action_depending_on_policy(
                self.policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],
            )
            # print(policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],num_draws,num_yellows[num_draws], num_blues[num_draws],action)
            actions.append(int(action))

        belief_trajectory = np.array(belief_trajectory)
        if action == int(2):
            reward = -1  # missed the deadline
        elif action == correct_decision:
            reward = 2
        elif action != correct_decision:
            reward = -2

        results = {
            "trajectory": belief_trajectory,
            "reward": reward,
            "actions": actions,
            "decision": actions[-1],
            "num_draws": num_draws,
            "num_draws_list": num_draws_list,
            "max_draws": deadline,
        }
        results["num_yellows"] = num_yellows[1 : num_draws + 1]
        results["num_blues"] = num_blues[1 : num_draws + 1]
        results["num_yellows_full_sequence"] = num_yellows[1:]
        results["num_blues_full_sequence"] = num_blues[1:]

        return results


class POMDP_Urgency(POMDP):
    """
    POMDP with an added sigmoidal "urgency" cost on the Wait action (see
    sigmoidal_cost), on top of the base class's hazard/deadline mechanics. Also
    supports overriding the Beta-prior belief bias (belief_bias).
    """

    def __init__(
        self,
        tau: float = TAU,
        xi: float = XI,
        hazard_lapse: float = HAZARD_LAPSE,
        horizon_condition: str = HORIZON_CONDITION,
        is_hazardous: bool = IS_HAZARDOUS,
        subjective_cost: float = SUBJECTIVE_COST,
        verbose: bool = VERBOSE,
        max_cards_per_draw: int = MAX_CARDS_PER_DRAW,
        patience: float = PATIENCE,
        urgency_coefficient: float = URGENCY_COEFFICIENT,
        urgency_slope: float = URGENCY_SLOPE,
        c_max: float = C_MAX,
        sweetspot_tau: float = SWEETSPOT_TAU,
        sweetspot_xi: float = SWEETSPOT_XI,
        belief_bias: float = BELIEF_BIAS,
        start_hazard: int | None = None,
    ):
        """
        Initialize a POMDP that adds a sigmoidal "urgency" cost to waiting, on top
        of the base POMDP's hazard/deadline mechanics.

        Args:
            tau (float): Softmax temperature for the action policy.
            xi (float): Lapse rate mixed into the Yellow/Blue softmax probabilities.
            hazard_lapse (float): Mixing weight between short/long hazard curves (see POMDP.__init__).
            horizon_condition (str): "short" or "long"; sets max_draws and the default start_hazard.
            is_hazardous (bool): Whether the hazard (deadline) function is applied during planning.
            subjective_cost (float): Added to the incorrect-choice reward.
            verbose (bool): Whether to print progress information.
            max_cards_per_draw (int): Number of cards revealed per draw.
            patience (float): Midpoint (inflection point) of the urgency sigmoid, in draws.
            urgency_coefficient (float): Lower bound (L) of the urgency sigmoid cost.
            urgency_slope (float): Steepness (k) of the urgency sigmoid; sign controls direction.
            c_max (float): Upper bound (U) of the urgency sigmoid cost.
            sweetspot_tau (float): Currently unused (kept for the commented-out
                per-draw-tau variant of softmax_policy below).
            sweetspot_xi (float): Currently unused (kept for the commented-out
                per-draw-xi variant of softmax_policy below).
            belief_bias (float): Overrides self.beta (the Beta-prior pseudo-count for blue),
                biasing the belief posterior independent of subjective_cost.
            start_hazard (int, optional): Draw at which the hazard function starts.
                Defaults to 4 (short) or 10 (long) when None.
        """

        super().__init__(
            tau=tau,
            xi=xi,
            horizon_condition=horizon_condition,
            hazard_lapse=hazard_lapse,
            is_hazardous=is_hazardous,
            verbose=verbose,
            max_cards_per_draw=max_cards_per_draw,
            subjective_cost=subjective_cost,
            start_hazard=start_hazard,
        )
        self.belief_bias = belief_bias
        self.beta = self.belief_bias
        self.sweetspot_xi = sweetspot_xi
        self.sweetspot_tau = sweetspot_tau

        # parameters for the urgency sigmoid function
        self.urgency_coefficient = urgency_coefficient
        self.urgency_slope = urgency_slope
        self.patience = patience
        self.c_max = c_max

    def calculate_belief_probability(
        self, num_blues: int, num_yellows: int, q: float = 0.5
    ) -> float:
        """
        Calculate the probability that the underlying generative probability q_y > q,
        given num_yellows and num_blues, using a Beta posterior.

        Note: identical to POMDP.calculate_belief_probability; self.beta already
        reflects belief_bias via __init__, so this override is currently redundant.

        Args:
            num_blues (int): Number of blue cards drawn.
            num_yellows (int): Number of yellow cards drawn.
            q (float): Threshold probability to compare against (default is 0.5).

        Returns:
            float: Posterior probability that the generative probability q_y > q.
        """

        Alpha = self.alpha + num_yellows
        Beta_ = self.beta + num_blues

        posterior_prob = 1 - beta.cdf(q, Alpha, Beta_)
        return posterior_prob

    def sigmoidal_cost(self, num_draws: int | np.ndarray) -> float | np.ndarray:
        """
        Generalized sigmoid cost, added to the "wait" action value to make waiting
        increasingly (or decreasingly) costly as more draws elapse.

        cost(x) = L + (U - L) / (1 + exp(-k * (x - x0)))

        Args:
            num_draws (int or np.ndarray): Current draw count (x), scalar or array.
                Uses self.urgency_slope (k, steepness; positive = increasing,
                negative = decreasing), self.patience (x0, midpoint/inflection),
                self.urgency_coefficient (L, lower bound), and self.c_max (U, upper bound).

        Returns:
            float or np.ndarray: The urgency cost at num_draws, same shape as input.
        """
        k = self.urgency_slope  # controls steepness
        p = self.patience  # midpoint (inflection point)
        L = self.urgency_coefficient  # independent minimum
        U = self.c_max  # independent maximum

        return L + (U - L) / (1 + np.exp(-k * (num_draws - p)))

    def value_iteration(self) -> None:
        """
        Perform value iteration using vectorized operations for the inner loops.

        Returns:
            None. Populates self.action_values, self.value_function, self.policy,
            self.transition_probability, and (via extract_actions) self.best_actions.
        """
        # Initialize terminal state values
        self.calculate_action_values_for_Y_and_B()

        self.action_values[-1, :, :, 2] = self.deadline_cost + self.sigmoidal_cost(
            self.max_draws
        )
        policy_probs = self.softmax_policy(
            self.action_values[-1, :, :, :], self.max_draws
        )
        # The policy at the terminal state is the softmax over the three Q-values.
        self.policy[-1, :, :, :] = policy_probs

        # The value of a terminal state V(s) is the maximum of all possible action values from that state.
        self.value_function[-1, :, :] = np.sum(
            policy_probs * self.action_values[-1, :, :, :], axis=-1
        )

        for draw in range(self.max_draws - 1, -1, -1):
            # --- 1. Define the state space for the current draw ---
            max_possible_yellows = draw * self.max_cards_per_draw
            current_yellows = np.arange(max_possible_yellows + 1)
            current_blues = draw * self.max_cards_per_draw - current_yellows

            # --- 2. Calculate the expected value of the next state ---

            # Create broadcastable arrays for outcomes of the next draw
            # next_yellows has shape (N_outcomes,)
            next_yellows_arr = np.array(self.next_yellow)
            next_blues_arr = self.max_cards_per_draw - next_yellows_arr

            # Use broadcasting to find all future states from all current states
            future_yellows = (
                current_yellows[:, np.newaxis] + next_yellows_arr[np.newaxis, :]
            )
            future_blues = current_blues[:, np.newaxis] + next_blues_arr[np.newaxis, :]

            # Get the action values at all possible future states
            # Result has shape (N_states, N_outcomes, N_actions)
            action_values_next = self.action_values[
                draw + 1, future_yellows, future_blues
            ]

            # Apply softmax policy and calculate the expected value for each future state
            # Assuming self.softmax_policy is vectorized or can be replaced by a vectorized version
            policy_probs = self.softmax_policy(
                action_values_next, draw + 1, axis=-1
            )  # axis=-1 applies softmax over actions
            expected_future_value = np.sum(
                policy_probs * action_values_next, axis=-1
            )  # Shape: (N_states, N_outcomes)

            # Store the calculated value function and policy
            self.value_function[draw + 1, future_yellows, future_blues] = (
                expected_future_value
            )
            self.policy[draw + 1, future_yellows, future_blues] = policy_probs

            # --- 3. Calculate transition probabilities ---
            transitions = self.calculate_transition_probability(
                i=next_yellows_arr[np.newaxis, :],
                num_yellows=current_yellows[:, np.newaxis],
                num_blues=current_blues[:, np.newaxis],
            )  # Shape: (N_states, N_outcomes)
            self.transition_probability[draw, future_yellows, future_blues] = (
                transitions
            )
            # --- 4. Calculate the "wait" action value ---
            if self.is_hazardous:
                discount_factor = 1 - self.cum_hazard[draw]
                # Sum over the "outcomes" axis (axis=1)
                wait_value_update = discount_factor * np.sum(
                    transitions * expected_future_value, axis=1
                )
                total_wait_value = (
                    wait_value_update
                    + self.cum_hazard[draw] * self.deadline_cost
                    + self.sigmoidal_cost(draw)
                )
            else:

                # hazard plus urgency.
                wait_value_update = np.sum(
                    transitions * expected_future_value, axis=1
                )  #
                total_wait_value = wait_value_update + self.sigmoidal_cost(
                    draw
                )  # Add costs

            # --- 5. Update the action_values table for the current draw ---
            self.action_values[draw, current_yellows, current_blues, 2] = (
                total_wait_value
            )
            current_q_values = self.action_values[draw, current_yellows, current_blues]

            self.policy[draw, current_yellows, current_blues, :] = self.softmax_policy(
                current_q_values, draw
            )

        # Finally, extract the best actions based on the fully populated table
        self.extract_actions()


class POMDP_Forgetting(POMDP_Urgency):
    """
    POMDP_Urgency variant that replaces the integer belief/value grids with a
    continuous (step_size-spaced) yellow/blue grid queried via interpolation, and
    discounts ("forgets") accumulated evidence by gamma toward uncertainty when
    projecting to future states during value iteration.
    """

    def __init__(
        self,
        tau: float = TAU,
        xi: float = XI,
        horizon_condition: str = HORIZON_CONDITION,
        hazard_lapse: float = HAZARD_LAPSE,
        is_hazardous: bool = IS_HAZARDOUS,
        subjective_cost: float = SUBJECTIVE_COST,
        verbose: bool = VERBOSE,
        max_cards_per_draw: int = MAX_CARDS_PER_DRAW,
        patience: float = PATIENCE,
        urgency_coefficient: float = URGENCY_COEFFICIENT,
        urgency_slope: float = URGENCY_SLOPE,
        c_max: float = C_MAX,
        belief_bias: float = BELIEF_BIAS,
        gamma: float = GAMMA,
        start_hazard: int | None = None,
    ):
        """
        Initialize a POMDP_Urgency variant that replaces the integer belief/value
        grids with a continuous (step_size-spaced) yellow/blue grid queried via
        interpolation, and applies a "forgetting" discount of past evidence
        (gamma) toward uncertainty when projecting to future states in value_iteration.

        Args:
            tau (float): Softmax temperature for the action policy.
            xi (float): Lapse rate mixed into the Yellow/Blue softmax probabilities.
            horizon_condition (str): "short" or "long"; sets max_draws and the default start_hazard.
            hazard_lapse (float): Mixing weight between short/long hazard curves (see POMDP.__init__).
            is_hazardous (bool): Whether the hazard (deadline) function is applied during planning.
            subjective_cost (float): Added to the incorrect-choice reward.
            verbose (bool): Whether to print progress information.
            max_cards_per_draw (int): Number of cards revealed per draw.
            patience (float): Midpoint of the urgency sigmoid, in draws.
            urgency_coefficient (float): Lower bound of the urgency sigmoid cost.
            urgency_slope (float): Steepness of the urgency sigmoid.
            c_max (float): Upper bound of the urgency sigmoid cost.
            belief_bias (float): Overrides self.beta (the Beta-prior pseudo-count for blue),
                biasing the belief posterior independent of subjective_cost.
            gamma (float): Forgetting factor in [0, 1] applied to accumulated
                yellow/blue counts when projecting to the next draw's future state
                (1 = no forgetting).
            start_hazard (int, optional): Draw at which the hazard function starts.
                Defaults to 4 (short) or 10 (long) when None.
        """
        super().__init__(
            tau=tau,
            xi=xi,
            horizon_condition=horizon_condition,
            hazard_lapse=hazard_lapse,
            is_hazardous=is_hazardous,
            subjective_cost=subjective_cost,
            verbose=verbose,
            max_cards_per_draw=max_cards_per_draw,
            patience=patience,
            urgency_coefficient=urgency_coefficient,
            urgency_slope=urgency_slope,
            c_max=c_max,
            start_hazard=start_hazard,
        )

        self.belief_bias = belief_bias
        self.beta = self.belief_bias
        self.gamma = gamma

        self.max_yellows_val = self.max_cards_per_draw * self.max_draws
        self.max_blues_val = self.max_cards_per_draw * self.max_draws
        self.step_size = 0.2

        # --- Define grid axes values ---
        self.draw_axis = np.arange(0, self.max_draws + 1)

        self.yellow_axis = np.arange(
            0, self.max_yellows_val + self.step_size, self.step_size
        )
        self.blue_axis = np.arange(
            0, self.max_blues_val + self.step_size, self.step_size
        )

        # Get grid dimensions
        self.l_draws = len(self.draw_axis)
        self.n_yellows = len(self.yellow_axis)
        self.m_blues = len(self.blue_axis)

        # --- Initialize grid arrays ---
        self.action_values = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 3))
        self.belief = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 3))
        self.value_function = np.zeros((self.l_draws, self.n_yellows, self.m_blues))
        self.belief_grid = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 2))
        self.transition_probability = np.zeros(
            (self.l_draws, self.n_yellows, self.m_blues, len(self.next_yellow))
        )
        self.policy = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 3))
        self.best_actions = np.ones((self.l_draws, self.n_yellows, self.m_blues)) * 5

        # Fill the belief grid
        self.generate_belief_grid()

    def generate_belief_grid(self) -> None:
        """
        Populate the belief grid for all valid states.

        Returns:
            None. Sets self.belief_grid in place.
        """
        # Iterate over all draws, including draw 0
        for i_draw, draw in enumerate(self.draw_axis):
            if draw == 0:
                belief_val_0 = 1 - beta.cdf(0.5, self.alpha, self.beta)  # Should be 0.5
                self.belief_grid[0, 0, 0, 0] = belief_val_0
                self.belief_grid[0, 0, 0, 1] = 1 - belief_val_0
                continue

            max_cards = draw * self.max_cards_per_draw

            # Find indices of valid yellow and blue axes
            # valid numbers by checking y <= max_cards and b <= max_cards,
            valid_y_indices = np.where(self.yellow_axis <= max_cards)[0]
            valid_b_indices = np.where(self.blue_axis <= max_cards)[0]

            # Get the actual values
            valid_yellows = self.yellow_axis[valid_y_indices]
            valid_blues = self.blue_axis[valid_b_indices]

            if len(valid_yellows) == 0 or len(valid_blues) == 0:
                continue

            # Create a meshgrid
            grid_y, grid_b = np.meshgrid(valid_yellows, valid_blues, indexing="ij")

            # Calculate beliefs
            Alpha = self.alpha + grid_y
            Beta_ = self.beta + grid_b
            belief_vals = 1 - beta.cdf(0.5, Alpha, Beta_)

            # Get a meshgrid of indices to slice the belief_grid
            idx_y, idx_b = np.meshgrid(valid_y_indices, valid_b_indices, indexing="ij")

            # Assign beliefs to the correct grid locations
            # Only populate beliefs where total cards is valid
            # This is a bit inefficient but correct.
            total_cards_grid = grid_y + grid_b
            valid_state_mask = total_cards_grid <= (max_cards + 1e-5)  # Add tolerance

            # Apply mask to indices
            valid_idx_y = idx_y[valid_state_mask]
            valid_idx_b = idx_b[valid_state_mask]

            # Apply mask to calculated beliefs
            valid_beliefs = belief_vals[valid_state_mask]

            self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 0] = valid_beliefs
            self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 1] = 1 - valid_beliefs

    def get_interpolated_belief(
        self, draw: float, yellow: float, blue: float
    ) -> np.ndarray:
        """
        Interpolate belief_grid at (draw, yellow, blue) values.

        Args:
            draw (float): Draw index (may be off-grid; interpolated on draw_axis).
            yellow (float): Total yellow count (interpolated on yellow_axis).
            blue (float): Total blue count (interpolated on blue_axis).

        Returns:
            np.ndarray: Length-2 array [P(Yellow), P(Blue)]; NaN if out of grid bounds.
        """

        if not hasattr(self, "belief_grid") or self.belief_grid is None:
            raise RuntimeError("self.belief_grid is not available")

        # Define the axes for the interpolator
        axes = (self.draw_axis, self.yellow_axis, self.blue_axis)

        # Create interpolator for P(Yellow)
        interpolator_y = _RGI(
            axes, self.belief_grid[..., 0], bounds_error=False, fill_value=np.nan
        )

        # Query point must be 2D: (n_points, ndim)
        pt = np.array([[float(draw), float(yellow), float(blue)]])
        result_y = interpolator_y(pt)

        # P(Blue) = 1 - P(Yellow)
        result_b = 1.0 - result_y

        # Squeeze to remove the (1,) dimension and stack
        return np.array([np.squeeze(result_y), np.squeeze(result_b)])

    def softmax_policy(self, action_values: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax with lapse applied ONLY to first two actions.
        Same as POMDP.softmax_policy but without the (unused) draw parameter.

        Args:
            action_values (np.ndarray): Q-values (..., n_actions), with Wait as the last action.
            axis (int): Axis over which to apply the softmax (must index the actions dimension).

        Returns:
            np.ndarray: Action-probability array with the same shape as action_values.
        """
        # Standard, stable softmax
        max_val = np.max(action_values, axis=axis, keepdims=True)
        all_impossible_mask = np.isneginf(max_val)
        safe_max_val = np.where(all_impossible_mask, 0, max_val)
        # tau is a softmax denominator -- as it approaches 0 the division
        # blows up (divide-by-zero/overflow), so floor it for this division
        # only; self.tau itself (the fitted/reported parameter) is untouched.
        safe_tau = max(self.tau, 1e-3)
        scaled_values = (action_values - safe_max_val) / safe_tau

        exps = np.exp(scaled_values)
        denominator = np.sum(exps, axis=axis, keepdims=True)
        softmax_probs = np.nan_to_num(exps / denominator)

        # Save original last action
        probs_last = softmax_probs[..., -1]

        # Get total probability of first two before mixing
        probs_first_two_sum = np.sum(softmax_probs[..., :2], axis=axis, keepdims=True)

        # Mix the first two: new = (1-xi)*orig + xi/2, but RESCALE so their total is unchanged
        mixed_first_two = softmax_probs[..., :2] * (1 - self.xi) + self.xi / 2
        mixed_sum = np.sum(mixed_first_two, axis=axis, keepdims=True)

        # Scale factor
        scaling = np.divide(
            probs_first_two_sum,
            mixed_sum,
            out=np.ones_like(mixed_sum),
            where=(mixed_sum != 0),
        )
        mixed_first_two = mixed_first_two * scaling

        # Stack all together
        final_policy = np.concatenate(
            [mixed_first_two, probs_last[..., None]], axis=axis
        )

        return final_policy

    def calculate_transition_probability(
        self, i: np.ndarray, num_yellows: np.ndarray, num_blues: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the transition probability using the analytical beta-binomial formula.
        Vectorized to handle array inputs.

        Args:
            i (np.ndarray): Number(s) of yellow cards in the next draw (k).
            num_yellows (np.ndarray): Total number of yellow cards observed so far.
            num_blues (np.ndarray): Total number of blue cards observed so far.

        Returns:
            np.ndarray: Transition probability P(i yellow cards in next draw | num_yellows, num_blues).
        """
        alpha_post = self.alpha + num_yellows
        beta_post = self.beta + num_blues
        n = self.max_cards_per_draw

        with np.errstate(divide="ignore", invalid="ignore"):
            log_comb = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
            log_prob = (
                log_comb
                + betaln(alpha_post + i, beta_post + n - i)
                - betaln(alpha_post, beta_post)
            )

        probabilities = np.exp(log_prob)
        valid_mask = (i >= 0) & (i <= self.max_cards_per_draw)
        return np.where(valid_mask, np.nan_to_num(probabilities), 0.0)

    def extract_actions(self) -> None:
        """
        Populate self.best_actions with argmax(self.policy, axis=-1) per state.

        Note: unlike POMDP.extract_actions, this is a plain argmax with no
        tolerance-based tie-breaking (ties resolve to the lowest action index).

        Returns:
            None. Sets self.best_actions in place.
        """
        policy = self.policy
        self.best_actions = np.argmax(policy, axis=-1)

    def calculate_action_values(self) -> None:
        """
        Calculate the action values (Yellow, Blue) for all valid states in the grid.
        This does NOT calculate the 'Wait' action value.

        Returns:
            None. Sets self.action_values[..., 0], self.action_values[..., 1], and
            self.belief[..., 0:2] in place.
        """
        for i_draw, draw in enumerate(self.draw_axis):
            max_cards = draw * self.max_cards_per_draw

            # Find valid indices for this draw
            valid_y_indices = np.where(self.yellow_axis <= max_cards)[0]
            valid_b_indices = np.where(self.blue_axis <= max_cards)[0]

            if len(valid_y_indices) == 0 or len(valid_b_indices) == 0:
                continue

            # Create index meshgrid
            idx_y, idx_b = np.meshgrid(valid_y_indices, valid_b_indices, indexing="ij")

            # --- Filter for valid states (y + b <= max_cards) ---
            grid_y = self.yellow_axis[idx_y]
            grid_b = self.blue_axis[idx_b]
            valid_state_mask = (grid_y + grid_b) <= (max_cards + 1e-5)

            if not np.any(valid_state_mask):
                continue

            valid_idx_y = idx_y[valid_state_mask]
            valid_idx_b = idx_b[valid_state_mask]

            # --- Get beliefs from pre-calculated grid ---
            belief_yellow = self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 0]
            belief_blue = self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 1]

            # --- Reward Calculation (vectorized) ---
            reward_correct = self.calculate_reward(is_correct=True)
            reward_incorrect = self.calculate_reward(is_correct=False)

            # --- Action Value Calculation ---
            yellow_action_value = (
                reward_correct * belief_yellow + reward_incorrect * belief_blue
            )
            blue_action_value = (
                reward_correct * belief_blue + reward_incorrect * belief_yellow
            )

            # --- Assign to the action_values grid ---
            self.action_values[i_draw, valid_idx_y, valid_idx_b, 0] = (
                yellow_action_value
            )
            self.belief[i_draw, valid_idx_y, valid_idx_b, 0] = belief_yellow

            self.action_values[i_draw, valid_idx_y, valid_idx_b, 1] = blue_action_value
            self.belief[i_draw, valid_idx_y, valid_idx_b, 1] = belief_blue

    def get_interpolated_policy(
        self, draw: float, yellow: float, blue: float
    ) -> np.ndarray:
        """
        Interpolates the Q-Values (action_values) for a given state
        and returns the resulting softmax policy.

        Args:
            draw (float): The number of draws (e.g., 3.0).
            yellow (float): The number of yellow cards (e.g., 1.5).
            blue (float): The number of blue cards (e.g., 2.5).

        Returns:
            np.ndarray: A 1D array of shape (3,) containing the
                        probabilities for [Action Y, Action B, Action W].
        """

        # 1. Create the axes for the interpolator
        axes = (self.draw_axis, self.yellow_axis, self.blue_axis)

        # 2. Create the interpolator for the Q-values (self.action_values)
        # We interpolate Q-values, not policies, because interpolating
        # vectors on a simplex (policies) is mathematically problematic.
        # We fill with the deadline_cost for out-of-bounds states.
        q_interpolator = RegularGridInterpolator(
            axes, self.action_values, bounds_error=False, fill_value=self.deadline_cost
        )

        # 3. Define the query point
        pt = np.array([[float(draw), float(yellow), float(blue)]])

        # 4. Get the interpolated Q-values
        interpolated_q_values = q_interpolator(pt)  # Shape (1, 3)

        # 5. Squeeze to a 1D vector
        q_vector = np.squeeze(interpolated_q_values)  # Shape (3,)

        # 6. Apply the softmax policy function to the interpolated Q-values
        # This ensures the output is a valid policy (sums to 1).
        policy_vector = self.softmax_policy(q_vector, axis=-1)

        return policy_vector

    def value_iteration(self) -> None:
        """
        Perform value iteration using interpolation for non-grid future states.

        Returns:
            None. Populates self.action_values, self.value_function, self.policy,
            self.transition_probability, and (via extract_actions) self.best_actions.
        """
        # Calculate Q(s, Y) and Q(s, B) for all states
        self.calculate_action_values()

        # Set Q-value for 'wait' at the terminal step (index -1)
        i_terminal_draw = self.l_draws - 1

        self.action_values[i_terminal_draw, :, :, 2] = (
            self.deadline_cost + self.sigmoidal_cost(self.max_draws)
        )
        # Policy at terminal state (softmax over Q)
        terminal_q_values = self.action_values[i_terminal_draw, :, :, :]
        terminal_policy = self.softmax_policy(terminal_q_values)

        # Store policy
        self.policy[i_terminal_draw, :, :, :] = terminal_policy

        # V(s) = Σ_a π(a|s) Q(s,a)
        terminal_value = np.sum(terminal_policy * terminal_q_values, axis=-1)

        self.value_function[i_terminal_draw, :, :] = terminal_value
        for i_draw in range(self.l_draws - 2, -1, -1):
            draw = self.draw_axis[i_draw]  # Current draw value (e.g., 7, 6, ...)

            # Create an interpolator for V(s')
            # We create a 2D interpolator for this slice.
            v_next_slice = self.value_function[i_draw + 1, :, :]
            axes_2d = (self.yellow_axis, self.blue_axis)

            v_interpolator = RegularGridInterpolator(
                axes_2d,
                v_next_slice,
                bounds_error=False,
                fill_value=np.nan,  # If we go off-grid,nan
            )
            # v_interpolator.values = self.value_function[i_draw + 1]

            #  Define the valid state space for the current draw
            max_cards = draw * self.max_cards_per_draw
            valid_y_indices = np.where(self.yellow_axis <= max_cards)[0]
            valid_b_indices = np.where(self.blue_axis <= max_cards)[0]

            if len(valid_y_indices) == 0 or len(valid_b_indices) == 0:
                continue

            # Create index meshgrid
            idx_y_grid, idx_b_grid = np.meshgrid(
                valid_y_indices, valid_b_indices, indexing="ij"
            )

            # Get the values of the current valid states
            current_y_grid = self.yellow_axis[idx_y_grid]
            current_b_grid = self.blue_axis[idx_b_grid]

            # Filter for valid states (y + b <= max_cards)
            valid_state_mask = (current_y_grid + current_b_grid) <= (max_cards + 1e-5)

            if not np.any(valid_state_mask):
                continue

            # Filter all relevant grids
            valid_idx_y = idx_y_grid[valid_state_mask]
            valid_idx_b = idx_b_grid[valid_state_mask]
            current_y_valid = current_y_grid[valid_state_mask]
            current_b_valid = current_b_grid[valid_state_mask]

            next_yellows_outcomes = np.array(self.next_yellow)
            next_blues_outcomes = self.max_cards_per_draw - next_yellows_outcomes
            n_outcomes = len(next_yellows_outcomes)

            # Broadcast to find all possible future states (values)
            # future_y_grid shape: (n_valid_states, n_outcomes)
            future_y_grid = (
                self.gamma * current_y_valid[..., np.newaxis] + (1 - self.gamma)
            ) + next_yellows_outcomes[np.newaxis, :]
            future_b_grid = (
                self.gamma * current_b_valid[..., np.newaxis] + (1 - self.gamma) *self.beta
            ) + next_blues_outcomes[np.newaxis, :]

            #  Get V(s') by interpolating
            # We need to query the interpolator at all (y', b') points.
            # v_interpolator expects points as (n_points, 2)

            # Flatten and stack future states for interpolation
            query_points = np.stack(
                [future_y_grid.ravel(), future_b_grid.ravel()], axis=-1
            )

            # Get interpolated values, V(s')
            interpolated_V_next = v_interpolator(query_points)

            # Reshape back to (n_valid_states, n_outcomes)
            expected_future_value = interpolated_V_next.reshape(
                current_y_valid.shape[0], n_outcomes
            )

            #  Calculate transition probabilities
            transitions = self.calculate_transition_probability(
                i=next_yellows_outcomes[np.newaxis, :],
                num_yellows=self.gamma * current_y_valid[..., np.newaxis]
                + (1 - self.gamma),
                num_blues=self.gamma * current_b_valid[..., np.newaxis]
                + (1- self.gamma)*self.beta,
            )  # Shape: (n_valid_states, n_outcomes)
            # Store transitions
            self.transition_probability[i_draw, valid_idx_y, valid_idx_b, :] = (
                transitions
            )

            # Calculate the "wait" action value Q(s, W)
            # Q(s, W) = Cost(s) + sum_s' P(s'|s, W) * V(s')
            # Sum over the "outcomes" axis (axis=-1)
            wait_action_value_undiscounted = np.sum(
                transitions * expected_future_value, axis=-1
            )  # Shape: (n_valid_states,)

            urgency_cost = self.sigmoidal_cost(draw)

            if self.is_hazardous:
                discount_factor = 1 - self.cum_hazard[draw]
                hazard_cost = self.cum_hazard[draw] * self.deadline_cost
                total_wait_value = (
                    discount_factor * wait_action_value_undiscounted
                    + hazard_cost
                    + urgency_cost
                )
            else:
                total_wait_value = wait_action_value_undiscounted + urgency_cost

            # --- 2g. Update Q(s, W), V(s), and Policy(s) for current draw ---

            # Assign Q(s, W)
            self.action_values[i_draw, valid_idx_y, valid_idx_b, 2] = total_wait_value

            # Get all Q-values for the current valid states
            current_q_values = self.action_values[i_draw, valid_idx_y, valid_idx_b]

            current_policy = self.softmax_policy(current_q_values)
            current_value_function_values = np.sum(
                current_policy * current_q_values, axis=-1
            )
            self.value_function[i_draw, valid_idx_y, valid_idx_b] = (
                current_value_function_values
            )

            # Update Policy(s)
            self.policy[i_draw, valid_idx_y, valid_idx_b, :] = current_policy

        self.extract_actions()

    @staticmethod
    def approximate_to_closest(value: float, gamma_values: list) -> float:
        """
        Snap a candidate gamma value to the nearest entry in gamma_values, since
        this class's belief/value grids are only pre-built for a fixed set of gammas.

        Args:
            value (float): Candidate gamma value (e.g. from an optimizer).
            gamma_values (list): Allowed discrete gamma values.

        Returns:
            float: The closest entry in gamma_values, rounded to 2 decimals.
        """
        return np.round(min(gamma_values, key=lambda x: abs(x - value)), 2)

    @staticmethod
    def fit_differential_evolution(
        param_ranges: dict, cost_function: "Callable[[list], float]", x0=None
    ) -> tuple:
        """
        Fit parameters by minimizing cost_function with scipy's differential evolution,
        then snap the fitted gamma to the closest value in gamma_values.

        Args:
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple;
                must include "gamma" for the snapping step below.
            cost_function (Callable[[list], float]): Function to minimize, e.g. one
                built by make_cost_function.

        Returns:
            tuple: (best_params_dict, best_fitness, hessian_placeholder), where
                hessian_placeholder is a 5x5 zero matrix. start_hazard, if present,
                is floored to an int.
        """
        bounds = [param_ranges[k] for k in param_ranges.keys()]

        result = run_differential_evolution(param_ranges, cost_function, x0=x0)
        best_params = result.x
        best_fitness = result.fun
        gamma_index = PARAM_ORDER.index("gamma")
        gamma_value_corrected = POMDP_Forgetting.approximate_to_closest(
            best_params[gamma_index], gamma_values=gamma_values
        )
        best_params[gamma_index] = gamma_value_corrected

        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        if "start_hazard" in best_params_dict.keys():
            best_params_dict["start_hazard"] = int(
                np.floor(best_params_dict["start_hazard"])
            )

        return best_params_dict, best_fitness, np.zeros((5, 5))

    def log_likelihood(self, data: "pd.DataFrame") -> float:
        """
        Sum the log-probability the fitted (interpolated) policy assigns to each
        subject's observed sequence of Wait actions followed by a final choice.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column
                of per-trial sequences, each an iterable of (draw, num_yellow, num_blue,
                action, outcome) tuples with action in {0: Yellow, 1: Blue, 2: Wait}.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        for action_seq in actions_col:
            for draw, y, b, act, _ in action_seq:

                p_y, p_b, p_w = self.get_interpolated_policy(draw, y, b)

                if act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    # Use safe log
                    total_ll += np.log(p_action + 1e-10)
                    break
                elif act == 2:
                    total_ll += np.log(p_w + 1e-10)

        return total_ll

    def log_likelihood_commit(self, data: "pd.DataFrame") -> float:
        """
        Same as log_likelihood, but collapses Yellow/Blue into a single "commit/go"
        probability (p_y + p_b) so the fit only distinguishes Wait from committing,
        not which color was chosen.

        The base-class version indexes self.policy directly, which this class
        cannot use: a forgetting model's policy is read through
        get_interpolated_policy, which interpolates between the two neighbouring
        gamma grid points. Inheriting the base version raises an indexing error on
        every evaluation, and the cost function swallows it and returns 1e10, so
        the fit silently optimizes a constant. This mirrors log_likelihood above,
        exactly as POMDP_exaggerate.log_likelihood_commit mirrors its own.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column,
                as in log_likelihood.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        for action_seq in actions_col:
            for draw, y, b, act, _ in action_seq:

                p_y, p_b, p_w = self.get_interpolated_policy(draw, y, b)

                if act == 0 or act == 1:  # choose yellow or blue
                    total_ll += np.log(p_y + p_b + 1e-10)
                    break
                elif act == 2:
                    total_ll += np.log(p_w + 1e-10)

        return total_ll

    def log_likelihood_extended(self, data: "pd.DataFrame") -> list:
        """
        Same walk as log_likelihood, but instead of a single summed value, returns
        one record per trial with the full policy breakdown, for diagnostics/plotting.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column,
                as in log_likelihood.

        Returns:
            list: One [draw, y, b, p_y, p_b, p_w, p_chosen, action, ll, cum_ll] row per
                trial, where cum_ll is the running log-likelihood within that trial.
        """
        total_ll = []
        ll = 0
        cum_ll = 0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        eps = 1e-10
        for action_seq in actions_col:

            for draw, y, b, act, outcome in action_seq:
                p_y, p_b, p_w = self.get_interpolated_policy(draw, y, b)
                if act == 2:  # wait
                    ll = np.log(p_w + eps)
                    cum_ll = cum_ll + ll

                    total_ll.append([draw, y, b, p_y, p_b, p_w, p_w, act, ll, cum_ll])
                elif act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    ll = np.log(p_action + eps)
                    cum_ll = cum_ll + ll

                    total_ll.append(
                        [draw, y, b, p_y, p_b, p_w, p_action, act, ll, cum_ll]
                    )
                    break  # I need this break, because I have the full sequence and subjects can decide but still see the pre-assigned sequence yet.

        return total_ll

    def make_cost_function(
        self, list_of_evidence_to_fit_dict: dict
    ) -> "Callable[[list], float]":
        """
        Build a cost function suitable for the optimizers below (ga_fit /
        fit_differential_evolution). Unlike POMDP.make_cost_function, the candidate
        gamma is first snapped to the nearest pre-built grid value (approximate_to_closest)
        and used to select which evidence dict to fit against, since this class's
        belief/value grids only exist for a fixed set of gammas.

        Args:
            list_of_evidence_to_fit_dict (dict): Maps gamma value -> {horizon_condition:
                DataFrame of trials}, i.e. one evidence dict per pre-built gamma grid.

        Returns:
            Callable[[list], float]: cost_function(params), where params is a flat
                list ordered per PARAM_ORDER (including "gamma"); returns 1e10 on
                any exception.
        """

        def cost_function(params):
            try:
                gamma_index = PARAM_ORDER.index("gamma")

                # now I have three datasets here in the evidence_to_fit_dict.
                # first I need to choose which one to proceed with.
                gamma_value = POMDP_Forgetting.approximate_to_closest(
                    params[gamma_index], gamma_values=gamma_values
                )
            except:
                gamma_value = 1.0
            evidence_to_fit_dict = list_of_evidence_to_fit_dict[gamma_value]

            # separate the dataset into two, and return each according to their separation. Assume that the data is separated with label long and short
            # I can have a disctionary of the two data data_dict['short']= evidence_to_fit and the other one and then
            try:
                # extract the keys from the evidence_to_fit_dict
                keys = evidence_to_fit_dict.keys()
                # loop over the keys
                # the key is either 'long' or 'short'
                ll = 0
                params = {k: v for k, v in zip(PARAM_ORDER, params)}
                # check if start_hazard is one of the keys of the Param_order, if so, make corresponding value to be intger.
                if "start_hazard" in params:
                    params["start_hazard"] = int(np.floor(params["start_hazard"]))

                params.update(
                    {
                        "verbose": VERBOSE,
                        "max_cards_per_draw": MAX_CARDS_PER_DRAW,
                        "gamma": gamma_value,
                    }
                )
                # loop over the keys and calculate the log likelihood for each key
                _ll_fn = self.log_likelihood_commit if POMDP_COMMIT else self.log_likelihood
                for key in keys:
                    params.update({"horizon_condition": key})

                    data_single_horizon = evidence_to_fit_dict[key]
                    self.__init__(**params)
                    self.value_iteration()
                    ll += _ll_fn(
                        data_single_horizon
                    )  # instead of that one return, I can return the sum of two
                cost = -ll

                return cost
            except Exception as e:
                print(f"[Cost Error] {e}")
                return 1e10

        return cost_function

    @staticmethod
    def ga_fit(param_ranges: dict, cost_function: "Callable[[list], float]") -> tuple:
        """
        Fit parameters by minimizing cost_function with a genetic algorithm
        (geneticalgorithm.geneticalgorithm), then snap the fitted gamma to the
        closest value in gamma_values if "gamma" is in param_ranges.

        Args:
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
                Order determines the parameter vector order passed to cost_function.
            cost_function (Callable[[list], float]): Function to minimize, e.g. one
                built by make_cost_function.

        Returns:
            tuple: (best_params_dict, best_fitness, hessian_placeholder), where
                hessian_placeholder is a zero matrix (GA does not estimate a Hessian).
                start_hazard, if present, is floored to an int.
        """
        varbound = np.array([param_ranges[k] for k in param_ranges.keys()])
        algorithm_param = get_ga_params()
        model = ga(
            function=cost_function,
            dimension=len(param_ranges.keys()),
            convergence_curve=False,
            progress_bar=False,
            variable_type_mixed=np.array(variable_type),
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param,
        )

        model.run()

        best_params = model.output_dict["variable"]
        try:
            gamma_index = PARAM_ORDER.index("gamma")
            # now I have three datasets here in the evidence_to_fit_dict.
            # first I need to choose which one to proceed with.
            gamma_value_corrected = POMDP_Forgetting.approximate_to_closest(
                best_params[gamma_index], gamma_values=gamma_values
            )
            best_params[gamma_index] = gamma_value_corrected
        except:
            print("Gamma is inactive")

        best_fitness = model.output_dict["function"]
        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        if "start_hazard" in best_params_dict.keys():
            best_params_dict["start_hazard"] = int(
                np.floor(best_params_dict["start_hazard"])
            )
        return (
            best_params_dict,
            best_fitness,
            np.zeros((len(param_ranges.keys()), len(param_ranges.keys()))),
        )

    def fit_subject(
        self,
        df_ev_simulated: dict,
        param_ranges: dict,
        subject_ID,
        algorithm: str,
    ) -> tuple:
        """
        Fit this POMDP's free parameters (including gamma) to one subject's data
        using the requested optimizer.

        Args:
            df_ev_simulated (dict): Maps gamma value -> {horizon_condition: DataFrame
                of trials}, passed through to make_cost_function.
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
            subject_ID: Identifier for the subject/dataset being fit, echoed back in
                the return value.
            algorithm (str): "ga" for genetic algorithm or "de" for differential evolution.

        Returns:
            tuple: (best_params, log_likelihood, df_ev_simulated[gamma], subject_ID,
                hessian_matrix), where gamma is the fitted value (or 1.0 if gamma
                was not a free parameter).
        """
        cost_function = self.make_cost_function(df_ev_simulated)
        if algorithm == "ga":
            best_params, best_cost, hessian_matrix = self.ga_fit(
                param_ranges, cost_function
            )
        if algorithm == "de":
            best_params, best_cost, hessian_matrix = self.fit_differential_evolution(
                param_ranges, cost_function, x0=de_seed_for(subject_ID, param_ranges)
            )
        log_likelihood = -best_cost
        try:
            gamma = best_params["gamma"]
        except:
            gamma = 1.0
        return (
            best_params,
            log_likelihood,
            df_ev_simulated[gamma],
            subject_ID,
            hessian_matrix,
        )

    # Not correctly implemented yet. Just if the data entered here is the same as before but the interpolation doesn't work yet
    # raise error not implemented if someone called the function
    def simulate_cards_pomdp(
        self,
        given_sequence: bool = False,
        card_sequence: list | None = None,
        start_hazard: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """
        Simulate one trial of the card task: generate (or take a given) sequence of
        yellow/blue draws, then step through it sampling actions from self.policy
        (or self.belief for the interpolated subclasses) until a Yellow/Blue choice
        is made, the deadline is reached, or the hazard-truncated sequence runs out.

        Args:
            given_sequence (bool): If True, replay card_sequence instead of generating
                a random one.
            card_sequence (list, optional): Required when given_sequence is True; a
                sequence of (draw, cum_yellow, cum_blue, action, outcome) tuples as
                recorded from the real experiment.
            start_hazard (int, optional): Draw at which the hazard function starts,
                for the randomly-generated case. Defaults to self.start_hazard.
            seed (int, optional): Seed for np.random, for reproducible simulation.

        Returns:
            dict: Belief trajectory, actions taken, reward, decision, num_draws,
                num_draws_list, max_draws (the realized deadline), and the observed
                yellow/blue counts (both up to the decision and for the full sequence).
        """
        # initalize the lists, and variables
        num_yellows = []
        num_blues = []
        yellow_trace = []
        blue_trace = []
        belief_trajectory = []
        actions = []
        num_draws = 1
        num_draws_list = []
        num_yellow_observed = []

        # Initializing the cards sequence for the case that no sequence is given. This is intialized randomly
        correct_decision = None

        np.random.seed(seed)
        if not given_sequence:
            # Draw a number to determine the number of draws
            u = np.random.random()

            # Here I need to recalculate the hazard even though it could've been calculated in the class.
            # the reason is that, I need the simulate_cards always to be hazardous and random at each draw.
            # but the one in the class is fixed. So that's why I should separate the generation of cards here from
            # what was previously calculated in the class.
            # also no matter whether is_hazardous is true or false, I need to have a hazardous generation of cards.
            # to match the real experiment.

            # first, I could have the start hazard as a free parameter in the future to be fitted.
            if start_hazard is None:
                start_hazard = self.start_hazard

            # but the end_hazard is always the same, i.e., the max draws depending on the horizon condition, long or short
            hazard = calculate_hazard_cum(int(start_hazard), self.max_draws)

            # now this is to calculate the actual deadline that could be smaller than the max_draws
            deadline = np.searchsorted(hazard, u, side="right")
            # I'm generating the trace now like tobias's one

            p_yellow = np.random.uniform(0.4, 0.6)
            while p_yellow == 0.5:
                p_yellow = np.random.uniform(0.4, 0.6)

            yellow_trace = np.random.binomial(
                n=self.max_cards_per_draw, p=p_yellow, size=deadline
            )
            blue_trace = self.max_cards_per_draw - yellow_trace

            # I just insert zeros at the beginning, so that when I calculate the cumsum, I have a full list of cards.
            yellow_trace = np.insert(yellow_trace, 0, 0)
            blue_trace = np.insert(blue_trace, 0, 0)

            # the belief and policy is always w.r.t. total evidence. So I need the num_yellows, num_blues, total
            num_yellows = np.cumsum(yellow_trace)
            num_blues = np.cumsum(blue_trace)
            correct_decision = 0 if p_yellow > 0.5 else 1

        elif given_sequence:
            # this randomize seed, because if I called the first if within the same code run,
            # the seed will be fixed always, so I just need to re randomize it again here.
            cum_y = np.array([trial[1] for trial in card_sequence], dtype=int)
            cum_b = np.array([trial[2] for trial in card_sequence], dtype=int)

            # add initial 0 to match the other branch (so traces start at draw 0)
            num_yellows = np.insert(cum_y, 0, 0)
            num_blues = np.insert(cum_b, 0, 0)

            # number of draws
            deadline = len(num_blues) - 1
            # extract the reward from the last trial in the sequence, which is the outcome of the decision
            real_reward = card_sequence[-1][-1]
            real_action = card_sequence[-1][-2]
            if real_reward == 2:
                correct_decision = real_action  # the action taken in the last trial
            elif real_reward == -2:
                correct_decision = 1 if real_action == 0 else 0
            else:
                # in case of missing here, the correct action is calculated based on the total number of yellow and total number of blue because I don't know the generative probability.
                correct_decision = 0 if num_yellows[-1] > num_blues[-1] else 1

        # Do the first action outside the loop, if action==2 go inside the loop
        num_yellow_observed.append(num_yellows[num_draws])
        num_draws_list.append(num_draws)
        belief = self.get_interpolated_belief(
            num_draws, num_yellows[num_draws], num_blues[num_draws]
        )[0]
        belief_trajectory.append(belief)  # belief of yellow
        policy = self.get_interpolated_policy(
            num_draws, num_yellows[num_draws], num_blues[num_draws]
        )
        action = self.get_stochastic_action_depending_on_policy(policy)
        actions.append(int(action))

        while action == 2 and num_draws < (len(num_yellows) - 1):
            num_draws += 1
            num_yellow_observed.append(num_yellows[num_draws])
            num_draws_list.append(num_draws)
            belief = self.get_interpolated_belief(
                num_draws, num_yellows[num_draws], num_blues[num_draws]
            )[0]
            belief_trajectory.append(belief)  # belief of yellow
            policy = self.get_interpolated_policy(
                num_draws, num_yellows[num_draws], num_blues[num_draws]
            )
            action = self.get_stochastic_action_depending_on_policy(policy)
            actions.append(int(action))

        belief_trajectory = np.array(belief_trajectory)
        if action == int(2):
            reward = -1  # missed the deadline
        elif action == correct_decision:
            reward = 2
        elif action != correct_decision:
            reward = -2

        results = {
            "trajectory": belief_trajectory,
            "reward": reward,
            "actions": actions,
            "decision": actions[-1],
            "num_draws": num_draws,
            "num_draws_list": num_draws_list,
            "max_draws": deadline,
        }

        results["num_yellows"] = num_yellows[1 : num_draws + 1]
        results["num_blues"] = num_blues[1 : num_draws + 1]
        results["num_yellows_full_sequence"] = num_yellows[1:]
        results["num_blues_full_sequence"] = num_blues[1:]

        return results

    def plot_policy(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot self.policy on the symmetric (yellow-blue difference) grid for this
        non-integer grid POMDP. Outputs one heatmap per action.

        Args:
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            None. Figures are shown (and optionally saved); the mapping of
            action_index -> masked array is currently not returned (see commented-out
            return at the end of this method).
        """

        # --- 1. Get data from the POMDP object ---
        policy = self.policy
        draw_axis = self.draw_axis
        yellow_axis = self.yellow_axis
        blue_axis = self.blue_axis
        step_size = self.step_size
        max_cards_per_draw = self.max_cards_per_draw
        max_yellows_val = self.max_yellows_val

        if policy.ndim != 4:
            raise ValueError(
                "policy must have shape (l_draws, n_yellows, m_blues, n_actions)"
            )

        l_draws, n_yellows, m_blues, n_actions = policy.shape

        action_labels = ["Action Yellow", "Action Blue", "Action Wait"]

        # --- 2. Define the new 'difference' axis ---
        min_diff_val = -max_yellows_val
        max_diff_val = max_yellows_val

        # Create the axis for (yellow - blue)
        diff_axis = np.arange(min_diff_val, max_diff_val + step_size, step_size)
        n_diffs = len(diff_axis)

        # Create the grid to hold the plot data
        adjusted_probs = np.full((n_actions, l_draws, n_diffs), np.nan, dtype=float)

        # --- 3. Iterate and populate the adjusted grid ---
        for i_draw, draw_val in enumerate(draw_axis):
            max_cards_for_this_draw = draw_val * max_cards_per_draw

            for i_yellow, yellow_val in enumerate(yellow_axis):

                # Check if this yellow value is possible for this draw
                if yellow_val > max_cards_for_this_draw + 1e-5:
                    continue

                # Calculate the corresponding 'blue' value for the "full hand slice"
                blue_val = max_cards_for_this_draw - yellow_val

                # Find the index for this blue value on the blue_axis
                if blue_val < -1e-5:  # Cannot be negative
                    continue

                i_blue = np.searchsorted(blue_axis, blue_val)

                # Check if index is valid and the value is actually on the grid
                if i_blue >= m_blues or not np.isclose(blue_axis[i_blue], blue_val):
                    continue  # This blue_val is not on our grid

                # --- We found a valid (i_draw, i_yellow, i_blue) on the slice ---

                # Calculate the difference value
                diff_val = yellow_val - blue_val

                # Find the corresponding index on the diff_axis
                i_diff = np.searchsorted(diff_axis, diff_val)

                if i_diff >= n_diffs or not np.isclose(diff_axis[i_diff], diff_val):
                    continue

                # Get the policy vector and store it
                for a in range(n_actions):
                    adjusted_probs[a, i_draw, i_diff] = policy[
                        i_draw, i_yellow, i_blue, a
                    ]

        # --- 4. Mask and Plot ---

        # mask invalid entries (NaNs)
        masks = np.isnan(adjusted_probs)
        masked = np.ma.array(adjusted_probs, mask=masks)
        max_val = np.nanmax(adjusted_probs)
        min_val = np.nanmin(adjusted_probs)
        _set_plot_style()

        # plotting: one figure per action
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))
            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=min_val,
                vmax=max_val,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Probability")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Yellow - Blue Difference (on 'full hand' slice)")
            ax.set_xlabel("Number of Draws")

            # --- Set X-ticks ---
            ax.set_xticks(np.arange(0, l_draws, 1))
            ax.set_xticklabels(draw_axis.astype(int))

            # --- Set Y-ticks intelligently ---
            tick_labels = []
            tick_indices = []

            max_abs_diff_int = int(np.floor(max_diff_val))
            tick_step = max(1, int(np.ceil(max_abs_diff_int / 5)))
            if tick_step > 1 and tick_step % 2 != 0:
                tick_step += 1  # Prefer even steps like 2, 4, 6

            desired_ticks = np.arange(
                -max_abs_diff_int, max_abs_diff_int + 1, tick_step
            )

            for tick_val in desired_ticks:
                i_tick = np.searchsorted(diff_axis, tick_val)
                if i_tick < len(diff_axis) and np.isclose(diff_axis[i_tick], tick_val):
                    tick_indices.append(i_tick)
                    tick_labels.append(int(tick_val))

            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels)

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
            plt.tight_layout()

            # --- 5. Save Figure ---
            if save_fig:
                os.makedirs(path, exist_ok=True)

                # Use a short label for the action in the filename
                action_fname = action_labels[a].replace(" ", "_").lower()

                base = f"{path}/policy_forgetting_{action_fname}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        # return dict of masked arrays per action for further inspection
        # return {action_labels[i]: masked[i] for i in range(n_actions)}

    def plot_action_values(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot self.action_values (Q-values) on the symmetric (yellow-blue difference)
        grid for this non-integer grid POMDP. Outputs one heatmap per action.

        Args:
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            None. Figures are shown (and optionally saved); the mapping of
            action_index -> masked array is currently not returned (see commented-out
            return at the end of this method).
        """

        # --- 1. Get data from the POMDP object ---
        action_values = self.action_values
        draw_axis = self.draw_axis
        yellow_axis = self.yellow_axis
        blue_axis = self.blue_axis
        step_size = self.step_size
        max_cards_per_draw = self.max_cards_per_draw
        max_yellows_val = self.max_yellows_val

        if action_values.ndim != 4:
            raise ValueError(
                "policy must have shape (l_draws, n_yellows, m_blues, n_actions)"
            )

        l_draws, n_yellows, m_blues, n_actions = action_values.shape

        action_labels = ["Action Yellow", "Action Blue", "Action Wait"]
        # --- 2. Define the new 'difference' axis ---
        min_diff_val = -max_yellows_val
        max_diff_val = max_yellows_val
        _set_plot_style()

        # Create the axis for (yellow - blue)
        diff_axis = np.arange(min_diff_val, max_diff_val + step_size, step_size)
        n_diffs = len(diff_axis)

        # Create the grid to hold the plot data
        adjusted_probs = np.full((n_actions, l_draws, n_diffs), np.nan, dtype=float)

        # --- 3. Iterate and populate the adjusted grid ---
        for i_draw, draw_val in enumerate(draw_axis):
            max_cards_for_this_draw = draw_val * max_cards_per_draw

            for i_yellow, yellow_val in enumerate(yellow_axis):

                # Check if this yellow value is possible for this draw
                if yellow_val > max_cards_for_this_draw + 1e-5:
                    continue

                # Calculate the corresponding 'blue' value for the "full hand slice"
                blue_val = max_cards_for_this_draw - yellow_val

                # Find the index for this blue value on the blue_axis
                if blue_val < -1e-5:  # Cannot be negative
                    continue

                i_blue = np.searchsorted(blue_axis, blue_val)

                # Check if index is valid and the value is actually on the grid
                if i_blue >= m_blues or not np.isclose(blue_axis[i_blue], blue_val):
                    continue  # This blue_val is not on our grid

                # --- We found a valid (i_draw, i_yellow, i_blue) on the slice ---

                # Calculate the difference value
                diff_val = yellow_val - blue_val

                # Find the corresponding index on the diff_axis
                i_diff = np.searchsorted(diff_axis, diff_val)

                if i_diff >= n_diffs or not np.isclose(diff_axis[i_diff], diff_val):
                    continue

                # Get the policy vector and store it
                for a in range(n_actions):
                    adjusted_probs[a, i_draw, i_diff] = action_values[
                        i_draw, i_yellow, i_blue, a
                    ]

        # --- 4. Mask and Plot ---

        # mask invalid entries (NaNs)
        masks = np.isnan(adjusted_probs)
        masked = np.ma.array(adjusted_probs, mask=masks)
        max_val = np.nanmax(adjusted_probs)
        min_val = np.nanmin(adjusted_probs)

        # plotting: one figure per action
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))
            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=min_val,
                vmax=max_val,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Action Values")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Yellow - Blue Difference (on 'full hand' slice)")
            ax.set_xlabel("Number of Draws")

            # --- Set X-ticks ---
            ax.set_xticks(np.arange(0, l_draws, 1))
            ax.set_xticklabels(draw_axis.astype(int))

            # --- Set Y-ticks intelligently ---
            tick_labels = []
            tick_indices = []

            max_abs_diff_int = int(np.floor(max_diff_val))
            tick_step = max(1, int(np.ceil(max_abs_diff_int / 5)))
            if tick_step > 1 and tick_step % 2 != 0:
                tick_step += 1  # Prefer even steps like 2, 4, 6

            desired_ticks = np.arange(
                -max_abs_diff_int, max_abs_diff_int + 1, tick_step
            )

            for tick_val in desired_ticks:
                i_tick = np.searchsorted(diff_axis, tick_val)
                if i_tick < len(diff_axis) and np.isclose(diff_axis[i_tick], tick_val):
                    tick_indices.append(i_tick)
                    tick_labels.append(int(tick_val))

            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels)

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
            plt.tight_layout()

            # --- 5. Save Figure ---
            if save_fig:
                os.makedirs(path, exist_ok=True)

                # Use a short label for the action in the filename
                action_fname = action_labels[a].replace(" ", "_").lower()

                base = f"{path}/action_prob_heatmap_forgetting_{action_fname}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        # return dict of masked arrays per action for further inspection
        # return {action_labels[i]: masked[i] for i in range(n_actions)}


class POMDP_Exaggeration(POMDP_Urgency):
    """
    Simulates the decision making using the generative and majority belief models.
    Uses a non-integer grid for state representation and interpolation, scaled by
    gamma so the perceived total evidence (grid extent) is exaggerated (gamma > 1)
    or shrunk (gamma < 1) relative to the actual number of cards drawn.
    """

    def __init__(
        self,
        tau: float = TAU,
        xi: float = XI,
        horizon_condition: str = HORIZON_CONDITION,
        is_hazardous: bool = IS_HAZARDOUS,
        subjective_cost: float = SUBJECTIVE_COST,
        verbose: bool = VERBOSE,
        max_cards_per_draw: int = MAX_CARDS_PER_DRAW,
        patience: float = PATIENCE,
        urgency_coefficient: float = URGENCY_COEFFICIENT,
        urgency_slope: float = URGENCY_SLOPE,
        c_max: float = C_MAX,
        gamma: float = GAMMA,
        start_hazard: int | None = None,
    ):
        """
        Initialize a POMDP_Urgency variant that replaces the integer belief/value
        grids with a continuous (step_size-spaced) yellow/blue grid, whose maximum
        extent (max_yellows_val/max_blues_val) is scaled by gamma.

        Args:
            tau (float): Softmax temperature for the action policy.
            xi (float): Lapse rate mixed into the Yellow/Blue softmax probabilities.
            horizon_condition (str): "short" or "long"; sets max_draws and the default start_hazard.
            is_hazardous (bool): Whether the hazard (deadline) function is applied during planning.
            subjective_cost (float): Added to the incorrect-choice reward.
            verbose (bool): Whether to print progress information.
            max_cards_per_draw (int): Number of cards revealed per draw.
            patience (float): Midpoint of the urgency sigmoid, in draws.
            urgency_coefficient (float): Lower bound of the urgency sigmoid cost.
            urgency_slope (float): Steepness of the urgency sigmoid.
            c_max (float): Upper bound of the urgency sigmoid cost.
            gamma (float): Multiplier on the grid's max yellow/blue extent, exaggerating
                (>1) or shrinking (<1) the perceived total evidence.
            start_hazard (int, optional): Draw at which the hazard function starts.
                Defaults to 4 (short) or 10 (long) when None.
        """
        super().__init__(
            tau=tau,
            xi=xi,
            horizon_condition=horizon_condition,
            is_hazardous=is_hazardous,
            subjective_cost=subjective_cost,
            verbose=verbose,
            max_cards_per_draw=max_cards_per_draw,
            patience=patience,
            urgency_coefficient=urgency_coefficient,
            urgency_slope=urgency_slope,
            c_max=c_max,
            start_hazard=start_hazard,
        )

        self.gamma = gamma

        self.max_yellows_val = self.max_cards_per_draw * self.max_draws * self.gamma
        self.max_blues_val = self.max_cards_per_draw * self.max_draws * self.gamma
        self.step_size = 0.2

        # --- Define grid axes values ---
        self.draw_axis = np.arange(0, self.max_draws + 1)

        self.yellow_axis = np.arange(
            0, self.max_yellows_val + self.step_size, self.step_size
        )
        self.blue_axis = np.arange(
            0, self.max_blues_val + self.step_size, self.step_size
        )

        # Get grid dimensions
        self.l_draws = len(self.draw_axis)
        self.n_yellows = len(self.yellow_axis)
        self.m_blues = len(self.blue_axis)

        # --- Initialize grid arrays ---
        self.action_values = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 3))
        self.belief = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 3))
        self.value_function = np.zeros((self.l_draws, self.n_yellows, self.m_blues))
        self.belief_grid = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 2))
        self.transition_probability = np.zeros(
            (self.l_draws, self.n_yellows, self.m_blues, len(self.next_yellow))
        )
        self.policy = np.zeros((self.l_draws, self.n_yellows, self.m_blues, 3))
        self.best_actions = np.ones((self.l_draws, self.n_yellows, self.m_blues)) * 5

        # Fill the belief grid
        self.generate_belief_grid()

    def generate_belief_grid(self) -> None:
        """
        Populate the belief grid for all valid states.

        Returns:
            None. Sets self.belief_grid in place.
        """
        # Iterate over all draws, including draw 0
        for i_draw, draw in enumerate(self.draw_axis):
            if draw == 0:
                belief_val_0 = 1 - beta.cdf(0.5, self.alpha, self.beta)  # Should be 0.5
                self.belief_grid[0, 0, 0, 0] = belief_val_0
                self.belief_grid[0, 0, 0, 1] = 1 - belief_val_0
                continue

            max_cards = draw * self.max_cards_per_draw * self.gamma

            # Find indices of valid yellow and blue axes
            # valid numbers by checking y <= max_cards and b <= max_cards,
            valid_y_indices = np.where(self.yellow_axis <= max_cards)[0]
            valid_b_indices = np.where(self.blue_axis <= max_cards)[0]

            # Get the actual values
            valid_yellows = self.yellow_axis[valid_y_indices]
            valid_blues = self.blue_axis[valid_b_indices]

            if len(valid_yellows) == 0 or len(valid_blues) == 0:
                continue

            # Create a meshgrid
            grid_y, grid_b = np.meshgrid(valid_yellows, valid_blues, indexing="ij")

            # Calculate beliefs
            Alpha = self.alpha + grid_y
            Beta_ = self.beta + grid_b
            belief_vals = 1 - beta.cdf(0.5, Alpha, Beta_)

            # Get a meshgrid of indices to slice the belief_grid
            idx_y, idx_b = np.meshgrid(valid_y_indices, valid_b_indices, indexing="ij")

            # Assign beliefs to the correct grid locations
            # Only populate beliefs where total cards is valid
            # This is a bit inefficient but correct.
            total_cards_grid = grid_y + grid_b
            valid_state_mask = total_cards_grid <= (max_cards + 1e-5)  # Add tolerance

            # Apply mask to indices
            valid_idx_y = idx_y[valid_state_mask]
            valid_idx_b = idx_b[valid_state_mask]

            # Apply mask to calculated beliefs
            valid_beliefs = belief_vals[valid_state_mask]

            self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 0] = valid_beliefs
            self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 1] = 1 - valid_beliefs

    def get_interpolated_belief(
        self, draw: float, yellow: float, blue: float
    ) -> np.ndarray:
        """
        Interpolate belief_grid at (draw, yellow, blue) values.

        Args:
            draw (float): Draw index (may be off-grid; interpolated on draw_axis).
            yellow (float): Total yellow count (interpolated on yellow_axis).
            blue (float): Total blue count (interpolated on blue_axis).

        Returns:
            np.ndarray: Length-2 array [P(Yellow), P(Blue)]; NaN if out of grid bounds.
        """

        if not hasattr(self, "belief_grid") or self.belief_grid is None:
            raise RuntimeError("self.belief_grid is not available")

        # Define the axes for the interpolator
        axes = (self.draw_axis, self.yellow_axis, self.blue_axis)

        # Create interpolator for P(Yellow)
        interpolator_y = _RGI(
            axes, self.belief_grid[..., 0], bounds_error=False, fill_value=np.nan
        )

        # Query point must be 2D: (n_points, ndim)
        pt = np.array([[float(draw), float(yellow), float(blue)]])
        result_y = interpolator_y(pt)

        # P(Blue) = 1 - P(Yellow)
        result_b = 1.0 - result_y

        # Squeeze to remove the (1,) dimension and stack
        return np.array([np.squeeze(result_y), np.squeeze(result_b)])

    def softmax_policy(self, action_values: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax with lapse applied ONLY to first two actions.
        Same as POMDP.softmax_policy but without the (unused) draw parameter.

        Args:
            action_values (np.ndarray): Q-values (..., n_actions), with Wait as the last action.
            axis (int): Axis over which to apply the softmax (must index the actions dimension).

        Returns:
            np.ndarray: Action-probability array with the same shape as action_values.
        """
        # Standard, stable softmax
        max_val = np.max(action_values, axis=axis, keepdims=True)
        all_impossible_mask = np.isneginf(max_val)
        safe_max_val = np.where(all_impossible_mask, 0, max_val)
        # tau is a softmax denominator -- as it approaches 0 the division
        # blows up (divide-by-zero/overflow), so floor it for this division
        # only; self.tau itself (the fitted/reported parameter) is untouched.
        safe_tau = max(self.tau, 1e-3)
        scaled_values = (action_values - safe_max_val) / safe_tau

        exps = np.exp(scaled_values)
        denominator = np.sum(exps, axis=axis, keepdims=True)
        softmax_probs = np.nan_to_num(exps / denominator)

        # Save original last action
        probs_last = softmax_probs[..., -1]

        # Get total probability of first two before mixing
        probs_first_two_sum = np.sum(softmax_probs[..., :2], axis=axis, keepdims=True)

        # Mix the first two: new = (1-xi)*orig + xi/2, but RESCALE so their total is unchanged
        mixed_first_two = softmax_probs[..., :2] * (1 - self.xi) + self.xi / 2
        mixed_sum = np.sum(mixed_first_two, axis=axis, keepdims=True)

        # Scale factor
        scaling = np.divide(
            probs_first_two_sum,
            mixed_sum,
            out=np.ones_like(mixed_sum),
            where=(mixed_sum != 0),
        )
        mixed_first_two = mixed_first_two * scaling

        # Stack all together
        final_policy = np.concatenate(
            [mixed_first_two, probs_last[..., None]], axis=axis
        )

        return final_policy

    def calculate_transition_probability(
        self, i: np.ndarray, num_yellows: np.ndarray, num_blues: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the transition probability using the analytical beta-binomial formula.
        Vectorized to handle array inputs.

        Args:
            i (np.ndarray): Number(s) of yellow cards in the next draw (k).
            num_yellows (np.ndarray): Total number of yellow cards observed so far.
            num_blues (np.ndarray): Total number of blue cards observed so far.

        Returns:
            np.ndarray: Transition probability P(i yellow cards in next draw | num_yellows, num_blues).
        """
        alpha_post = self.alpha + num_yellows
        beta_post = self.beta + num_blues
        n = self.max_cards_per_draw

        with np.errstate(divide="ignore", invalid="ignore"):
            log_comb = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
            log_prob = (
                log_comb
                + betaln(alpha_post + i, beta_post + n - i)
                - betaln(alpha_post, beta_post)
            )

        probabilities = np.exp(log_prob)
        valid_mask = (i >= 0) & (i <= self.max_cards_per_draw)
        return np.where(valid_mask, np.nan_to_num(probabilities), 0.0)

    def extract_actions(self) -> None:
        """
        Populate self.best_actions with argmax(self.policy, axis=-1) per state.

        Note: unlike POMDP.extract_actions, this is a plain argmax with no
        tolerance-based tie-breaking (ties resolve to the lowest action index).

        Returns:
            None. Sets self.best_actions in place.
        """
        policy = self.policy
        self.best_actions = np.argmax(policy, axis=-1)

    def calculate_action_values(self) -> None:
        """
        Calculate the action values (Yellow, Blue) for all valid states in the grid.
        This does NOT calculate the 'Wait' action value.

        Returns:
            None. Sets self.action_values[..., 0], self.action_values[..., 1], and
            self.belief[..., 0:2] in place.
        """
        for i_draw, draw in enumerate(self.draw_axis):
            max_cards = draw * self.max_cards_per_draw * self.gamma

            # Find valid indices for this draw
            valid_y_indices = np.where(self.yellow_axis <= max_cards)[0]
            valid_b_indices = np.where(self.blue_axis <= max_cards)[0]

            if len(valid_y_indices) == 0 or len(valid_b_indices) == 0:
                continue

            # Create index meshgrid
            idx_y, idx_b = np.meshgrid(valid_y_indices, valid_b_indices, indexing="ij")

            # --- Filter for valid states (y + b <= max_cards) ---
            grid_y = self.yellow_axis[idx_y]
            grid_b = self.blue_axis[idx_b]
            valid_state_mask = (grid_y + grid_b) <= (max_cards + 1e-5)

            if not np.any(valid_state_mask):
                continue

            valid_idx_y = idx_y[valid_state_mask]
            valid_idx_b = idx_b[valid_state_mask]

            # --- Get beliefs from pre-calculated grid ---
            belief_yellow = self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 0]
            belief_blue = self.belief_grid[i_draw, valid_idx_y, valid_idx_b, 1]

            # --- Reward Calculation (vectorized) ---
            reward_correct = self.calculate_reward(is_correct=True)
            reward_incorrect = self.calculate_reward(is_correct=False)

            # --- Action Value Calculation ---
            yellow_action_value = (
                reward_correct * belief_yellow + reward_incorrect * belief_blue
            )
            blue_action_value = (
                reward_correct * belief_blue + reward_incorrect * belief_yellow
            )

            # --- Assign to the action_values grid ---
            self.action_values[i_draw, valid_idx_y, valid_idx_b, 0] = (
                yellow_action_value
            )
            self.belief[i_draw, valid_idx_y, valid_idx_b, 0] = belief_yellow

            self.action_values[i_draw, valid_idx_y, valid_idx_b, 1] = blue_action_value
            self.belief[i_draw, valid_idx_y, valid_idx_b, 1] = belief_blue

    def get_interpolated_policy(
        self, draw: float, yellow: float, blue: float
    ) -> np.ndarray:
        """
        Interpolates the Q-Values (action_values) for a given state
        and returns the resulting softmax policy.

        Args:
            draw (float): The number of draws (e.g., 3.0).
            yellow (float): The number of yellow cards (e.g., 1.5).
            blue (float): The number of blue cards (e.g., 2.5).

        Returns:
            np.ndarray: A 1D array of shape (3,) containing the
                        probabilities for [Action Y, Action B, Action W].
        """

        # 1. Create the axes for the interpolator
        axes = (self.draw_axis, self.yellow_axis, self.blue_axis)

        # 2. Create the interpolator for the Q-values (self.action_values)
        # We interpolate Q-values, not policies, because interpolating
        # vectors on a simplex (policies) is mathematically problematic.
        # We fill with the deadline_cost for out-of-bounds states.
        q_interpolator = RegularGridInterpolator(
            axes, self.action_values, bounds_error=False, fill_value=self.deadline_cost
        )

        # 3. Define the query point
        pt = np.array([[float(draw), float(yellow), float(blue)]])

        # 4. Get the interpolated Q-values
        interpolated_q_values = q_interpolator(pt)  # Shape (1, 3)

        # 5. Squeeze to a 1D vector
        q_vector = np.squeeze(interpolated_q_values)  # Shape (3,)

        # 6. Apply the softmax policy function to the interpolated Q-values
        # This ensures the output is a valid policy (sums to 1).
        policy_vector = self.softmax_policy(q_vector, axis=-1)

        return policy_vector

    def value_iteration(self) -> None:
        """
        Perform value iteration using interpolation for non-grid future states.

        Returns:
            None. Populates self.action_values, self.value_function, self.policy,
            self.transition_probability, and (via extract_actions) self.best_actions.
        """
        # Calculate Q(s, Y) and Q(s, B) for all states
        self.calculate_action_values()

        # Set Q-value for 'wait' at the terminal step (index -1)
        i_terminal_draw = self.l_draws - 1

        self.action_values[i_terminal_draw, :, :, 2] = (
            self.deadline_cost + self.sigmoidal_cost(self.max_draws)
        )
        # Policy at terminal state (softmax over Q)
        terminal_q_values = self.action_values[i_terminal_draw, :, :, :]
        terminal_policy = self.softmax_policy(terminal_q_values)

        # Store policy
        self.policy[i_terminal_draw, :, :, :] = terminal_policy

        # V(s) = Σ_a π(a|s) Q(s,a)
        terminal_value = np.sum(terminal_policy * terminal_q_values, axis=-1)

        self.value_function[i_terminal_draw, :, :] = terminal_value
        for i_draw in range(self.l_draws - 2, -1, -1):
            draw = self.draw_axis[i_draw]  # Current draw value (e.g., 7, 6, ...)

            # Create an interpolator for V(s')
            # We create a 2D interpolator for this slice.
            v_next_slice = self.value_function[i_draw + 1, :, :]
            axes_2d = (self.yellow_axis, self.blue_axis)

            v_interpolator = RegularGridInterpolator(
                axes_2d,
                v_next_slice,
                bounds_error=False,
                fill_value=np.nan,  # If we go off-grid,nan
            )
            # v_interpolator.values = self.value_function[i_draw + 1]

            #  Define the valid state space for the current draw
            max_cards = draw * self.max_cards_per_draw * self.gamma
            valid_y_indices = np.where(self.yellow_axis <= max_cards)[0]
            valid_b_indices = np.where(self.blue_axis <= max_cards)[0]

            if len(valid_y_indices) == 0 or len(valid_b_indices) == 0:
                continue

            # Create index meshgrid
            idx_y_grid, idx_b_grid = np.meshgrid(
                valid_y_indices, valid_b_indices, indexing="ij"
            )

            # Get the values of the current valid states
            current_y_grid = self.yellow_axis[idx_y_grid]
            current_b_grid = self.blue_axis[idx_b_grid]

            # Filter for valid states (y + b <= max_cards)
            valid_state_mask = (current_y_grid + current_b_grid) <= (max_cards + 1e-5)

            if not np.any(valid_state_mask):
                continue

            # Filter all relevant grids
            valid_idx_y = idx_y_grid[valid_state_mask]
            valid_idx_b = idx_b_grid[valid_state_mask]
            current_y_valid = current_y_grid[valid_state_mask]
            current_b_valid = current_b_grid[valid_state_mask]

            next_yellows_outcomes = np.array(self.next_yellow)
            next_blues_outcomes = self.max_cards_per_draw - next_yellows_outcomes
            n_outcomes = len(next_yellows_outcomes)

            # Broadcast to find all possible future states (values)
            # future_y_grid shape: (n_valid_states, n_outcomes)
            future_y_grid = (
                current_y_valid[..., np.newaxis] + next_yellows_outcomes[np.newaxis, :]
            )
            future_b_grid = (
                current_b_valid[..., np.newaxis] + next_blues_outcomes[np.newaxis, :]
            )

            #  Get V(s') by interpolating
            # We need to query the interpolator at all (y', b') points.
            # v_interpolator expects points as (n_points, 2)

            # Flatten and stack future states for interpolation
            query_points = np.stack(
                [future_y_grid.ravel(), future_b_grid.ravel()], axis=-1
            )

            # Get interpolated values, V(s')
            interpolated_V_next = v_interpolator(query_points)

            # Reshape back to (n_valid_states, n_outcomes)
            expected_future_value = interpolated_V_next.reshape(
                current_y_valid.shape[0], n_outcomes
            )

            transitions = self.calculate_transition_probability(
                i=next_yellows_outcomes[np.newaxis, :],
                num_yellows=current_y_valid[..., np.newaxis],
                num_blues=current_b_valid[..., np.newaxis],
            )  # Shape: (n_valid_states, n_outcomes)

            # Store transitions
            self.transition_probability[i_draw, valid_idx_y, valid_idx_b, :] = (
                transitions
            )

            # Calculate the "wait" action value Q(s, W)
            # Q(s, W) = Cost(s) + sum_s' P(s'|s, W) * V(s')
            # Sum over the "outcomes" axis (axis=-1)
            wait_action_value_undiscounted = np.sum(
                transitions * expected_future_value, axis=-1
            )  # Shape: (n_valid_states,)

            urgency_cost = self.sigmoidal_cost(draw)

            if self.is_hazardous:
                discount_factor = 1 - self.cum_hazard[draw]
                hazard_cost = self.cum_hazard[draw] * self.deadline_cost
                total_wait_value = (
                    discount_factor * wait_action_value_undiscounted
                    + hazard_cost
                    + urgency_cost
                )
            else:
                total_wait_value = wait_action_value_undiscounted + urgency_cost

            # --- 2g. Update Q(s, W), V(s), and Policy(s) for current draw ---

            # Assign Q(s, W)
            self.action_values[i_draw, valid_idx_y, valid_idx_b, 2] = total_wait_value

            # Get all Q-values for the current valid states
            current_q_values = self.action_values[i_draw, valid_idx_y, valid_idx_b]

            current_policy = self.softmax_policy(current_q_values)
            current_value_function_values = np.sum(
                current_policy * current_q_values, axis=-1
            )
            self.value_function[i_draw, valid_idx_y, valid_idx_b] = (
                current_value_function_values
            )

            # Update Policy(s)
            self.policy[i_draw, valid_idx_y, valid_idx_b, :] = current_policy

        self.extract_actions()

    @staticmethod
    def approximate_to_closest(value: float, gamma_values: list) -> float:
        """
        Snap a candidate gamma value to the nearest entry in gamma_values, since
        this class's belief/value grids are only pre-built for a fixed set of gammas.

        Args:
            value (float): Candidate gamma value (e.g. from an optimizer).
            gamma_values (list): Allowed discrete gamma values.

        Returns:
            float: The closest entry in gamma_values, rounded to 2 decimals.
        """
        return np.round(min(gamma_values, key=lambda x: abs(x - value)), 2)

    @staticmethod
    def fit_differential_evolution(
        param_ranges: dict, cost_function: "Callable[[list], float]", x0=None
    ) -> tuple:
        """
        Fit parameters by minimizing cost_function with scipy's differential evolution,
        then snap the fitted gamma to the closest value in gamma_values.

        Args:
            param_ranges (dict): Maps parameter name to a (low, high) bound tuple;
                must include "gamma" for the snapping step below.
            cost_function (Callable[[list], float]): Function to minimize, e.g. one
                built by make_cost_function.

        Returns:
            tuple: (best_params_dict, best_fitness, hessian_placeholder), where
                hessian_placeholder is a 5x5 zero matrix. start_hazard, if present,
                is floored to an int.
        """
        bounds = [param_ranges[k] for k in param_ranges.keys()]

        result = run_differential_evolution(param_ranges, cost_function, x0=x0)
        best_params = result.x
        best_fitness = result.fun
        gamma_index = PARAM_ORDER.index("gamma")
        gamma_value_corrected = POMDP_Forgetting.approximate_to_closest(
            best_params[gamma_index], gamma_values=gamma_values
        )
        best_params[gamma_index] = gamma_value_corrected

        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        if "start_hazard" in best_params_dict.keys():
            best_params_dict["start_hazard"] = int(
                np.floor(best_params_dict["start_hazard"])
            )

        return best_params_dict, best_fitness, np.zeros((5, 5))

    def log_likelihood(self, data: "pd.DataFrame") -> float:
        """
        Sum the log-probability the fitted (interpolated) policy assigns to each
        subject's observed sequence of Wait actions followed by a final choice.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column
                of per-trial sequences, each an iterable of (draw, num_yellow, num_blue,
                action, outcome) tuples with action in {0: Yellow, 1: Blue, 2: Wait}.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        for action_seq in actions_col:
            for draw, y, b, act, _ in action_seq:

                p_y, p_b, p_w = self.get_interpolated_policy(draw, y, b)

                if act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    # Use safe log
                    total_ll += np.log(p_action + 1e-10)
                    break
                elif act == 2:
                    total_ll += np.log(p_w + 1e-10)

        return total_ll

    def log_likelihood_extended(self, data: "pd.DataFrame") -> list:
        """
        Same walk as log_likelihood, but instead of a single summed value, returns
        one record per trial with the full policy breakdown, for diagnostics/plotting.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column,
                as in log_likelihood.

        Returns:
            list: One [draw, y, b, p_y, p_b, p_w, p_chosen, action, ll, cum_ll] row per
                trial, where cum_ll is the running log-likelihood within that trial.
        """
        total_ll = []
        ll = 0
        cum_ll = 0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access
        eps = 1e-10
        for action_seq in actions_col:

            for draw, y, b, act, outcome in action_seq:
                p_y, p_b, p_w = self.get_interpolated_policy(draw, y, b)
                if act == 2:  # wait
                    ll = np.log(p_w + eps)
                    cum_ll = cum_ll + ll

                    total_ll.append([draw, y, b, p_y, p_b, p_w, p_w, act, ll, cum_ll])
                elif act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    ll = np.log(p_action + eps)
                    cum_ll = cum_ll + ll

                    total_ll.append(
                        [draw, y, b, p_y, p_b, p_w, p_action, act, ll, cum_ll]
                    )
                    break  # I need this break, because I have the full sequence and subjects can decide but still see the pre-assigned sequence yet.

        return total_ll

    def make_cost_function(
        self, list_of_evidence_to_fit_dict: dict
    ) -> "Callable[[list], float]":
        """
        Build a cost function suitable for the optimizers below (ga_fit /
        fit_differential_evolution). Unlike POMDP.make_cost_function, the candidate
        gamma is first snapped to the nearest pre-built grid value (approximate_to_closest)
        and used to select which evidence dict to fit against, since this class's
        belief/value grids only exist for a fixed set of gammas.

        Args:
            list_of_evidence_to_fit_dict (dict): Maps gamma value -> {horizon_condition:
                DataFrame of trials}, i.e. one evidence dict per pre-built gamma grid.

        Returns:
            Callable[[list], float]: cost_function(params), where params is a flat
                list ordered per PARAM_ORDER (including "gamma"); returns 1e10 on
                any exception.
        """

        def cost_function(params):
            try:
                gamma_index = PARAM_ORDER.index("gamma")

                # now I have three datasets here in the evidence_to_fit_dict.
                # first I need to choose which one to proceed with.
                gamma_value = POMDP_Forgetting.approximate_to_closest(
                    params[gamma_index], gamma_values=gamma_values
                )
            except:
                gamma_value = 1.0
            evidence_to_fit_dict = list_of_evidence_to_fit_dict[gamma_value]

            # separate the dataset into two, and return each according to their separation. Assume that the data is separated with label long and short
            # I can have a disctionary of the two data data_dict['short']= evidence_to_fit and the other one and then
            try:
                # extract the keys from the evidence_to_fit_dict
                keys = evidence_to_fit_dict.keys()
                # loop over the keys
                # the key is either 'long' or 'short'
                ll = 0
                params = {k: v for k, v in zip(PARAM_ORDER, params)}
                # check if start_hazard is one of the keys of the Param_order, if so, make corresponding value to be intger.
                if "start_hazard" in params:
                    params["start_hazard"] = int(np.floor(params["start_hazard"]))

                params.update(
                    {
                        "verbose": VERBOSE,
                        "max_cards_per_draw": MAX_CARDS_PER_DRAW,
                        "gamma": gamma_value,
                    }
                )
                # loop over the keys and calculate the log likelihood for each key
                _ll_fn = self.log_likelihood_commit if POMDP_COMMIT else self.log_likelihood
                for key in keys:
                    params.update({"horizon_condition": key})

                    data_single_horizon = evidence_to_fit_dict[key]
                    self.__init__(**params)
                    self.value_iteration()
                    ll += _ll_fn(
                        data_single_horizon
                    )  # instead of that one return, I can return the sum of two
                cost = -ll

                return cost
            except Exception as e:
                print(f"[Cost Error] {e}")
                return 1e10

        return cost_function

    @staticmethod
    def ga_fit(param_ranges, cost_function):
        varbound = np.array([param_ranges[k] for k in param_ranges.keys()])
        algorithm_param = get_ga_params()

        model = ga(
            function=cost_function,
            dimension=len(param_ranges.keys()),
            convergence_curve=False,
            progress_bar=False,
            variable_type_mixed=np.array(variable_type),
            variable_boundaries=varbound,
            algorithm_parameters=algorithm_param,
        )

        model.run()

        best_params = model.output_dict["variable"]
        try:
            gamma_index = PARAM_ORDER.index("gamma")
            # now I have three datasets here in the evidence_to_fit_dict.
            # first I need to choose which one to proceed with.
            gamma_value_corrected = POMDP_Forgetting.approximate_to_closest(
                best_params[gamma_index], gamma_values=gamma_values
            )
            best_params[gamma_index] = gamma_value_corrected
        except:
            print("Gamma is inactive")

        best_fitness = model.output_dict["function"]
        best_params_dict = {k: v for k, v in zip(param_ranges.keys(), best_params)}
        if "start_hazard" in best_params_dict.keys():
            best_params_dict["start_hazard"] = int(
                np.floor(best_params_dict["start_hazard"])
            )
        return (
            best_params_dict,
            best_fitness,
            np.zeros((len(param_ranges.keys()), len(param_ranges.keys()))),
        )

    def fit_subject(self, df_ev_simulated, param_ranges, subject_ID, algorithm):
        cost_function = self.make_cost_function(df_ev_simulated)
        if algorithm == "ga":
            best_params, best_cost, hessian_matrix = self.ga_fit(
                param_ranges, cost_function
            )
        if algorithm == "de":
            best_params, best_cost, hessian_matrix = self.fit_differential_evolution(
                param_ranges, cost_function, x0=de_seed_for(subject_ID, param_ranges)
            )
        log_likelihood = -best_cost
        try:
            gamma = best_params["gamma"]
        except:
            gamma = 1.0
        return (
            best_params,
            log_likelihood,
            df_ev_simulated[gamma],
            subject_ID,
            hessian_matrix,
        )

    def simulate_cards_pomdp(
        self,
        given_sequence: bool = False,
        card_sequence: list | None = None,
        start_hazard: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """
        Simulate one trial of the card task: generate (or take a given) sequence of
        yellow/blue draws, then step through it sampling actions from self.policy
        (or self.belief for the interpolated subclasses) until a Yellow/Blue choice
        is made, the deadline is reached, or the hazard-truncated sequence runs out.

        Args:
            given_sequence (bool): If True, replay card_sequence instead of generating
                a random one.
            card_sequence (list, optional): Required when given_sequence is True; a
                sequence of (draw, cum_yellow, cum_blue, action, outcome) tuples as
                recorded from the real experiment.
            start_hazard (int, optional): Draw at which the hazard function starts,
                for the randomly-generated case. Defaults to self.start_hazard.
            seed (int, optional): Seed for np.random, for reproducible simulation.

        Returns:
            dict: Belief trajectory, actions taken, reward, decision, num_draws,
                num_draws_list, max_draws (the realized deadline), and the observed
                yellow/blue counts (both up to the decision and for the full sequence).
        """
        # initalize the lists, and variables
        num_yellows = []
        num_blues = []
        yellow_trace = []
        blue_trace = []
        belief_trajectory = []
        actions = []
        num_draws = 1
        num_draws_list = []
        num_yellow_observed = []

        # Initializing the cards sequence for the case that no sequence is given. This is intialized randomly
        correct_decision = None

        np.random.seed(seed)
        if not given_sequence:
            # Draw a number to determine the number of draws
            u = np.random.random()

            # Here I need to recalculate the hazard even though it could've been calculated in the class.
            # the reason is that, I need the simulate_cards always to be hazardous and random at each draw.
            # but the one in the class is fixed. So that's why I should separate the generation of cards here from
            # what was previously calculated in the class.
            # also no matter whether is_hazardous is true or false, I need to have a hazardous generation of cards.
            # to match the real experiment.

            # first, I could have the start hazard as a free parameter in the future to be fitted.
            if start_hazard is None:
                start_hazard = self.start_hazard

            # but the end_hazard is always the same, i.e., the max draws depending on the horizon condition, long or short
            hazard = calculate_hazard_cum(int(start_hazard), self.max_draws)

            # now this is to calculate the actual deadline that could be smaller than the max_draws
            deadline = np.searchsorted(hazard, u, side="right")
            # I'm generating the trace now like tobias's one

            p_yellow = np.random.uniform(0.4, 0.6)
            while p_yellow == 0.5:
                p_yellow = np.random.uniform(0.4, 0.6)

            yellow_trace = np.random.binomial(
                n=self.max_cards_per_draw, p=p_yellow, size=deadline
            )
            blue_trace = self.max_cards_per_draw - yellow_trace

            # according to the max number of draws, i.e., the deadline, I generate the yellow and blue cards here.

            # yellow_trace = np.array(
            #     [
            #         np.random.randint(0, self.max_cards_per_draw + 1)
            #         for _ in range(deadline)
            #     ]
            # )
            # blue_trace = self.max_cards_per_draw - yellow_trace

            # I just insert zeros at the beginning, so that when I calculate the cumsum, I have a full list of cards.
            yellow_trace = np.insert(yellow_trace, 0, 0)
            blue_trace = np.insert(blue_trace, 0, 0)

            # the belief and policy is always w.r.t. total evidence. So I need the num_yellows, num_blues, total
            num_yellows = np.cumsum(yellow_trace)
            num_blues = np.cumsum(blue_trace)
            correct_decision = 0 if p_yellow > 0.5 else 1

        elif given_sequence:
            # this randomize seed, because if I called the first if within the same code run,
            # the seed will be fixed always, so I just need to re randomize it again here.
            cum_y = np.array([trial[1] for trial in card_sequence], dtype=int)
            cum_b = np.array([trial[2] for trial in card_sequence], dtype=int)

            # add initial 0 to match the other branch (so traces start at draw 0)
            num_yellows = np.insert(cum_y, 0, 0)
            num_blues = np.insert(cum_b, 0, 0)

            # number of draws
            deadline = len(num_blues) - 1
            # extract the reward from the last trial in the sequence, which is the outcome of the decision
            real_reward = card_sequence[-1][-1]
            real_action = card_sequence[-1][-2]
            if real_reward == 2:
                correct_decision = real_action  # the action taken in the last trial
            elif real_reward == -2:
                correct_decision = 1 if real_action == 0 else 0
            else:
                # in case of missing here, the correct action is calculated based on the total number of yellow and total number of blue because I don't know the generative probability.
                correct_decision = 0 if num_yellows[-1] > num_blues[-1] else 1

        # Do the first action outside the loop, if action==2 go inside the loop
        num_yellow_observed.append(num_yellows[num_draws])
        num_draws_list.append(num_draws)
        belief = self.get_interpolated_belief(
            num_draws, num_yellows[num_draws], num_blues[num_draws]
        )[0]
        belief_trajectory.append(belief)  # belief of yellow
        policy = self.get_interpolated_policy(
            num_draws, num_yellows[num_draws], num_blues[num_draws]
        )
        action = self.get_stochastic_action_depending_on_policy(policy)
        actions.append(int(action))

        while action == 2 and num_draws < (len(num_yellows) - 1):
            num_draws += 1
            num_yellow_observed.append(num_yellows[num_draws])
            num_draws_list.append(num_draws)
            belief = self.get_interpolated_belief(
                num_draws, num_yellows[num_draws], num_blues[num_draws]
            )[0]
            belief_trajectory.append(belief)  # belief of yellow
            policy = self.get_interpolated_policy(
                num_draws, num_yellows[num_draws], num_blues[num_draws]
            )
            action = self.get_stochastic_action_depending_on_policy(policy)
            actions.append(int(action))

        belief_trajectory = np.array(belief_trajectory)
        if action == int(2):
            reward = -1  # missed the deadline
        elif action == correct_decision:
            reward = 2
        elif action != correct_decision:
            reward = -2

        results = {
            "trajectory": belief_trajectory,
            "reward": reward,
            "actions": actions,
            "decision": actions[-1],
            "num_draws": num_draws,
            "num_draws_list": num_draws_list,
            "max_draws": deadline,
        }

        results["num_yellows"] = num_yellows[1 : num_draws + 1]
        results["num_blues"] = num_blues[1 : num_draws + 1]
        results["num_yellows_full_sequence"] = num_yellows[1:]
        results["num_blues_full_sequence"] = num_blues[1:]

        return results

    def plot_policy(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot self.policy on the symmetric (yellow-blue difference) grid for this
        non-integer grid POMDP. Outputs one heatmap per action.

        Args:
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            None. Figures are shown (and optionally saved); the mapping of
            action_index -> masked array is currently not returned (see commented-out
            return at the end of this method).
        """

        # --- 1. Get data from the POMDP object ---
        policy = self.policy
        draw_axis = self.draw_axis
        yellow_axis = self.yellow_axis
        blue_axis = self.blue_axis
        step_size = self.step_size
        max_cards_per_draw = self.max_cards_per_draw
        max_yellows_val = self.max_yellows_val

        if policy.ndim != 4:
            raise ValueError(
                "policy must have shape (l_draws, n_yellows, m_blues, n_actions)"
            )

        l_draws, n_yellows, m_blues, n_actions = policy.shape

        action_labels = ["Action Yellow", "Action Blue", "Action Wait"]

        # --- 2. Define the new 'difference' axis ---
        min_diff_val = -max_yellows_val
        max_diff_val = max_yellows_val
        _set_plot_style()

        # Create the axis for (yellow - blue)
        diff_axis = np.arange(min_diff_val, max_diff_val + step_size, step_size)
        n_diffs = len(diff_axis)

        # Create the grid to hold the plot data
        adjusted_probs = np.full((n_actions, l_draws, n_diffs), np.nan, dtype=float)

        # --- 3. Iterate and populate the adjusted grid ---
        for i_draw, draw_val in enumerate(draw_axis):
            max_cards_for_this_draw = draw_val * max_cards_per_draw * self.gamma

            for i_yellow, yellow_val in enumerate(yellow_axis):

                # Check if this yellow value is possible for this draw
                if yellow_val > max_cards_for_this_draw + 1e-5:
                    continue

                # Calculate the corresponding 'blue' value for the "full hand slice"
                blue_val = max_cards_for_this_draw - yellow_val

                # Find the index for this blue value on the blue_axis
                if blue_val < -1e-5:  # Cannot be negative
                    continue

                i_blue = np.searchsorted(blue_axis, blue_val)

                # Check if index is valid and the value is actually on the grid
                if i_blue >= m_blues or not np.isclose(blue_axis[i_blue], blue_val):
                    continue  # This blue_val is not on our grid

                # --- We found a valid (i_draw, i_yellow, i_blue) on the slice ---

                # Calculate the difference value
                diff_val = yellow_val - blue_val

                # Find the corresponding index on the diff_axis
                i_diff = np.searchsorted(diff_axis, diff_val)

                if i_diff >= n_diffs or not np.isclose(diff_axis[i_diff], diff_val):
                    continue

                # Get the policy vector and store it
                for a in range(n_actions):
                    adjusted_probs[a, i_draw, i_diff] = policy[
                        i_draw, i_yellow, i_blue, a
                    ]

        # --- 4. Mask and Plot ---

        # mask invalid entries (NaNs)
        masks = np.isnan(adjusted_probs)
        masked = np.ma.array(adjusted_probs, mask=masks)
        max_val = np.nanmax(adjusted_probs)
        min_val = np.nanmin(adjusted_probs)

        # plotting: one figure per action
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))
            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=min_val,
                vmax=max_val,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Probability")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Yellow - Blue Difference (on 'full hand' slice)")
            ax.set_xlabel("Number of Draws")

            # --- Set X-ticks ---
            ax.set_xticks(np.arange(0, l_draws, 1))
            ax.set_xticklabels(draw_axis.astype(int))

            # --- Set Y-ticks intelligently ---
            tick_labels = []
            tick_indices = []

            max_abs_diff_int = int(np.floor(max_diff_val))
            tick_step = max(1, int(np.ceil(max_abs_diff_int / 5)))
            if tick_step > 1 and tick_step % 2 != 0:
                tick_step += 1  # Prefer even steps like 2, 4, 6

            desired_ticks = np.arange(
                -max_abs_diff_int, max_abs_diff_int + 1, tick_step
            )

            for tick_val in desired_ticks:
                i_tick = np.searchsorted(diff_axis, tick_val)
                if i_tick < len(diff_axis) and np.isclose(diff_axis[i_tick], tick_val):
                    tick_indices.append(i_tick)
                    tick_labels.append(int(tick_val))

            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels)

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
            plt.tight_layout()

            # --- 5. Save Figure ---
            if save_fig:
                os.makedirs(path, exist_ok=True)

                # Use a short label for the action in the filename
                action_fname = action_labels[a].replace(" ", "_").lower()

                base = f"{path}/policy_exaggerate_{action_fname}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        # return dict of masked arrays per action for further inspection
        # return {action_labels[i]: masked[i] for i in range(n_actions)}

    def plot_action_values(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot self.action_values (Q-values) on the symmetric (yellow-blue difference)
        grid for this non-integer grid POMDP. Outputs one heatmap per action.

        Args:
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            None. Figures are shown (and optionally saved); the mapping of
            action_index -> masked array is currently not returned (see commented-out
            return at the end of this method).
        """

        # --- 1. Get data from the POMDP object ---
        action_values = self.action_values
        draw_axis = self.draw_axis
        yellow_axis = self.yellow_axis
        blue_axis = self.blue_axis
        step_size = self.step_size
        max_cards_per_draw = self.max_cards_per_draw
        max_yellows_val = self.max_yellows_val

        if action_values.ndim != 4:
            raise ValueError(
                "policy must have shape (l_draws, n_yellows, m_blues, n_actions)"
            )
        _set_plot_style()

        l_draws, n_yellows, m_blues, n_actions = action_values.shape

        action_labels = ["Action Yellow", "Action Blue", "Action Wait"]
        # --- 2. Define the new 'difference' axis ---
        min_diff_val = -max_yellows_val
        max_diff_val = max_yellows_val

        # Create the axis for (yellow - blue)
        diff_axis = np.arange(min_diff_val, max_diff_val + step_size, step_size)
        n_diffs = len(diff_axis)

        # Create the grid to hold the plot data
        adjusted_probs = np.full((n_actions, l_draws, n_diffs), np.nan, dtype=float)

        # --- 3. Iterate and populate the adjusted grid ---
        for i_draw, draw_val in enumerate(draw_axis):
            max_cards_for_this_draw = draw_val * max_cards_per_draw * self.gamma

            for i_yellow, yellow_val in enumerate(yellow_axis):

                # Check if this yellow value is possible for this draw
                if yellow_val > max_cards_for_this_draw + 1e-5:
                    continue

                # Calculate the corresponding 'blue' value for the "full hand slice"
                blue_val = max_cards_for_this_draw - yellow_val

                # Find the index for this blue value on the blue_axis
                if blue_val < -1e-5:  # Cannot be negative
                    continue

                i_blue = np.searchsorted(blue_axis, blue_val)

                # Check if index is valid and the value is actually on the grid
                if i_blue >= m_blues or not np.isclose(blue_axis[i_blue], blue_val):
                    continue  # This blue_val is not on our grid

                # --- We found a valid (i_draw, i_yellow, i_blue) on the slice ---

                # Calculate the difference value
                diff_val = yellow_val - blue_val

                # Find the corresponding index on the diff_axis
                i_diff = np.searchsorted(diff_axis, diff_val)

                if i_diff >= n_diffs or not np.isclose(diff_axis[i_diff], diff_val):
                    continue

                # Get the policy vector and store it
                for a in range(n_actions):
                    adjusted_probs[a, i_draw, i_diff] = action_values[
                        i_draw, i_yellow, i_blue, a
                    ]

        # --- 4. Mask and Plot ---

        # mask invalid entries (NaNs)
        masks = np.isnan(adjusted_probs)
        masked = np.ma.array(adjusted_probs, mask=masks)
        max_val = np.nanmax(adjusted_probs)
        min_val = np.nanmin(adjusted_probs)

        # plotting: one figure per action
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))
            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=min_val,
                vmax=max_val,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Action Values")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Yellow - Blue Difference (on 'full hand' slice)")
            ax.set_xlabel("Number of Draws")

            # --- Set X-ticks ---
            ax.set_xticks(np.arange(0, l_draws, 1))
            ax.set_xticklabels(draw_axis.astype(int))

            # --- Set Y-ticks intelligently ---
            tick_labels = []
            tick_indices = []

            max_abs_diff_int = int(np.floor(max_diff_val))
            tick_step = max(1, int(np.ceil(max_abs_diff_int / 5)))
            if tick_step > 1 and tick_step % 2 != 0:
                tick_step += 1  # Prefer even steps like 2, 4, 6

            desired_ticks = np.arange(
                -max_abs_diff_int, max_abs_diff_int + 1, tick_step
            )

            for tick_val in desired_ticks:
                i_tick = np.searchsorted(diff_axis, tick_val)
                if i_tick < len(diff_axis) and np.isclose(diff_axis[i_tick], tick_val):
                    tick_indices.append(i_tick)
                    tick_labels.append(int(tick_val))

            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels)

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
            plt.tight_layout()

            # --- 5. Save Figure ---
            if save_fig:
                os.makedirs(path, exist_ok=True)

                # Use a short label for the action in the filename
                action_fname = action_labels[a].replace(" ", "_").lower()

                base = f"{path}/action_prob_heatmap_exaggerate_{action_fname}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)

                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        # return dict of masked arrays per action for further inspection
        # return {action_labels[i]: masked[i] for i in range(n_actions)}


class POMDP_exaggerate(POMDP_Urgency):
    """
    POMDP_Urgency variant using dense NumPy arrays keyed by (draw, prev_yellow,
    curr_yellow) rather than (draw, total_yellow, total_blue). The current draw's
    yellow/blue counts are multiplied by exaggeration_factor when updating the
    belief posterior, so recent evidence is weighted more (or less) heavily than
    evidence already folded into prev_yellow/prev_blue.
    """

    def __init__(
        self,
        tau: float = TAU,
        xi: float = XI,
        horizon_condition: str = HORIZON_CONDITION,
        hazard_lapse: float = HAZARD_LAPSE,
        is_hazardous: bool = IS_HAZARDOUS,
        subjective_cost: float = SUBJECTIVE_COST,
        verbose: bool = VERBOSE,
        max_cards_per_draw: int = MAX_CARDS_PER_DRAW,
        patience: float = PATIENCE,
        urgency_coefficient: float = URGENCY_COEFFICIENT,
        urgency_slope: float = URGENCY_SLOPE,
        c_max: float = C_MAX,
        exaggeration_factor: float = EXAGGERATION_FACTOR,
        belief_bias: float = BELIEF_BIAS,
        gamma: float = GAMMA,
        start_hazard: int | None = None,
    ):
        """
        Initialize a POMDP_Urgency variant using dense NumPy arrays keyed by
        (draw, prev_yellow, curr_yellow) instead of (draw, total_yellow, total_blue),
        where the current draw's yellow/blue counts are exaggerated by
        exaggeration_factor when updating the belief.

        Args:
            tau (float): Softmax temperature for the action policy.
            xi (float): Lapse rate mixed into the Yellow/Blue softmax probabilities.
            horizon_condition (str): "short" or "long"; sets max_draws and the default start_hazard.
            hazard_lapse (float): Mixing weight between short/long hazard curves (see POMDP.__init__).
            is_hazardous (bool): Whether the hazard (deadline) function is applied during planning.
            subjective_cost (float): Added to the incorrect-choice reward.
            verbose (bool): Whether to print progress/diagnostic messages.
            max_cards_per_draw (int): Number of cards revealed per draw.
            patience (float): Midpoint of the urgency sigmoid, in draws.
            urgency_coefficient (float): Lower bound of the urgency sigmoid cost.
            urgency_slope (float): Steepness of the urgency sigmoid.
            c_max (float): Upper bound of the urgency sigmoid cost.
            exaggeration_factor (float): Multiplier applied to the current draw's
                yellow/blue counts (not prior counts) when updating the belief posterior.
            belief_bias (float): Overrides self.beta (the Beta-prior pseudo-count for blue).
            gamma (float): Stored but not otherwise used by this class's value_iteration
                (kept for interface parity with POMDP_Forgetting/POMDP_Exaggeration).
            start_hazard (int, optional): Draw at which the hazard function starts.
                Defaults to 4 (short) or 10 (long) when None.
        """
        # --- 1. Initialize Base Class ---
        super().__init__(
            tau=tau,
            xi=xi,
            horizon_condition=horizon_condition,
            is_hazardous=is_hazardous,
            hazard_lapse=hazard_lapse,
            subjective_cost=subjective_cost,
            verbose=verbose,
            max_cards_per_draw=max_cards_per_draw,
            patience=patience,
            urgency_coefficient=urgency_coefficient,
            urgency_slope=urgency_slope,
            c_max=c_max,
            start_hazard=start_hazard,
        )

        self.exaggeration_factor = exaggeration_factor
        self.gamma = gamma

        if self.verbose:
            print("Initializing vectorized POMDP...")

        # --- 3. Define State Space Dimensions ---
        # are inherited from POMDP_Urgency

        # Max possible yellow total from *previous* draws
        # (happens at draw T, based on T-1 previous draws)
        self.max_prev_y = (self.max_draws - 1) * self.max_cards_per_draw

        # Max possible yellow cards *in current draw* (0 to M)
        self.max_curr_y = self.max_cards_per_draw

        # Define the shape of our state: (Draw, Prev_Y, Curr_Y)
        # We use T+1 because draws are 1-indexed
        # We use +1 for 0-based indexing of card counts
        self.state_shape = (
            self.max_draws + 1,
            self.max_prev_y + 1,
            self.max_curr_y + 1,
        )

        if self.verbose:
            print(f"State shape (T, max_prev_Y, max_curr_Y): {self.state_shape}")
        self.belief_bias = belief_bias
        self.beta = self.belief_bias

        # --- 4. Initialize Dense Arrays ---
        # We use np.nan to mark unreachable/invalid states

        # Q-values: (T, Prev_Y, Curr_Y, Action)
        # Actions: 0=Yellow, 1=Blue, 2=Wait
        self.action_values = np.full(self.state_shape + (3,), np.nan)

        # Beliefs: (T, Prev_Y, Curr_Y, Belief)
        # Beliefs: 0=Belief_Y, 1=Belief_B
        self.belief = np.full(self.state_shape + (2,), np.nan)

        # Value Function V(s) = max_a Q(s,a)
        # (D, Prev_Y, Curr_Y)
        self.value_function = np.full(self.state_shape, np.nan)

        # Softmax Policy
        self.policy = np.full(self.state_shape + (3,), np.nan)

        # Best Actions
        self.best_actions = np.full(self.state_shape, -5, dtype=int)

        self.transition_probability = np.full(self.state_shape, np.nan)

    def calculate_belief_probability(
        self,
        curr_yellow: int,
        curr_blue: int,
        prev_yellow: int = 1,
        prev_blue: int = 1,
        # noise: float = 0,
        q: float = 0.5,
    ) -> float:
        """
        Calculate the probability that the underlying generative probability q_y > q,
        given accumulated (prev) and current-draw yellow/blue counts, using a Beta
        posterior. The current draw's counts are scaled by self.exaggeration_factor
        before being added to the posterior, so a single draw is weighted more (or
        less) heavily than draws already folded into prev_yellow/prev_blue.

        This function works on scalar or array inputs thanks to scipy.stats.

        Args:
            curr_yellow (int): Yellow cards drawn in the current draw.
            curr_blue (int): Blue cards drawn in the current draw.
            prev_yellow (int): Total yellow cards accumulated before the current draw.
            prev_blue (int): Total blue cards accumulated before the current draw.
            q (float): Threshold probability to compare against (default is 0.5).

        Returns:
            float: Posterior probability that the generative probability q_y > q.
        """
        Alpha = self.alpha + prev_yellow + curr_yellow * (self.exaggeration_factor)
        Beta_ = self.beta + prev_blue + curr_blue * (self.exaggeration_factor)

        posterior_prob = 1 - beta.cdf(q, Alpha, Beta_)
        return posterior_prob

    def calculate_action_values_for_Y_and_B(self) -> None:
        """
        Calculate action values Q(s, 0) and Q(s, 1) for all draws.
        This replaces the card_sequence generation.

        Returns:
            None. Sets self.action_values[..., 0:2] and self.belief[..., 0:2] in place.
        """
        if self.verbose:
            print("  Calculating Q(s, Yellow) and Q(s, Blue) for all states...")

        # 1. Handle Draw 1st
        # State is (draw=1, prev_y=0, curr_y)
        draw = 1

        # All possible current draws (0 to M)
        curr_yellow_grid = np.arange(self.max_cards_per_draw + 1)  # Shape (M+1,)
        # Corresponding blue draws
        curr_blue_grid = self.max_cards_per_draw - curr_yellow_grid

        # Priors are (0, 0) for the first draw's belief calculation
        # The Beta(1,1) is added inside calculate_belief_probability
        belief_yellow = self.calculate_belief_probability(
            curr_yellow_grid, curr_blue_grid, 0, 0
        )
        belief_blue = 1.0 - belief_yellow

        # Store beliefs
        self.belief[draw, 0, :] = np.stack([belief_yellow, belief_blue], axis=-1)

        # Get rewards (assuming self.reward is vectorized or constant)
        reward_correct = self.calculate_reward(is_correct=True)
        reward_incorrect = self.calculate_reward(is_correct=False)

        # Calculate Q-values
        Q_y = reward_correct * belief_yellow + reward_incorrect * belief_blue
        Q_b = reward_correct * belief_blue + reward_incorrect * belief_yellow

        self.action_values[draw, 0, :, 0] = Q_y
        self.action_values[draw, 0, :, 1] = Q_b

        # 2. Handle Draws 2 to max_draws
        for draw in range(2, self.max_draws + 1):
            # Max possible prev_y for *this* draw
            max_prev_yellow_d = (draw - 1) * self.max_cards_per_draw

            # Create 2D grids for all valid (prev_y, curr_y) states
            # prev_yellow_grid shape: (max_prev_yellow_d + 1, 1)
            prev_yellow_grid = np.arange(max_prev_yellow_d + 1)[:, np.newaxis]
            # curr_yellow_grid shape: (1, max_cards_per_draw + 1)
            curr_yellow_grid = np.arange(self.max_cards_per_draw + 1)[np.newaxis, :]

            # Use broadcasting to get all derived values
            prev_blue_grid = (draw - 1) * self.max_cards_per_draw - prev_yellow_grid
            curr_blue_grid = self.max_cards_per_draw - curr_yellow_grid

            # Calculate beliefs for all states at once
            belief_yellow = self.calculate_belief_probability(
                curr_yellow_grid, curr_blue_grid, prev_yellow_grid, prev_blue_grid
            )
            belief_blue = 1.0 - belief_yellow

            # Store beliefs
            self.belief[draw, : max_prev_yellow_d + 1, :] = np.stack(
                [belief_yellow, belief_blue], axis=-1
            )

            # Get rewards
            reward_correct = self.calculate_reward(is_correct=True)
            reward_incorrect = self.calculate_reward(is_correct=False)

            # Calculate Q-values
            Q_y = reward_correct * belief_yellow + reward_incorrect * belief_blue
            Q_b = reward_correct * belief_blue + reward_incorrect * belief_yellow

            # Store Q-values
            self.action_values[draw, : max_prev_yellow_d + 1, :, 0] = Q_y
            self.action_values[draw, : max_prev_yellow_d + 1, :, 1] = Q_b

        if self.verbose:
            print("  ... Q-values for Yellow/Blue calculated.")

    def extract_actions_vec(self) -> None:
        """
        Extract the best actions based on the final action_values table (plain
        argmax after replacing NaN/unreachable states with -inf; no tie-breaking).

        Returns:
            None. Sets self.best_actions in place.
        """
        if self.verbose:
            print("Extracting best actions...")

        # Create a copy where NaNs are replaced with -infinity.
        # become [-inf, -inf, -inf], which np.argmax can handle.
        action_values_filled = np.nan_to_num(self.action_values, nan=-np.inf)

        best_action_indices = np.argmax(action_values_filled, axis=-1)

        self.best_actions = best_action_indices
        if self.verbose:
            print("...best actions extracted.")

    def calculate_transition_probability(
        self, i: np.ndarray, num_yellows: np.ndarray, num_blues: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the transition probability using the analytical beta-binomial formula.
        Vectorized to handle array inputs.

        Args:
            i (np.ndarray): Number(s) of yellow cards in the next draw (k).
            num_yellows (np.ndarray): Total number of yellow cards observed so far.
            num_blues (np.ndarray): Total number of blue cards observed so far.

        Returns:
            np.ndarray: Transition probability P(i yellow cards in next draw | num_yellows, num_blues).
        """
        alpha_post = self.alpha + num_yellows
        beta_post = self.beta + num_blues
        n = self.max_cards_per_draw

        with np.errstate(divide="ignore", invalid="ignore"):
            log_comb = gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
            log_prob = (
                log_comb
                + betaln(alpha_post + i, beta_post + n - i)
                - betaln(alpha_post, beta_post)
            )

        probabilities = np.exp(log_prob)
        valid_mask = (i >= 0) & (i <= self.max_cards_per_draw)
        return np.where(valid_mask, np.nan_to_num(probabilities), 0.0)

    def value_iteration(self) -> None:
        """
        Perform backward induction (value iteration) using fully vectorized operations
        over the reachable state space at each draw. Note the loop stops at draw=1
        (draws are 1-indexed here), so self.policy[0, ...] is left as NaN.

        Returns:
            None. Populates self.action_values, self.value_function, self.policy,
            self.transition_probability, and (via extract_actions_vec) self.best_actions.
        """
        if self.verbose:
            print("Starting value iteration...")

        # 1. Calculate Q-values for actions 0 (Yellow) and 1 (Blue)
        # This populates the tables for all time steps.
        self.calculate_action_values_for_Y_and_B()

        # 2. Set terminal state values (Draw = max_draws)
        draw = self.max_draws
        if self.verbose:
            print(f"  Setting terminal values for draw {draw}...")

        # Q(s, Wait) at terminal step
        action_wait = self.deadline_cost + self.sigmoidal_cost(draw)
        self.action_values[draw, ..., 2] = action_wait

        # Fill NaNs with -inf to safely pass to the softmax policy
        # This prevents np.max inside softmax_policy from returning NaN
        action_values_filled_term = np.nan_to_num(
            self.action_values[draw, ...], nan=-np.inf
        )
        terminal_policy = self.softmax_policy(action_values_filled_term, draw=draw)

        self.policy[draw, ...] = terminal_policy
        # V(s) = sum(pi(a|s) * Q(s,a))
        self.value_function[draw, ...] = np.sum(
            terminal_policy * action_values_filled_term, axis=-1
        )

        # 3. Main backward induction loop
        if self.verbose:
            print("  Starting backward induction loop...")

        for draw in range(self.max_draws - 1, 0, -1):

            # Determine the maximum possible 'previous yellow' cards at this specific draw
            max_prev_yellow_d = (draw - 1) * self.max_cards_per_draw

            # Create 2D grids for current states: shape (max_prev_y + 1, M + 1)
            prev_yellow_grid = np.arange(max_prev_yellow_d + 1)[:, np.newaxis]
            curr_yellow_grid = np.arange(self.max_cards_per_draw + 1)[np.newaxis, :]

            # Map current states to the exact 'previous yellow' index in the NEXT draw
            # If I have `prev_y` and draw `curr_y`, my next step's `prev_y` will be their sum.
            next_prev_y_indices = prev_yellow_grid + curr_yellow_grid

            # Calculate the total yellow and blue counts for belief/transition mapping
            total_num_yellows = (
                prev_yellow_grid + self.exaggeration_factor * curr_yellow_grid
            )
            prev_blue_grid = self.max_cards_per_draw * (draw - 1) - prev_yellow_grid
            curr_blue_grid = self.max_cards_per_draw - curr_yellow_grid
            total_num_blues = (
                prev_blue_grid + self.exaggeration_factor * curr_blue_grid
            )

            # Initialize accumulator for the expected value of waiting
            wait_action_expected_value = np.zeros(
                (max_prev_yellow_d + 1, self.max_cards_per_draw + 1)
            )

            # Loop over all possible outcomes of the NEXT draw
            for next_yellow_card in self.next_yellow:

                # Transition probability: P(next_curr_y = next_yellow_card | total_yellows, total_blues)
                transition_probability = self.calculate_transition_probability(
                    i=next_yellow_card,
                    num_yellows=total_num_yellows,
                    num_blues=total_num_blues,
                )
                self.transition_probability[
                    draw + 1, next_prev_y_indices, next_yellow_card
                ] = transition_probability
                # print(draw + 1, next_prev_y_indices, next_yellow_card)

                # Fetch the value of the specific future state from the already computed V(s')
                # Future state is at (draw + 1), with Prev_Y = next_prev_y_indices, and Curr_Y = next_yellow_card
                V_future_specific = self.value_function[
                    draw + 1, next_prev_y_indices, next_yellow_card
                ]

                # Accumulate the expected value element-wise (no np.sum flattening here!)
                wait_action_expected_value += transition_probability * V_future_specific

            # Apply costs to the wait action
            if self.is_hazardous:
                discount_factor = 1 - self.cum_hazard[draw]
                Q_wait = (
                    discount_factor * wait_action_expected_value
                    + self.cum_hazard[draw] * self.deadline_cost
                    + self.sigmoidal_cost(draw)
                )
            else:
                Q_wait = wait_action_expected_value + self.sigmoidal_cost(draw)

            # Store the Q-value for waiting (only for the reachable prev_y states)
            self.action_values[draw, : max_prev_yellow_d + 1, :, 2] = Q_wait

            # Mask out NaNs for the current valid states
            action_values_filled = np.nan_to_num(
                self.action_values[draw, : max_prev_yellow_d + 1, :], nan=-np.inf
            )

            # Update Policy and Value Function for the current draw
            policy = self.softmax_policy(action_values_filled, draw=draw)
            self.policy[draw, : max_prev_yellow_d + 1, :] = policy
            self.value_function[draw, : max_prev_yellow_d + 1, :] = np.sum(
                policy * action_values_filled, axis=-1
            )

        if self.verbose:
            print("  ...backward induction complete.")

        # 4. Extract the best actions
        self.extract_actions_vec()

        if self.verbose:
            print("Value iteration finished.")

    def log_likelihood(self, data: "pd.DataFrame") -> float:
        """
        Sum the log-probability the fitted policy assigns to each subject's observed
        sequence of Wait actions followed by a final Yellow/Blue choice. Unlike the
        base classes, this indexes self.policy by (draw, prev_yellow, curr_yellow)
        rather than (draw, total_yellow, total_blue).

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column
                of per-trial sequences, each an iterable of (draw, cum_yellow, cum_blue,
                action, outcome) tuples with action in {0: Yellow, 1: Blue, 2: Wait}.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access

        for action_seq in actions_col:
            y_old = 0
            b_old = 0
            for draw, y, b, act, _ in action_seq:
                # print(draw,max_draws)
                curr_y = y - y_old
                curr_b = b - b_old
                p_y, p_b, p_w = self.policy[draw, y_old, curr_y]

                if act == 2:  # wait,but not missing
                    total_ll += np.log(p_w + 1e-10)
                elif act == 0 or act == 1:  # choose yellow or blue
                    p_action = p_y if act == 0 else p_b
                    total_ll += np.log(p_action + 1e-10)
                    break
                y_old = y
                b_old = b
        return total_ll

    def log_likelihood_per_draw(self, data: "pd.DataFrame") -> list:
        """Same iteration as log_likelihood, but returns one record per draw
        instead of a sum -- for diagnosing which trials/draws are penalized
        heavily (e.g. when assigning low probability to what the subject
        actually did). Returns a list of dicts with keys: game_idx, draw,
        action ('wait'/'yellow'/'blue'), p_action, ll.
        """
        records = []
        actions_col = data["draw_yellow_blue_action_outcome"].values

        for game_idx, action_seq in enumerate(actions_col):
            y_old = 0
            b_old = 0
            for draw, y, b, act, _ in action_seq:
                curr_y = y - y_old
                curr_b = b - b_old
                p_y, p_b, p_w = self.policy[draw, y_old, curr_y]

                if act == 2:
                    p_action, action_name = p_w, "wait"
                elif act == 0 or act == 1:
                    p_action, action_name = (p_y, "yellow") if act == 0 else (p_b, "blue")
                else:
                    y_old = y
                    b_old = b
                    continue

                records.append({
                    "game_idx": game_idx,
                    "draw": draw,
                    "action": action_name,
                    "p_action": float(p_action),
                    "ll": float(np.log(p_action + 1e-10)),
                })

                if act == 0 or act == 1:
                    break
                y_old = y
                b_old = b
        return records

    def log_likelihood_commit(self, data: "pd.DataFrame") -> float:
        """
        Same as log_likelihood, but collapses Yellow/Blue into a single "commit/go"
        probability (p_y + p_b) so the fit only distinguishes Wait from committing.

        Args:
            data (pd.DataFrame): Must contain a "draw_yellow_blue_action_outcome" column,
                as in log_likelihood.

        Returns:
            float: Total log-likelihood across all trials.
        """
        total_ll = 0.0
        actions_col = data["draw_yellow_blue_action_outcome"].values  # faster access

        for action_seq in actions_col:
            y_old = 0
            b_old = 0
            for draw, y, b, act, _ in action_seq:
                curr_y = y - y_old
                p_y, p_b, p_w = self.policy[draw, y_old, curr_y]
                p_go = p_y + p_b

                if act == 2:  # wait, but not missing
                    total_ll += np.log(p_w + 1e-10)
                elif act == 0 or act == 1:  # choose yellow or blue
                    total_ll += np.log(p_go + 1e-10)
                    break
                y_old = y
                b_old = b
        return total_ll

    # Plotting functions

    def preprocess_for_plotting_loop(self, data: str) -> dict:
        """
        Average self.action_values or self.policy over all (prev_yellow, curr_yellow)
        states that share the same (draw, total_yellow = prev_yellow + curr_yellow),
        collapsing this class's (draw, prev_y, curr_y) state space down to the
        (draw, total_yellow) grid used by the other classes' plots.

        Args:
            data (str): Which table to average: "action_values" or "policy".

        Returns:
            dict: Maps (draw, total_yellow) -> length-3 np.ndarray of averaged
                [Q_yellow/P_yellow, Q_blue/P_blue, Q_wait/P_wait] values.
        """

        # These dicts will hold the running sum and count
        summed_action_values = {}
        state_counts = {}

        T = self.max_draws
        M = self.max_cards_per_draw

        # Loop over all draws (1-indexed)
        for draw in range(1, T + 1):
            # Max prev_y for *this* draw
            max_py_t = (draw - 1) * M

            # Loop over all valid prev_y (0 to max_py_t)
            for py in range(max_py_t + 1):
                # Loop over all valid curr_y (0 to M)
                for cy in range(M + 1):

                    # Get the action values for this state
                    # This is a (3,) array [Qy, Qb, Qw]
                    if data == "action_values":
                        action_vals = self.action_values[draw, py, cy]
                    elif data == "policy":
                        action_vals = self.policy[draw, py, cy]

                    # Check if this state is valid (skips unreachable states)
                    if not np.isnan(action_vals[0]):

                        # This is the grouping key (draw, total_yellow)
                        total_y = py + cy
                        key = (draw, total_y)

                        if key not in summed_action_values:
                            # Initialize with the first valid array found
                            summed_action_values[key] = action_vals
                            state_counts[key] = 1
                        else:
                            # Add to the running sum
                            summed_action_values[key] += action_vals
                            state_counts[key] += 1

        # Now, compute the average for each group
        avg_action_values = {}
        for key, total_sum in summed_action_values.items():
            count = state_counts[key]
            avg_action_values[key] = total_sum / count

        return avg_action_values

    def plot_action_values(
        self,
        action_labels: list | None = None,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> dict:
        """
        Plot self.action_values, averaged over (prev_yellow, curr_yellow) states via
        preprocess_for_plotting_loop, as one heatmap per action keyed by (draw,
        total_yellow - total_blue difference).

        Args:
            action_labels (list, optional): Names of actions. Defaults to ["Action 0", "Action 1", ...].
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            dict: Maps action_label -> masked array of shape (num_draws, n_diffs).
                Empty dict if no valid states were found.
        """
        avg_value_dict = self.preprocess_for_plotting_loop("action_values")

        # --- 1. Extract dimensions from the dictionary ---
        if not avg_value_dict:
            print("Value dictionary is empty. No plots generated.")
            return {}
        _set_plot_style()

        draws = [k[0] for k in avg_value_dict.keys()]
        num_draws = max(draws) + 1  # +1 because draws are 1-indexed

        # Get number of actions from the first value
        n_actions = len(next(iter(avg_value_dict.values())))

        if action_labels is None:
            action_labels = [f"Action {i}" for i in range(n_actions)]
        elif len(action_labels) != n_actions:
            raise ValueError("action_labels length must match number of actions")

        # --- 2. Determine the full range of the Y-axis (the "diff") ---
        max_total_cards = (num_draws - 1) * self.max_cards_per_draw
        max_diff = max_total_cards
        min_diff = -max_total_cards
        diff_range = np.arange(min_diff, max_diff + 1)
        n_diffs = len(diff_range)

        # --- 3. Build the 2D array for plotting ---
        adjusted_values = np.full((n_actions, num_draws, n_diffs), np.nan, dtype=float)

        for (draw, total_yellow), values in avg_value_dict.items():
            total_cards = draw * self.max_cards_per_draw
            total_blue = total_cards - total_yellow
            diff = total_yellow - total_blue
            diff_index = diff - min_diff

            if 0 <= draw < num_draws and 0 <= diff_index < n_diffs:
                adjusted_values[:, draw, diff_index] = values

        # --- 4. Skip draw 0 ---
        adjusted_values = adjusted_values[:, 1:, :]
        num_draws_plot = adjusted_values.shape[1]

        # --- 5. Mask invalid entries and get color limits ---
        masks = np.isnan(adjusted_values)
        masked = np.ma.array(adjusted_values, mask=masks)
        vmin = masked.min()
        vmax = masked.max()

        # --- 6. Plotting ---
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))

            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Average Q-Value")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Total Yellow - Total Blue Difference")
            ax.set_xlabel("Number of Draws")

            # X-axis: display 1, 2, 3, ... instead of 0, 1, 2, ...
            step = max(1, (num_draws_plot - 1) // 10)
            tick_positions = np.arange(0, num_draws_plot, step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions + 1)

            # Y-axis ticks
            num_y_ticks = 15
            tick_indices = np.linspace(0, n_diffs - 1, num_y_ticks, dtype=int)
            tick_labels = diff_range[tick_indices]
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels)

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
            plt.tight_layout()

            if save_fig:
                os.makedirs(path, exist_ok=True)

                base = f"{path}/values_heatmap_exaggerate_{action_labels[a].replace(' ', '_')}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)
                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        return {action_labels[i]: masked[i] for i in range(n_actions)}

    def plot_policy(
        self,
        action_labels: list | None = None,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> dict:
        """
        Plot self.policy, averaged over (prev_yellow, curr_yellow) states via
        preprocess_for_plotting_loop, as one heatmap per action keyed by (draw,
        total_yellow - total_blue difference).

        Args:
            action_labels (list, optional): Names of actions. Defaults to ["Action 0", "Action 1", ...].
            label (str, optional): Extra suffix appended to saved filenames.
            path (str, optional): Directory to save figures into.
            save_fig (bool, optional): Whether to save the figures to disk (.pdf/.svg/.png).

        Returns:
            dict: Maps action_label -> masked array of shape (num_draws, n_diffs).
                Empty dict if no valid states were found.
        """
        avg_value_dict = self.preprocess_for_plotting_loop("policy")
        _set_plot_style()

        # --- 1. Extract dimensions from the dictionary ---
        if not avg_value_dict:
            print("Value dictionary is empty. No plots generated.")
            return {}

        draws = [k[0] for k in avg_value_dict.keys()]
        num_draws = max(draws) + 1  # +1 because draws are 1-indexed

        # Get number of actions from the first value
        n_actions = len(next(iter(avg_value_dict.values())))

        if action_labels is None:
            action_labels = [f"Action {i}" for i in range(n_actions)]
        elif len(action_labels) != n_actions:
            raise ValueError("action_labels length must match number of actions")

        # --- 2. Determine the full range of the Y-axis (the "diff") ---
        max_total_cards = (num_draws - 1) * self.max_cards_per_draw
        max_diff = max_total_cards
        min_diff = -max_total_cards
        diff_range = np.arange(min_diff, max_diff + 1)
        n_diffs = len(diff_range)

        # --- 3. Build the 2D array for plotting ---
        adjusted_values = np.full((n_actions, num_draws, n_diffs), np.nan, dtype=float)

        for (draw, total_yellow), values in avg_value_dict.items():
            total_cards = draw * self.max_cards_per_draw
            total_blue = total_cards - total_yellow
            diff = total_yellow - total_blue
            diff_index = diff - min_diff

            if 0 <= draw < num_draws and 0 <= diff_index < n_diffs:
                adjusted_values[:, draw, diff_index] = values

        # --- 4. Skip draw 0 ---
        adjusted_values = adjusted_values[:, 1:, :]
        num_draws_plot = adjusted_values.shape[1]

        # --- 5. Mask invalid entries and get color limits ---
        masks = np.isnan(adjusted_values)
        masked = np.ma.array(adjusted_values, mask=masks)
        vmin = masked.min()
        vmax = masked.max()

        # --- 6. Plotting ---
        for a in range(n_actions):
            fig, ax = plt.subplots(figsize=(14, 8))

            heat = ax.imshow(
                masked[a].T,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Average Policy")

            ax.set_title(action_labels[a])
            ax.set_ylabel("Total Yellow - Total Blue Difference")
            ax.set_xlabel("Number of Draws")

            # X-axis: display 1, 2, 3, ... instead of 0, 1, 2, ...
            step = max(1, (num_draws_plot - 1) // 10)
            tick_positions = np.arange(0, num_draws_plot, step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_positions + 1)

            # Y-axis ticks
            num_y_ticks = 15
            tick_indices = np.linspace(0, n_diffs - 1, num_y_ticks, dtype=int)
            tick_labels = diff_range[tick_indices]
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels)

            ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
            plt.tight_layout()

            if save_fig:
                os.makedirs(path, exist_ok=True)

                base = f"{path}/action_values_heatmap_exaggerate_{action_labels[a].replace(' ', '_')}"

                if label is not None:
                    base += f"_{label}"

                plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
                plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)
                plt.savefig(
                    base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03
                )

            plt.show()

        return {action_labels[i]: masked[i] for i in range(n_actions)}

    def plot_best_actions(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot the best actions heatmap derived from the policy, with 'Draw' on the x-axis
        and 'Total Yellow - Total Blue Difference' on the y-axis.
        The best action at each state is determined by taking the argmax of the policy.

        Args:
            label (str, optional): Additional label for filename.
            path (str, optional): Directory to save figures.
            save_fig (bool, optional): Whether to save the figures.

        Returns:
            None
        """
        avg_value_dict = self.preprocess_for_plotting_loop("policy")
        _set_plot_style()

        # --- 1. Extract dimensions from the dictionary ---
        if not avg_value_dict:
            print("Value dictionary is empty. No plots generated.")
            return

        draws = [k[0] for k in avg_value_dict.keys()]
        num_draws = max(draws) + 1  # +1 because draws are 1-indexed

        # --- 2. Determine the full range of the Y-axis (the "diff") ---
        max_total_cards = (num_draws - 1) * self.max_cards_per_draw
        max_diff = max_total_cards
        min_diff = -max_total_cards
        diff_range = np.arange(min_diff, max_diff + 1)
        n_diffs = len(diff_range)

        # --- 3. Build the 2D array of best actions (argmax over policy) ---
        adjusted_best_actions = np.full((num_draws, n_diffs), np.nan, dtype=float)

        for (draw, total_yellow), values in avg_value_dict.items():
            total_cards = draw * self.max_cards_per_draw
            total_blue = total_cards - total_yellow
            diff = total_yellow - total_blue
            diff_index = diff - min_diff

            if 0 <= draw < num_draws and 0 <= diff_index < n_diffs:
                adjusted_best_actions[draw, diff_index] = np.argmax(values)

        # --- 4. Skip draw 0 ---
        adjusted_best_actions = adjusted_best_actions[1:, :]
        num_draws_plot = adjusted_best_actions.shape[0]

        # --- 5. Mask invalid entries ---
        mask = np.isnan(adjusted_best_actions)
        masked = np.ma.array(adjusted_best_actions, mask=mask)

        # --- 6. Custom colormap: Yellow=0, Blue=1, Green=2 ---
        from matplotlib.colors import ListedColormap, BoundaryNorm

        action_cmap = ListedColormap(["yellow", "blue", "green"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = BoundaryNorm(bounds, action_cmap.N)

        # --- 7. Plotting ---
        fig, ax = plt.subplots(figsize=(14, 8))

        heatmap = ax.imshow(
            masked.T,
            cmap=action_cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
        )

        # Colorbar legend displayed in a different order (Blue, Wait, Yellow
        # from bottom to top) than the actual action codes used in the heatmap.
        legend_cmap = ListedColormap(["blue", "green", "yellow"])
        legend_mappable = plt.cm.ScalarMappable(cmap=legend_cmap, norm=norm)
        cbar = fig.colorbar(legend_mappable, ax=ax, ticks=[0, 1, 2])
        cbar.set_ticklabels(["Blue (1)", "Wait (2)", "Yellow (0)"])

        ax.set_ylabel("Total Yellow - Total Blue Difference")
        ax.set_xlabel("Number of Draws")

        # X-axis: display 1, 2, 3, ... instead of 0, 1, 2, ...
        step = max(1, (num_draws_plot - 1) // 10)
        tick_positions = np.arange(0, num_draws_plot, step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions + 1)

        # Y-axis ticks
        num_y_ticks = 15
        tick_indices = np.linspace(0, n_diffs - 1, num_y_ticks, dtype=int)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(diff_range[tick_indices])

        ax.grid(which="major", color="w", linestyle="-", linewidth=0.3, alpha=0.35)
        plt.tight_layout()

        if save_fig:
            os.makedirs(path, exist_ok=True)

            base = f"{path}/best_actions_heatmap"

            if label is not None:
                base += f"_{label}"

            plt.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
            plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)
            plt.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03)

        plt.show()

    def plot_best_actions_by_curr_yellow(
        self,
        label: str | None = None,
        path: str = "figures",
        save_fig: bool = True,
    ) -> None:
        """
        Plot the best-actions heatmap without collapsing states that share the
        same (draw, total_yellow) but differ in how the current draw's yellow
        count (curr_yellow) is split from previously-accumulated yellow
        (prev_yellow). Because exaggeration_factor scales curr_yellow before
        it enters the belief update, two states with the same total_yellow
        can have different beliefs -- and thus different best actions --
        depending on this split.

        Same draw-vs-diff layout as plot_best_actions, except each draw's
        single column is subdivided horizontally into up to
        max_cards_per_draw + 1 thin sub-columns, one per possible curr_yellow
        value (0 on the left of the block to max_cards_per_draw on the
        right), each colored by its own best action. The diff (y) axis is
        not subdivided, so this figure is the same height as the standard
        heatmap and only max_cards_per_draw + 1 times wider.

        Args:
            label (str, optional): Additional label for filename.
            path (str, optional): Directory to save figures.
            save_fig (bool, optional): Whether to save the figures.

        Returns:
            None
        """
        # is_latex=False: this plot's tick/label text is new (not already
        # present in matplotlib's tex render cache), and the environment
        # this runs in has no usetex-capable LaTeX install, so usetex=True
        # would fail outright rather than silently falling back.
        _set_plot_style(is_latex=False)

        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Rectangle

        T = self.max_draws
        M = self.max_cards_per_draw
        num_draws_plot = T
        n_sub = M + 1  # curr_yellow = 0..M

        max_total_cards = T * M
        diff_range = np.arange(-max_total_cards, max_total_cards + 1)
        n_diffs = len(diff_range)

        # grid[draw-1, diff_idx, curr_yellow] = best action (0=Y, 1=B, 2=W)
        grid = np.full((num_draws_plot, n_diffs, n_sub), np.nan)

        for draw in range(1, T + 1):
            max_py = (draw - 1) * M
            total_cards = draw * M
            for py in range(max_py + 1):
                for cy in range(n_sub):
                    policy_vec = self.policy[draw, py, cy]
                    if np.isnan(policy_vec[0]):
                        continue
                    total_y = py + cy
                    diff = 2 * total_y - total_cards
                    diff_idx = diff + max_total_cards
                    if 0 <= diff_idx < n_diffs:
                        grid[draw - 1, diff_idx, cy] = np.argmax(policy_vec)

        action_colors = {0: "yellow", 1: "blue", 2: "green"}

        fig, ax = plt.subplots(figsize=(max(14, num_draws_plot * 1.6), 8))

        # Draw each (draw, diff) cell as however many curr_yellow splits are
        # actually valid there, stretched to fill the block's full width --
        # a diff reachable by only one split (e.g. every row at draw 1, where
        # prev_yellow=0 always) renders as one full-width cell, exactly like
        # the non-exaggerated heatmap, rather than a fixed 1/n_sub sliver.
        for draw_idx in range(num_draws_plot):
            block_left = draw_idx
            for diff_idx in range(n_diffs):
                row = grid[draw_idx, diff_idx, :]
                valid_cy = np.where(~np.isnan(row))[0]
                n_valid = len(valid_cy)
                if n_valid == 0:
                    continue
                cell_w = 1.0 / n_valid
                for rank, cy in enumerate(valid_cy):
                    ax.add_patch(
                        Rectangle(
                            (block_left + rank * cell_w, diff_idx - 0.5),
                            cell_w,
                            1.0,
                            facecolor=action_colors[int(row[cy])],
                            edgecolor="none",
                        )
                    )

        ax.set_xlim(0, num_draws_plot)
        ax.set_ylim(-0.5, n_diffs - 0.5)

        # Colorbar legend in Blue-bottom / Wait-middle / Yellow-top order
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = BoundaryNorm(bounds, 3)
        legend_cmap = ListedColormap(["blue", "green", "yellow"])
        legend_mappable = plt.cm.ScalarMappable(cmap=legend_cmap, norm=norm)
        cbar = fig.colorbar(legend_mappable, ax=ax, ticks=[0, 1, 2])
        cbar.set_ticklabels(["Blue", "Wait", "Yellow"])

        # Bold separators between draw blocks only.
        for draw_idx in range(1, num_draws_plot):
            ax.axvline(draw_idx, color="black", linewidth=1.1, alpha=0.8)

        ax.set_xlabel("Number of Draws")
        ax.set_xticks(np.arange(num_draws_plot) + 0.5)
        ax.set_xticklabels(np.arange(1, num_draws_plot + 1))

        # Y-axis ticks, same convention as plot_best_actions.
        num_y_ticks = 15
        tick_indices = np.linspace(0, n_diffs - 1, num_y_ticks, dtype=int)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(diff_range[tick_indices])
        ax.set_ylabel("Total Yellow - Total Blue Difference")
        ax.set_title(
            "Each cell splits by how many curr_yellow values reach that "
            f"total (0-{M} possible, left to right)"
        )

        fig.tight_layout()

        if save_fig:
            os.makedirs(path, exist_ok=True)

            base = f"{path}/best_actions_heatmap_by_curr_yellow"

            if label is not None:
                base += f"_{label}"

            fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.03)
            fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.03)
            fig.savefig(base + ".png", dpi=600, bbox_inches="tight", pad_inches=0.03)

        plt.show()

    def simulate_cards_pomdp(
        self,
        given_sequence: bool = False,
        card_sequence: list | None = None,
        start_hazard: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """
        Simulate one trial of the card task: generate (or take a given) sequence of
        yellow/blue draws, then step through it sampling actions from self.policy
        (or self.belief for the interpolated subclasses) until a Yellow/Blue choice
        is made, the deadline is reached, or the hazard-truncated sequence runs out.

        Args:
            given_sequence (bool): If True, replay card_sequence instead of generating
                a random one.
            card_sequence (list, optional): Required when given_sequence is True; a
                sequence of (draw, cum_yellow, cum_blue, action, outcome) tuples as
                recorded from the real experiment.
            start_hazard (int, optional): Draw at which the hazard function starts,
                for the randomly-generated case. Defaults to self.start_hazard.
            seed (int, optional): Seed for np.random, for reproducible simulation.

        Returns:
            dict: Belief trajectory, actions taken, reward, decision, num_draws,
                num_draws_list, max_draws (the realized deadline), and the observed
                yellow/blue counts (both up to the decision and for the full sequence).
        """
        # initalize the lists, and variables
        num_yellows = []
        num_blues = []
        yellow_trace = []
        blue_trace = []
        belief_trajectory = []
        actions = []
        num_draws = 1
        num_draws_list = []
        num_yellow_observed = []
        correct_decision = None
        # Initializing the cards sequence for the case that no sequence is given. This is intialized randomly
        np.random.seed(seed)
        if not given_sequence:
            # Draw a number to determine the number of draws
            u = np.random.random()

            # Here I need to recalculate the hazard even though it could've been calculated in the class.
            # the reason is that, I need the simulate_cards always to be hazardous and random at each draw.
            # but the one in the class is fixed. So that's why I should separate the generation of cards here from
            # what was previously calculated in the class.
            # also no matter whether is_hazardous is true or false, I need to have a hazardous generation of cards.
            # to match the real experiment.

            # first, I could have the start hazard as a free parameter in the future to be fitted.
            if start_hazard is None:
                start_hazard = self.start_hazard

            # but the end_hazard is always the same, i.e., the max draws depending on the horizon condition, long or short
            hazard = calculate_hazard_cum(int(start_hazard), self.max_draws)

            # now this is to calculate the actual deadline that could be smaller than the max_draws
            deadline = np.searchsorted(hazard, u, side="right")
            # I'm generating the trace now like tobias's one

            p_yellow = np.random.uniform(0.4, 0.6)
            while p_yellow == 0.5:
                p_yellow = np.random.uniform(0.4, 0.6)

            yellow_trace = np.random.binomial(
                n=self.max_cards_per_draw, p=p_yellow, size=deadline
            )
            blue_trace = self.max_cards_per_draw - yellow_trace

            # according to the max number of draws, i.e., the deadline, I generate the yellow and blue cards here.

            # yellow_trace = np.array(
            #     [
            #         np.random.randint(0, self.max_cards_per_draw + 1)
            #         for _ in range(deadline)
            #     ]
            # )
            # blue_trace = self.max_cards_per_draw - yellow_trace

            # I just insert zeros at the beginning, so that when I calculate the cumsum, I have a full list of cards.
            yellow_trace = np.insert(yellow_trace, 0, 0)
            blue_trace = np.insert(blue_trace, 0, 0)

            # the belief and policy is always w.r.t. total evidence. So I need the num_yellows, num_blues, total
            num_yellows = np.cumsum(yellow_trace)
            num_blues = np.cumsum(blue_trace)
            correct_decision = 0 if p_yellow > 0.5 else 1

        elif given_sequence:
            # this randomize seed, because if I called the first if within the same code run,
            # the seed will be fixed always, so I just need to re randomize it again here.
            cum_y = np.array([trial[1] for trial in card_sequence], dtype=int)
            cum_b = np.array([trial[2] for trial in card_sequence], dtype=int)

            # add initial 0 to match the other branch (so traces start at draw 0)
            num_yellows = np.insert(cum_y, 0, 0)
            num_blues = np.insert(cum_b, 0, 0)

            # number of draws
            deadline = len(num_blues) - 1
            # extract the reward from the last trial in the sequence, which is the outcome of the decision
            real_reward = card_sequence[-1][-1]
            real_action = card_sequence[-1][-2]
            if real_reward == 2:
                correct_decision = real_action  # the action taken in the last trial
            elif real_reward == -2:
                correct_decision = 1 if real_action == 0 else 0
            else:
                # in case of missing here, the correct action is calculated based on the total number of yellow and total number of blue because I don't know the generative probability.
                correct_decision = 0 if num_yellows[-1] > num_blues[-1] else 1

        num_yellow_observed.append(num_yellows[num_draws])
        num_draws_list.append(num_draws)
        prev_yellow = num_yellows[num_draws - 1]
        curr_yellow = num_yellows[num_draws] - prev_yellow

        belief_trajectory.append(
            [self.belief[num_draws, prev_yellow, curr_yellow, 0]]
        )  # belief of yellow

        action = self.get_stochastic_action_depending_on_policy(
            self.policy[num_draws, prev_yellow, curr_yellow],
        )
        # print(policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],num_draws,num_yellows[num_draws], num_blues[num_draws],action)
        actions.append(int(action))

        # ensure that the first action is always 2 (wait), otherwise raise an error with zero draws
        while action == 2 and num_draws < (len(num_yellows) - 1):
            num_draws += 1
            num_yellow_observed.append(num_yellows[num_draws])
            prev_yellow = num_yellows[num_draws - 1]
            curr_yellow = num_yellows[num_draws] - prev_yellow
            num_draws_list.append(num_draws)
            belief_trajectory.append(
                [self.belief[num_draws, prev_yellow, curr_yellow, 0]]
            )  # belief of yellow
            action = self.get_stochastic_action_depending_on_policy(
                self.policy[num_draws, prev_yellow, curr_yellow],
            )
            # print(policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],num_draws,num_yellows[num_draws], num_blues[num_draws],action)
            actions.append(int(action))

        belief_trajectory = np.array(belief_trajectory)
        if action == int(2):
            reward = -1  # missed the deadline
        elif action == correct_decision:
            reward = 2
        elif action != correct_decision:
            reward = -2

        results = {
            "trajectory": belief_trajectory,
            "reward": reward,
            "actions": actions,
            "decision": actions[-1],
            "num_draws": num_draws,
            "num_draws_list": num_draws_list,
            "max_draws": deadline,
        }

        results["num_yellows"] = num_yellows[1 : num_draws + 1]
        results["num_blues"] = num_blues[1 : num_draws + 1]
        results["num_yellows_full_sequence"] = num_yellows[1:]
        results["num_blues_full_sequence"] = num_blues[1:]

        return results


# Here is the implementation of the factory pattern.
def POMDPFactory(pomdp_type: str = POMDP_TYPE) -> POMDP:
    """
    Construct a POMDP subclass instance (with default-config parameters) by name.

    Args:
        pomdp_type (str): One of "vanilla", "urgency", "exaggerate",
            "exaggerate_data", "forgetting".

    Returns:
        POMDP: A new instance of the requested subclass, constructed with its
            default (config-module) parameters.

    Raises:
        ValueError: If pomdp_type is not one of the recognized keys.
    """
    pomdp_dic = {
        "vanilla": POMDP,
        "urgency": POMDP_Urgency,
        "exaggerate": POMDP_exaggerate,
        "exaggerate_data": POMDP_Exaggeration,
        "forgetting": POMDP_Forgetting,
    }
    if pomdp_type in pomdp_dic.keys():
        return pomdp_dic[pomdp_type]()
    else:
        raise ValueError(
            f"Invalid pomdp type, the type should be one of the following list {list(pomdp_dic.keys())}"
        )





def _fill_diagonal_gaps(grid):
    """Fill internal NaN gaps along each row with the (averaged) nearest
    valid neighbour(s), leaving the outer triangular NaN regions untouched.
    Mirrors the gap-filling used by plot_best_actions so heatmaps built on
    the same diagonal yellow-blue grid don't show checkerboard white cells.
    """
    for i in range(grid.shape[0]):
        row = grid[i, :]
        valid_indices = np.where(~np.isnan(row))[0]
        if len(valid_indices) == 0:
            continue

        first_valid, last_valid = valid_indices[0], valid_indices[-1]
        segment = row[first_valid : last_valid + 1]
        seg_valid_idx = np.where(~np.isnan(segment))[0]
        seg_invalid_idx = np.where(np.isnan(segment))[0]

        if len(seg_invalid_idx) > 0 and len(seg_valid_idx) > 0:
            distances = np.abs(seg_invalid_idx[:, None] - seg_valid_idx)
            for k_pos, k in enumerate(seg_invalid_idx):
                min_d = distances[k_pos].min()
                tied = seg_valid_idx[distances[k_pos] == min_d]
                segment[k] = segment[tied].mean()

        grid[i, first_valid : last_valid + 1] = segment

    return grid


def _set_plot_style(font_size=20, is_latex=IS_LATEX):
    """Helper to set consistent plot styles."""
    if is_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
            }
        )
    else:
        # Reset to default or specify non-LaTeX styles here if needed
        plt.rcParams.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
            }
        )


colors = ["lightgray", "#3FD24B", "#E92424"]
cmap = mcolors.LinearSegmentedColormap.from_list("RedGreyBlack", colors)


# Single source of truth for the GA settings. These used to be duplicated
# verbatim inside POMDP.ga_fit, POMDP_Forgetting.ga_fit and
# POMDP_Exaggeration.ga_fit, which made them impossible to tune coherently.
GA_ALGORITHM_PARAMS = {
    "max_num_iteration": 200,
    "population_size": 100,
    "mutation_probability": 0.4,
    "elit_ratio": 0.05,
    "crossover_probability": 0.5,
    "parents_portion": 0.5,
    "crossover_type": "uniform",
    "max_iteration_without_improv": None,
}


def get_ga_params() -> dict:
    """Return the GA settings, letting POMDP_GA_PARAMS override any of them.

    The environment variable holds a JSON object merged over
    GA_ALGORITHM_PARAMS, so convergence settings can be swept across a fitting
    run without editing library code (e.g. POMDP_GA_PARAMS='{"population_size":
    400}'). Absent the variable the defaults are returned unchanged.

    Returns:
        dict: Algorithm parameters to hand to geneticalgorithm2.
    """
    params = dict(GA_ALGORITHM_PARAMS)
    override = os.environ.get("POMDP_GA_PARAMS")
    if override:
        params.update(json.loads(override))
    return params


# Differential-evolution settings, validated against the GA on a nested model
# pair (a model whose search space contains another's must not fit worse). The
# GA left ~300 logL on the table for the larger model and violated nesting for
# 17/20 subjects; these settings satisfy it for 20/20 at a comparable evaluation
# budget, and reach an identical optimum from independent seeds. "polish" is the
# key difference: it runs a local L-BFGS-B refinement the GA has no equivalent of.
DE_ALGORITHM_PARAMS = {
    "strategy": "best1bin",
    "popsize": 15,
    "maxiter": 300,
    "tol": 0,
    "mutation": (0.5, 1.0),
    "recombination": 0.9,
    "init": "sobol",
    "polish": True,
}


def get_de_params() -> dict:
    """Return the DE settings, letting POMDP_DE_PARAMS override any of them.

    Mirrors get_ga_params: the environment variable holds a JSON object merged
    over DE_ALGORITHM_PARAMS.

    Returns:
        dict: Keyword arguments for scipy's differential_evolution.
    """
    params = dict(DE_ALGORITHM_PARAMS)
    override = os.environ.get("POMDP_DE_PARAMS")
    if override:
        params.update(json.loads(override))
    if isinstance(params.get("mutation"), list):
        params["mutation"] = tuple(params["mutation"])
    return params


_DE_SEEDS = None


def de_seed_for(subject_ID, param_ranges: dict):
    """Starting point for this subject under the active TASK, or None.

    The table is written by scripts/build_de_seeds.py and named by the
    POMDP_DE_SEEDS environment variable. Each entry is a point the current
    model can occupy (a nested neighbour's fit), so starting there prevents the
    optimizer from returning a worse fit than a model this one contains.

    Args:
        subject_ID: Identifier for the subject being fit.
        param_ranges (dict): Used to check the seed has the expected length.

    Returns:
        list | None: Seed ordered like param_ranges, or None if unavailable.
    """
    global _DE_SEEDS
    path = os.environ.get("POMDP_DE_SEEDS")
    if not path:
        return None
    if _DE_SEEDS is None:
        try:
            import pickle

            with open(path, "rb") as fh:
                _DE_SEEDS = pickle.load(fh)
        except Exception:
            _DE_SEEDS = {}
    vec = _DE_SEEDS.get(TASK, {}).get(str(subject_ID))
    return vec if vec is not None and len(vec) == len(param_ranges) else None


def run_differential_evolution(param_ranges: dict, cost_function, x0=None):
    """Minimize cost_function over param_ranges with scipy differential evolution.

    Args:
        param_ranges (dict): Maps parameter name to a (low, high) bound tuple.
            Iteration order defines the coordinate order of the result.
        cost_function (Callable[[list], float]): Function to minimize.

    Returns:
        scipy.optimize.OptimizeResult: Raw result, so each caller can apply its
            own post-processing (e.g. snapping gamma to the pre-built grid).

    Notes:
        is_hazardous is the one genuinely discrete parameter (0/1). Declaring it
        via `integrality` makes DE search it as a binary gene rather than
        optimizing a continuous value that is rounded afterwards.
    """
    keys = list(param_ranges.keys())
    bounds = [tuple(param_ranges[k]) for k in keys]
    integrality = np.array([1 if k == "is_hazardous" else 0 for k in keys])
    kwargs = dict(get_de_params())
    if x0 is not None:
        # clipped so a seed taken from a model with wider bounds stays feasible
        kwargs["x0"] = np.array(
            [min(max(float(v), lo), hi) for v, (lo, hi) in zip(x0, bounds)],
            dtype=float,
        )
    return differential_evolution(
        cost_function,
        bounds,
        integrality=integrality,
        seed=int(os.environ.get("POMDP_DE_SEED", "0")),
        **kwargs,
    )


