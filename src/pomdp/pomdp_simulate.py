import numpy as np
from src.utils import (
    calculate_hazard_cum,
)
from src.config import IS_HAZARDOUS, EXAGGERATION_FACTOR


def get_stochastic_action_depending_on_policy(
    soft_max_policy_at_current_draw: np.ndarray, draw: int
) -> int:
    """Sample an action from the softmax policy for the current state.

    Args:
        soft_max_policy_at_current_draw (np.ndarray): Softmax action
            probabilities (yellow, blue, wait) for the current
            (draw, num_yellow, num_blue) state; shape (3,).
        draw (int): Current draw index (unused inside this function).

    Returns:
        int: The sampled action (0=yellow, 1=blue, 2=wait).
    """
    actions = [0, 1, 2]
    chosen_action = np.random.choice(actions, p=soft_max_policy_at_current_draw)
    return chosen_action


def simulate_cards_pomdp(
    soft_max_policy: np.ndarray,
    beliefs: np.ndarray,
    horizon_condition: str = "long",
    max_cards_per_draw: int = 5,
    given_sequence: bool = False,
    card_sequence: list | None = None,
    start_hazard: int | None = None,
) -> dict:
    """
    Simulate one trial of the card task: draw a (random or given) sequence of
    yellow/blue cards, then step through it sampling actions from
    `soft_max_policy` until a Yellow/Blue choice is made, the deadline is
    reached, or the sequence runs out.

    Args:
        soft_max_policy (np.ndarray): Softmax action-probability array indexed
            [draw, num_yellow, num_blue] -> probabilities over (yellow, blue,
            wait) actions.
        beliefs (np.ndarray): Belief array indexed
            [draw, num_yellow, num_blue, 0] giving the belief that yellow is
            the majority color.
        horizon_condition (str, optional): "long" or "short"; sets end_hazard
            (14 vs 8) and the default start_hazard (10 vs 4) when generating a
            random sequence. Defaults to "long".
        max_cards_per_draw (int, optional): Max cards drawn per step when
            generating a random sequence. Defaults to 5.
        given_sequence (bool, optional): If True, replay `card_sequence`
            instead of generating a random one. Defaults to False.
        card_sequence (list, optional): Required when given_sequence is True;
            a sequence of [draw_idx, cum_yellow, cum_blue, ...] entries.
        start_hazard (int, optional): Draw at which the hazard function
            starts, for the randomly-generated case. Defaults to 10 (long) or
            4 (short) when None or when IS_HAZARDOUS is False.

    Returns:
        dict: "trajectory" (belief-of-yellow per step), "reward", "actions",
            "confidence", "decision" (final action), "num_draws",
            "num_draws_list", "max_draws" (the realized deadline), and
            "num_yellows"/"num_blues"/"yellow_trace"/"blue_trace" up to the
            decision draw.
    """
    num_yellows = []
    num_blues = []
    yellow_trace = []
    blue_trace = []

    if not given_sequence:
        end_hazard = 14 if horizon_condition == "long" else 8
        # start_hazard = 10 if horizon_condition == "long" else 4
        # Draw a number to predetermine the number of draws
        u = np.random.random()
        # get the hazard
        if start_hazard is None or IS_HAZARDOUS == False:
            start_hazard = 10 if horizon_condition == "long" else 4

        hazard = calculate_hazard_cum(int(start_hazard), end_hazard)
        deadline = np.searchsorted(hazard, u, side="left")
        max_draws = deadline  # to be determine from the previous hazard

        yellow_trace = np.array(
            [np.random.randint(0, max_cards_per_draw + 1) for _ in range(max_draws)]
        )
        blue_trace = max_cards_per_draw - yellow_trace
        yellow_trace = np.insert(yellow_trace, 0, 0)
        blue_trace = np.insert(blue_trace, 0, 0)

        num_yellows = np.cumsum(yellow_trace)
        num_blues = np.cumsum(blue_trace)

    elif given_sequence:
        # card_sequence expected format: list of [draw_idx, cum_yellow, cum_blue, ...]
        # extract cumulative counts (per-draw cumulative, without initial 0)
        cum_y = np.array([trial[1] for trial in card_sequence], dtype=int)
        cum_b = np.array([trial[2] for trial in card_sequence], dtype=int)

        # add initial 0 to match the other branch (so traces start at draw 0)
        cum_y_with0 = np.insert(cum_y, 0, 0)
        cum_b_with0 = np.insert(cum_b, 0, 0)

        # per-draw counts with leading 0 -> shape (n_draws + 1,)
        yellow_trace = np.insert(np.diff(cum_y_with0), 0, 0)
        blue_trace = np.insert(np.diff(cum_b_with0), 0, 0)

        # cumulative counts matching shapes used elsewhere (cumsum of trace)
        num_yellows = np.cumsum(yellow_trace)
        num_blues = np.cumsum(blue_trace)

        # number of draws (max_draws consistent with other branch)
        max_draws = len(blue_trace) - 1

    belief_trajectory = []
    actions = []
    decision = []
    decision.append(0)
    num_draws = 0

    num_draws_list = [num_draws]

    correct_decision = 0 if num_yellows[-1] > num_blues[-1] else 1

    # At least for now make the first step is to draw one card
    action = 2
    # actions.append(int(action))
    num_yellow_observed = []
    # belief_trajectory.append(
    #     [beliefs[num_draws, num_yellows[num_draws], num_blues[num_draws], 0]]
    # )  # belief of yellow
    num_yellow_observed.append(num_yellows[num_draws])
    # ensure that the first action is always 2 (wait), otherwise raise an error with zero draws
    while action == 2 and num_draws < (len(num_yellows) - 1):
        num_draws += 1
        num_yellow_observed.append(num_yellows[num_draws])
        num_draws_list.append(num_draws)
        belief_trajectory.append(
            [beliefs[num_draws, num_yellows[num_draws], num_blues[num_draws], 0]]
        )  # belief of yellow
        action = get_stochastic_action_depending_on_policy(
            soft_max_policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],
            num_draws,
        )
        # print(soft_max_policy[num_draws, num_yellows[num_draws], num_blues[num_draws]],num_draws,num_yellows[num_draws], num_blues[num_draws],action)
        actions.append(int(action))

    belief_trajectory = np.array(belief_trajectory)
    reward = 0  # if there is a tie between both of
    if num_yellows[-1] != num_blues[-1]:
        if action == int(2):
            reward = -1  # missed the deadline
        elif action == correct_decision:
            reward = 2
        elif action != correct_decision:
            reward = -2

    # try and except such that if only one step happended reject the sample and print it
    # what I'm doing is that I'm subtracting the beliefs of yellow minus blue then I add all elements to get to the total evidence. Then the last evidence is just the subtraction of the total belief and the last belief difference
    confidence = np.abs(200 * float(belief_trajectory[-1]) - 100)

    results = {
        "trajectory": belief_trajectory,
        "reward": reward,
        "actions": actions,
        "confidence": confidence,
        "decision": actions[-1],
        "num_draws": num_draws,
        "num_draws_list": num_draws_list,
        "max_draws": max_draws,
    }

    results["num_yellows"] = num_yellows[1 : num_draws + 1]
    results["num_blues"] = num_blues[1 : num_draws + 1]
    results["yellow_trace"] = yellow_trace[1 : num_draws + 1]
    results["blue_trace"] = blue_trace[1 : num_draws + 1]
    return results


# def get_stochastic_action_depending_on_policy(
#     soft_max_policy_at_current_draw, draw, min_prob=0.1
# ):
#     actions = [0, 1, 2]
#     # Clamp small probabilities to zero
#     if draw < 4:
#         clipped_probs = np.where(
#             soft_max_policy_at_current_draw < min_prob,
#             0,
#             soft_max_policy_at_current_draw,
#         )
#         # Renormalize so sum is 1
#         clipped_probs /= clipped_probs.sum()
#         chosen_action = np.random.choice(actions, p=clipped_probs)
#         return chosen_action
#     else:
#         chosen_action = np.random.choice(actions, p=soft_max_policy_at_current_draw)
#         return chosen_action
