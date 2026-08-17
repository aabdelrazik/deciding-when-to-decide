"""The POMDP itself: construction, solving, policy sanity, simulation.

None of this needs the dataset, so it runs anywhere in about a minute and is
the fastest way to tell whether an install is working.

Action indices are 0 commit yellow, 1 commit blue, 2 wait, which the first test
pins down rather than assuming.
"""
import inspect
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from src.pomdp import POMDPFactory  # noqa: E402

YELLOW, BLUE, WAIT = 0, 1, 2

# The normative setting: every heuristic mechanism at its neutral value and a
# near deterministic policy. Set explicitly rather than relying on class
# defaults, which carry an urgency penalty (urgency_coefficient=-10) that makes
# waiting costly from the first draw.
BASE = dict(horizon_condition="short", max_cards_per_draw=5,
            is_hazardous=True, verbose=False, tau=1e-8, xi=0.0,
            subjective_cost=0.0, patience=0.0, c_max=0.0,
            urgency_coefficient=0.0, urgency_slope=0.0,
            belief_bias=1.0, exaggeration_factor=1.0, gamma=1.0,
            hazard_lapse=0.0)


def build(pomdp_type, **overrides):
    """Instantiate and solve, passing only what this class accepts."""
    m = POMDPFactory(pomdp_type)
    kw = {**BASE, **overrides}
    accepted = set(inspect.signature(type(m).__init__).parameters)
    m.__init__(**{k: v for k, v in kw.items() if k in accepted})
    m.value_iteration()
    return m


class TestVariantsConstructAndSolve(unittest.TestCase):
    """Every model variant in the paper must build and solve."""

    def test_each_variant_solves(self):
        for pomdp_type in ("vanilla", "urgency", "exaggerate", "forgetting"):
            with self.subTest(pomdp_type=pomdp_type):
                m = build(pomdp_type)
                av = np.asarray(m.action_values)
                self.assertEqual(av.shape[-1], 3, "three actions expected")
                self.assertTrue(np.isfinite(av).any())


class TestPolicySanity(unittest.TestCase):
    """Properties the optimal policy must have, whatever the parameters."""

    @classmethod
    def setUpClass(cls):
        cls.m = build("urgency")
        cls.av = np.asarray(cls.m.action_values)

    def test_action_indices_are_yellow_blue_wait(self):
        """Pins the encoding the rest of the suite and the analysis rely on."""
        late = self.m.max_draws - 1
        self.assertEqual(int(np.argmax(self.av[late, 35, 0])), YELLOW)
        self.assertEqual(int(np.argmax(self.av[late, 0, 35])), BLUE)
        self.assertEqual(int(np.argmax(self.av[0, 0, 0])), WAIT)

    def test_no_evidence_means_wait(self):
        """With nothing observed there is no reason to commit."""
        self.assertEqual(int(np.argmax(self.av[0, 0, 0])), WAIT)

    def test_commits_to_the_majority_colour(self):
        """Where it commits at all, it never commits against the evidence."""
        late = self.m.max_draws - 1
        for y, b in ((30, 5), (25, 10), (5, 30), (10, 25)):
            with self.subTest(y=y, b=b):
                a = int(np.argmax(self.av[late, y, b]))
                if a != WAIT:
                    self.assertEqual(a, YELLOW if y > b else BLUE)

    def test_symmetric_evidence_is_symmetric_in_value(self):
        """Swapping the colours must swap the two commit values."""
        late = self.m.max_draws - 1
        a = self.av[late, 24, 11]
        b = self.av[late, 11, 24]
        self.assertAlmostEqual(float(a[YELLOW]), float(b[BLUE]), places=6)
        self.assertAlmostEqual(float(a[BLUE]), float(b[YELLOW]), places=6)


class TestBelief(unittest.TestCase):
    def test_belief_is_one_half_with_no_evidence(self):
        m = build("urgency")
        self.assertAlmostEqual(float(m.calculate_belief_probability(0, 0)), 0.5, places=6)

    def test_belief_increases_with_yellow_evidence(self):
        m = build("urgency")
        # signature is (num_blues, num_yellows)
        weak = float(m.calculate_belief_probability(2, 4))
        strong = float(m.calculate_belief_probability(2, 12))
        self.assertGreater(strong, weak)
        self.assertGreater(weak, 0.5)

    def test_belief_is_symmetric_under_colour_swap(self):
        m = build("urgency")
        y = float(m.calculate_belief_probability(3, 9))
        b = float(m.calculate_belief_probability(9, 3))
        self.assertAlmostEqual(y, 1.0 - b, places=6)


class TestSimulation(unittest.TestCase):
    def test_simulating_a_trial_returns_a_valid_outcome(self):
        m = build("urgency")
        res = m.simulate_cards_pomdp(given_sequence=False, card_sequence=None)
        self.assertIn("reward", res)
        self.assertIn("num_draws", res)
        # +2 correct, -2 incorrect, -1 deadline reached undecided, 0 tie
        self.assertIn(float(res["reward"]), {2.0, -2.0, -1.0, 0.0})
        self.assertGreaterEqual(float(res["num_draws"]), 1)
        self.assertLessEqual(float(res["num_draws"]), m.max_draws)

    def test_replaying_a_fixed_sequence_is_deterministic(self):
        """A near deterministic policy replaying the same cards must repeat.

        Every scoring analysis in the paper depends on this.
        """
        m = build("urgency")
        seq = [[i + 1, 3 * (i + 1), 2 * (i + 1), 2, 0] for i in range(8)]
        first = m.simulate_cards_pomdp(given_sequence=True, card_sequence=seq)
        for _ in range(4):
            again = m.simulate_cards_pomdp(given_sequence=True, card_sequence=seq)
            self.assertEqual(float(again["num_draws"]), float(first["num_draws"]))
            self.assertEqual(float(again["reward"]), float(first["reward"]))


class TestMechanismsChangeBehaviour(unittest.TestCase):
    """Each mechanism must actually do something, or the comparison is empty."""

    def test_subjective_cost_widens_the_decision_boundary(self):
        """A penalty on errors should make the agent wait in at least some
        states where the neutral agent commits."""
        neutral = np.asarray(build("urgency", subjective_cost=0.0).action_values)
        costly = np.asarray(build("urgency", subjective_cost=-50.0).action_values)
        self.assertFalse(np.array_equal(np.argmax(neutral, axis=-1),
                                        np.argmax(costly, axis=-1)))

    def test_exaggeration_factor_changes_the_policy(self):
        plain = np.asarray(build("exaggerate", exaggeration_factor=1.0).action_values)
        exag = np.asarray(build("exaggerate", exaggeration_factor=3.0).action_values)
        self.assertFalse(np.array_equal(plain, exag))


if __name__ == "__main__":
    unittest.main()
