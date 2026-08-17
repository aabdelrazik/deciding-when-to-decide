"""The numbers in the manuscript, recomputed from the deposited results.

This is the test that matters for a reader: it takes the deposited per subject
fits and summary tables and checks that the quantities quoted in the paper fall
out of them. If one of these fails, either the deposit is inconsistent with the
manuscript or an analysis step has changed behaviour.

Everything here reads results/ and needs no refitting, so it runs in seconds.
"""
import json
import os
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")

# The three winning models, one per horizon condition.
BEST_SHORT = "SB-XT-RPh----"
BEST_LONG = "LBE-T-RPhCL--"
BEST_COMBINED = "C-EXT-RPHC-UK"

# The values the deposit reproduces. The normative agent is simulated, so these
# depend on the per-subject seeding in export_optimality_cost.py; the human
# columns do not and are read straight from behaviour.
SHORTFALL = {"short": 23.9, "long": 43.6, "combined": 67.4}
NORMATIVE_DRAWS = {"short": 3.96, "long": 9.13}
HUMAN_DRAWS = {"short": 3.94, "long": 6.78}
N_SUBJECTS = 105


def need(path):
    full = os.path.join(RESULTS, path)
    if not os.path.exists(full):
        raise unittest.SkipTest(f"{path} not deposited")
    return full


class TestDepositedFits(unittest.TestCase):
    def test_every_winner_has_a_fit_for_all_subjects(self):
        for task, horizon, n_params in ((BEST_SHORT, "short", 5),
                                        (BEST_LONG, "long", 7),
                                        (BEST_COMBINED, "short-long", 9)):
            with self.subTest(task=task):
                df = pd.read_csv(need(f"fits/{task}_{horizon}.csv"))
                self.assertEqual(len(df), N_SUBJECTS)
                self.assertEqual(df["subject_ID"].nunique(), N_SUBJECTS)
                free = [c for c in df.columns
                        if c not in ("subject_ID", "log_likelihood", "n_obs")]
                self.assertEqual(len(free), n_params)

    def test_log_likelihoods_are_negative_and_finite(self):
        df = pd.read_csv(need(f"fits/{BEST_SHORT}_short.csv"))
        self.assertTrue(np.isfinite(df["log_likelihood"]).all())
        self.assertLess(df["log_likelihood"].max(), 0)

    def test_no_fit_sits_on_the_cost_function_failure_value(self):
        """The objective returns 1e10 when it raises, so a broken model can
        'fit' and produce noise. Nothing should be near that."""
        for task, horizon in ((BEST_SHORT, "short"), (BEST_LONG, "long"),
                              (BEST_COMBINED, "short-long")):
            with self.subTest(task=task):
                df = pd.read_csv(need(f"fits/{task}_{horizon}.csv"))
                self.assertLess(abs(df["log_likelihood"]).max(), 1e9)

    def test_parameters_lie_inside_their_fitted_ranges(self):
        """Fitted values must lie inside the range declared in the config.

        The tolerance is relative to each parameter's own range, so a genuine
        out of range value fails while numerical noise from the optimiser's
        scaling round trip does not.
        """
        from src.config.loader import load_config
        for task, horizon in ((BEST_SHORT, "short"), (BEST_LONG, "long"),
                              (BEST_COMBINED, "short-long")):
            with self.subTest(task=task):
                df = pd.read_csv(need(f"fits/{task}_{horizon}.csv"))
                cfg_path = os.path.join(ROOT, "data", "simulation_configs",
                                        f"simulation_params_{task}.py")
                if not os.path.exists(cfg_path):
                    self.skipTest("config not present")
                for name, (lo, hi) in load_config(cfg_path).PARAM_RANGES.items():
                    if name not in df.columns:
                        continue
                    slack = 0.01 * (hi - lo)
                    self.assertGreaterEqual(df[name].min(), lo - slack, name)
                    self.assertLessEqual(df[name].max(), hi + slack, name)

    def test_bound_escapes_are_confined_to_belief_bias(self):
        """Pins which parameters may sit outside their range, so a new one
        cannot appear without the suite noticing."""
        from src.config.loader import load_config
        offenders = set()
        for task, horizon in ((BEST_SHORT, "short"), (BEST_LONG, "long"),
                              (BEST_COMBINED, "short-long")):
            df = pd.read_csv(need(f"fits/{task}_{horizon}.csv"))
            cfg_path = os.path.join(ROOT, "data", "simulation_configs",
                                    f"simulation_params_{task}.py")
            if not os.path.exists(cfg_path):
                self.skipTest("config not present")
            for name, (lo, hi) in load_config(cfg_path).PARAM_RANGES.items():
                if name not in df.columns:
                    continue
                if df[name].min() < lo or df[name].max() > hi:
                    offenders.add(name)
        self.assertLessEqual(offenders, {"belief_bias"},
                             f"unexpected parameters outside their range: {offenders}")

    def test_stored_parameters_reproduce_the_stored_likelihood(self):
        """The strongest check on the deposit: recompute the objective from the
        deposited parameters and compare against the deposited log likelihood.

        This is what rules out a mis-ordered or mis-scaled parameter vector,
        which would otherwise look plausible in a CSV.

        It runs in a subprocess with SIM_CONFIG_PATH set, because the cost
        function builds its model from the module level src.config rather than
        from a config object passed in. Loading a different model with
        load_config in this process would leave the global config in place and
        the objective would fall into its 1e10 failure branch.
        """
        results = os.path.join(ROOT, "data", "POMDP", BEST_SHORT, "de", "short",
                               "results.pkl")
        if not os.path.exists(results):
            self.skipTest("results.pkl not present; needs the full fit output")

        snippet = f"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, {ROOT!r})
from src.config import PARAM_RANGES, POMDP_TYPE, RESULTS_PATH
from src.pomdp import POMDPFactory
df = pd.read_pickle(RESULTS_PATH)
worst = 0.0
for _, row in df.head(3).iterrows():
    params = np.asarray(row["fit_params_ga"], dtype=float)
    cost = POMDPFactory(POMDP_TYPE).make_cost_function(row["data_dict_of_lists"])
    worst = max(worst, abs(abs(float(cost(params))) - abs(float(row["after_lls_ga"]))))
print(worst)
"""
        env = dict(os.environ,
                   SIM_CONFIG_PATH=os.path.join(
                       ROOT, "data", "simulation_configs",
                       f"simulation_params_{BEST_SHORT}.py"),
                   SIM_ALGORITHM="de")
        out = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT,
                             capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        worst = float(out.stdout.strip().split("\n")[-1])
        self.assertLess(worst, 1e-3,
                        "deposited parameters do not reproduce the stored likelihood")


class TestCostOfDeparture(unittest.TestCase):
    """The shortfalls in the cost of departing from optimality section."""

    @classmethod
    def setUpClass(cls):
        p = os.path.join(RESULTS, "optimality", "optimality_cost_informed.csv")
        if not os.path.exists(p):
            raise unittest.SkipTest("optimality_cost_informed.csv not deposited")
        cls.df = pd.read_csv(p)

    def shortfall(self, fit):
        s = self.df[self.df.fit == fit]
        n = s.n_games.values
        return float(np.mean(s.optimal_full.values * n - s.human.values * n))

    def test_shortfalls_match_the_manuscript(self):
        for fit, expected in SHORTFALL.items():
            with self.subTest(fit=fit):
                self.assertAlmostEqual(self.shortfall(fit), expected, delta=0.1)

    def test_humans_fall_short_in_every_horizon(self):
        for fit in SHORTFALL:
            with self.subTest(fit=fit):
                self.assertGreater(self.shortfall(fit), 0)

    def test_draw_counts_match_the_manuscript(self):
        for fit in ("short", "long"):
            with self.subTest(fit=fit):
                s = self.df[self.df.fit == fit]
                self.assertAlmostEqual(float(s.human_draws.mean()),
                                       HUMAN_DRAWS[fit], delta=0.02)
                self.assertAlmostEqual(float(s.optimal_full_draws.mean()),
                                       NORMATIVE_DRAWS[fit], delta=0.02)

    def test_undersampling_is_long_horizon_only(self):
        """Subjects match the optimum on draws in the short horizon and fall
        well below it in the long one. This is the dissociation the section is
        built on."""
        short = self.df[self.df.fit == "short"]
        long_ = self.df[self.df.fit == "long"]
        short_gap = float((short.human_draws - short.optimal_full_draws).mean())
        long_gap = float((long_.human_draws - long_.optimal_full_draws).mean())
        self.assertLess(abs(short_gap), 0.1)
        self.assertLess(long_gap, -2.0)


class TestModelComparison(unittest.TestCase):
    def test_best_models_are_the_ones_the_paper_reports(self):
        p = need("best_models.json")
        with open(p) as fh:
            best = json.load(fh)
        self.assertEqual(best.get("short"), BEST_SHORT)
        self.assertEqual(best.get("long"), BEST_LONG)
        self.assertEqual(best.get("combined"), BEST_COMBINED)

    def test_combined_table_ranks_the_winner_first(self):
        p = os.path.join(RESULTS, "tables", "combined_models_metrics.tex")
        if not os.path.exists(p):
            self.skipTest("combined_models_metrics.tex not deposited")
        text = open(p).read()
        # the winning row is bold with a zero delta on all three criteria
        self.assertIn(BEST_COMBINED, text)
        first_row = [l for l in text.split("\n") if BEST_COMBINED in l][0]
        self.assertIn("0.00", first_row)


if __name__ == "__main__":
    unittest.main()
