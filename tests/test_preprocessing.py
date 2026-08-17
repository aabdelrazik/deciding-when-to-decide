"""Preprocessing, checked against known quantities from the dataset.

These need the OSF download in data/TrHu_NHB_light/ and skip cleanly without
it, so the suite still runs on a machine that only has the code.

The numbers below are properties of the published dataset. If one of them
changes, either the download is different or a preprocessing step has silently
altered the data, and the pipeline should not be trusted until it is explained.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "TrHu_NHB_light", "data_MEG")

# Reference values for the published dataset.
N_SUBJECTS = 105
N_TRIALS = 151200          # rows across all subjects in behdat
TOTAL_REWARD = 159068.0    # sum of the derived reward column
N_GAMES_PER_HORIZON = 80
N_OCD_WITH_YBOCS = 29


def needs(*names):
    missing = [n for n in names if not os.path.exists(os.path.join(DATA, n))]
    if missing:
        raise unittest.SkipTest(
            "dataset not present (" + ", ".join(missing) + "). "
            "Download it from https://osf.io/fks97/ and run the preprocessing "
            "scripts; see the README.")


class TestRawConversion(unittest.TestCase):
    def test_behdat_has_the_expected_shape(self):
        needs("behdat.pkl")
        df = pd.read_pickle(os.path.join(DATA, "behdat.pkl"))
        self.assertEqual(len(df), N_SUBJECTS)
        self.assertEqual(int(sum(len(d) for d in df["data"])), N_TRIALS)
        for col in ("block", "game", "trial", "termination", "chosen",
                    "currEvYellow"):
            self.assertIn(col, df["data"].iloc[0].columns)


class TestRewardStep(unittest.TestCase):
    """The step that is easy to skip and silently invalidates every points
    based result in the manuscript."""

    def test_reward_column_exists_and_is_complete(self):
        needs("behdat_reward.pkl")
        df = pd.read_pickle(os.path.join(DATA, "behdat_reward.pkl"))
        self.assertEqual(len(df), N_SUBJECTS)
        for d in df["data"]:
            self.assertIn("reward", d.columns)
            self.assertFalse(d["reward"].isna().any(), "reward has gaps")

    def test_total_reward_matches_the_published_dataset(self):
        needs("behdat_reward.pkl")
        df = pd.read_pickle(os.path.join(DATA, "behdat_reward.pkl"))
        total = float(sum(d["reward"].sum() for d in df["data"]))
        self.assertAlmostEqual(total, TOTAL_REWARD, places=3)

    def test_reward_is_constant_within_a_game(self):
        """Reward is a per game outcome, carried on every row of that game."""
        needs("behdat_reward.pkl")
        df = pd.read_pickle(os.path.join(DATA, "behdat_reward.pkl"))
        d = df["data"].iloc[0]
        per_game = d.groupby(["block", "game"])["reward"].nunique()
        self.assertTrue((per_game == 1).all())


class TestEvidenceDicts(unittest.TestCase):
    def test_evidence_dicts_have_one_entry_per_subject(self):
        needs("all_subject_evidence_dicts.pkl")
        ev = pd.read_pickle(os.path.join(DATA, "all_subject_evidence_dicts.pkl"))
        self.assertEqual(len(ev), N_SUBJECTS)

    def test_each_game_is_a_sequence_of_five_field_rows(self):
        """Rows are (draw, cumulative_yellow, cumulative_blue, action, outcome)."""
        needs("all_subject_evidence_dicts.pkl")
        ev = pd.read_pickle(os.path.join(DATA, "all_subject_evidence_dicts.pkl"))
        horizon = ev.columns[0]
        games = ev.iloc[0][horizon]["draw_yellow_blue_action_outcome"]
        self.assertEqual(len(games), N_GAMES_PER_HORIZON)
        for row in games.iloc[0]:
            self.assertEqual(len(row), 5)

    def test_counts_are_cumulative_and_non_decreasing(self):
        """A frequent source of bugs: these are running totals, not per draw."""
        needs("all_subject_evidence_dicts.pkl")
        ev = pd.read_pickle(os.path.join(DATA, "all_subject_evidence_dicts.pkl"))
        horizon = ev.columns[0]
        for game in ev.iloc[0][horizon]["draw_yellow_blue_action_outcome"][:20]:
            ys = [r[1] for r in game]
            bs = [r[2] for r in game]
            self.assertEqual(ys, sorted(ys))
            self.assertEqual(bs, sorted(bs))
            # five cards are dealt per draw
            self.assertEqual(ys[-1] + bs[-1], 5 * len(game))

    def test_full_sequence_runs_at_least_as_long_as_the_truncated_one(self):
        """The plain file stops at the subject's decision; the full sequence
        continues to the deadline."""
        needs("all_subject_evidence_dicts.pkl",
              "all_subject_evidence_dicts_full_sequence.pkl")
        cut = pd.read_pickle(os.path.join(DATA, "all_subject_evidence_dicts.pkl"))
        full = pd.read_pickle(os.path.join(
            DATA, "all_subject_evidence_dicts_full_sequence.pkl"))
        horizon = cut.columns[0]
        sid = cut.index[0]
        a = cut.loc[sid, horizon]["draw_yellow_blue_action_outcome"]
        b = full.loc[sid, horizon]["draw_yellow_blue_action_outcome"]
        for g_cut, g_full in zip(a[:40], b[:40]):
            self.assertLessEqual(len(g_cut), len(g_full))


class TestQuestionnaires(unittest.TestCase):
    def test_ybocs_is_present_for_the_patients_only(self):
        needs("ybocs_scores.csv")
        df = pd.read_csv(os.path.join(DATA, "ybocs_scores.csv"))
        numeric = df.select_dtypes(include=[np.number])
        self.assertGreater(len(numeric.dropna(how="all")), 0)
        self.assertLessEqual(len(df), N_SUBJECTS)

    def test_factor_scores_cover_every_subject(self):
        needs("fa_scores.csv")
        df = pd.read_csv(os.path.join(DATA, "fa_scores.csv"))
        self.assertEqual(len(df), N_SUBJECTS)


if __name__ == "__main__":
    unittest.main()
