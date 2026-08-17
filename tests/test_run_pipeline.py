"""The pipeline runner: its stage table and the ordering it has to enforce.

Getting the order wrong here is not a crash, it is a stale result that looks
fine, which is the failure mode this table exists to prevent. None of it needs
the dataset.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

spec = importlib.util.spec_from_file_location(
    "run_pipeline", os.path.join(ROOT, "scripts", "run_pipeline.py"))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class TestStageTable(unittest.TestCase):
    def test_every_stage_names_a_script_that_exists(self):
        for name, (_group, script, _env, _req, _desc) in pipeline.STAGES.items():
            if script is None:
                continue                        # the notebook stage
            with self.subTest(stage=name):
                self.assertTrue(
                    os.path.isfile(os.path.join(ROOT, "scripts", script)),
                    f"{name} points at scripts/{script}, which is not here")

    def test_every_stage_is_in_a_known_group(self):
        for name, (group, *_rest) in pipeline.STAGES.items():
            with self.subTest(stage=name):
                self.assertIn(group, pipeline.GROUPS)

    def test_dependencies_refer_to_real_stages(self):
        for name, needs in pipeline.NEEDS.items():
            self.assertIn(name, pipeline.STAGES)
            for dependency in needs:
                self.assertIn(dependency, pipeline.STAGES)

    def test_no_stage_depends_on_itself(self):
        for name, needs in pipeline.NEEDS.items():
            self.assertNotIn(name, needs)


class TestOrdering(unittest.TestCase):
    def test_preprocessing_is_a_strict_chain(self):
        """Step 2 adds the reward and step 3 reads it, so order is not cosmetic."""
        self.assertEqual(pipeline.NEEDS["add_reward"], ("convert_mat",))
        self.assertEqual(pipeline.NEEDS["build_evidence"], ("add_reward",))

    def test_selection_runs_after_the_winners_are_written(self):
        """best_models.json is what makes the downstream exports read the right fits."""
        for stage in ("per_subject_selection", "per_subject_selection_commit"):
            self.assertIn("winners", pipeline.NEEDS[stage])

    def test_ordering_is_honoured_even_when_running_wide(self):
        """A dry run with many jobs must still emit the chain in order."""
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "run_pipeline.py"),
             "preprocess", "--dry-run", "--jobs", "8"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        order = [line.split()[1] for line in out.stdout.splitlines()
                 if line.startswith("=== ")]
        self.assertLess(order.index("convert_mat"), order.index("add_reward"))
        self.assertLess(order.index("add_reward"), order.index("build_evidence"))


class TestMissingEnvironment(unittest.TestCase):
    def test_stages_that_need_a_config_declare_it(self):
        for stage in ("fit", "fit_forgetting", "model_recovery"):
            _group, _script, _env, requires, _desc = pipeline.STAGES[stage]
            self.assertTrue(requires, f"{stage} should declare what it needs")

    def test_a_stage_missing_its_config_is_skipped_not_run(self):
        """Better a message naming the variable than a traceback from inside."""
        env = dict(os.environ)
        env.pop("SIM_CONFIG_PATH", None)
        env.pop("GEN_TASK", None)
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "run_pipeline.py"),
             "--only", "model_recovery"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("skipped, needs", out.stdout)
        self.assertIn("GEN_TASK", out.stdout)


if __name__ == "__main__":
    unittest.main()
