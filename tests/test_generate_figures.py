"""The figure runner: its target table, how it parallelises, and how it fails.

The runner is the one entry point a reader is told to use, so the table has to
name scripts that exist, and a failure has to say what went wrong rather than
only that something did. None of this needs the dataset.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "make_figures", os.path.join(ROOT, "scripts", "generate_figures.py"))
make_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_figures)


class TestTargetTable(unittest.TestCase):
    def test_every_target_names_a_script_that_exists(self):
        for name, (_section, script, _env, _desc) in make_figures.TARGETS.items():
            with self.subTest(target=name):
                self.assertTrue(
                    os.path.isfile(os.path.join(ROOT, "scripts", script)),
                    f"{name} points at scripts/{script}, which is not here")

    def test_every_target_is_in_a_known_section(self):
        for name, (section, *_rest) in make_figures.TARGETS.items():
            with self.subTest(target=name):
                self.assertIn(section, make_figures.SECTIONS)

    def test_dependencies_refer_to_real_targets(self):
        """A typo in NEEDS would silently drop the ordering it exists to enforce."""
        for name, needs in make_figures.NEEDS.items():
            self.assertIn(name, make_figures.TARGETS)
            for dependency in needs:
                self.assertIn(dependency, make_figures.TARGETS)

    def test_optimality_runs_after_the_cost_that_feeds_it(self):
        """The figure and the stats both read the csv export_optimality_cost writes."""
        for name in ("optimality", "optimality_stats"):
            self.assertIn("optimality_cost", make_figures.NEEDS.get(name, ()))


class TestJobCount(unittest.TestCase):
    def test_default_is_two_thirds_of_the_available_cores(self):
        available = len(os.sched_getaffinity(0))
        self.assertEqual(make_figures.default_jobs(), max(1, available * 2 // 3))

    def test_default_is_at_least_one_and_never_more_than_available(self):
        available = len(os.sched_getaffinity(0))
        self.assertGreaterEqual(make_figures.default_jobs(), 1)
        self.assertLessEqual(make_figures.default_jobs(), available)


class TestFailureReporting(unittest.TestCase):
    def test_exception_line_is_pulled_out_of_a_traceback(self):
        captured = (
            "some progress output\n"
            "Traceback (most recent call last):\n"
            '  File "scripts/recovery/export_model_recovery.py", line 268, in <module>\n'
            "    raise SystemExit(...)\n"
            "KeyError: 'GEN_TASK'\n"
        )
        self.assertEqual(make_figures.last_exception(captured), "KeyError: 'GEN_TASK'")

    def test_empty_output_still_gives_a_cause(self):
        self.assertEqual(make_figures.last_exception(""), "no output")

    def test_a_failing_target_reports_the_reason_not_just_the_code(self):
        """Run the real runner against a target rigged to fail, and read stdout.

        The earlier version inherited the child's streams, so the traceback
        landed hundreds of lines away from the target it belonged to and the
        summary said only 'exit code 1'.
        """
        env = dict(os.environ)
        env["POMDP_ROOT"] = ROOT
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "generate_figures.py"),
             "--only", "model_recovery", "--jobs", "1"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        # It fails without its inputs, which is the case we want to see reported.
        if out.returncode == 0:
            self.skipTest("model recovery inputs are present, nothing failed")
        self.assertIn("FAILED, exit code", out.stdout)
        self.assertIn("model_recovery", out.stdout)
        # the summary line carries a cause, not a bare number
        summary = [line for line in out.stdout.splitlines()
                   if line.strip().startswith("model_recovery:")]
        self.assertTrue(summary, f"no explained failure in:\n{out.stdout[-2000:]}")
        self.assertRegex(summary[-1], r"(Error|SystemExit|no output|not )")


if __name__ == "__main__":
    unittest.main()
