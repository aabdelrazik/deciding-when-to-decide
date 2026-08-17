"""The configuration layer: root resolution, defaults, and output paths.

These are the settings that silently change which results you read, so they are
worth pinning. None of this needs the dataset.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


class TestProjectRoot(unittest.TestCase):
    def test_root_resolves_from_any_working_directory(self):
        """PROJECT_ROOT must come from the file location, not the cwd.

        This used to be os.getcwd()/.. which forced every script to be run from
        scripts/ and failed confusingly anywhere else.
        """
        snippet = (
            "import sys; sys.path.insert(0, %r)\n"
            "from src.config import CONFIG\n"
            "print(CONFIG.PROJECT_ROOT)\n" % ROOT
        )
        for cwd in (ROOT, os.path.join(ROOT, "scripts"), "/tmp"):
            with self.subTest(cwd=cwd):
                out = subprocess.run([sys.executable, "-c", snippet], cwd=cwd,
                                     capture_output=True, text=True)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.strip(), ROOT)

    def test_pomdp_root_env_overrides(self):
        snippet = (
            "import sys; sys.path.insert(0, %r)\n"
            "from src.config import CONFIG\n"
            "print(CONFIG.PROJECT_ROOT)\n" % ROOT
        )
        env = dict(os.environ, POMDP_ROOT="/tmp/somewhere-else")
        out = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT,
                             capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "/tmp/somewhere-else")


class TestDefaults(unittest.TestCase):
    def test_default_optimizer_is_de(self):
        """Every reported fit uses differential evolution. A 'ga' default sends
        readers to the superseded first pass without any error."""
        from src.config.schema import SimulationConfig
        self.assertEqual(SimulationConfig.ALGORITHM, "de")

    def test_full_likelihood_is_the_default(self):
        from src.config.schema import SimulationConfig
        self.assertFalse(SimulationConfig.POMDP_COMMIT)


class TestOutputPaths(unittest.TestCase):
    def test_commit_and_full_never_share_a_directory(self):
        """A commit fit must not be able to overwrite a full fit."""
        from src.config.loader import load_config
        cfg_dir = os.path.join(ROOT, "data", "simulation_configs")
        base = os.path.join(cfg_dir, "simulation_params_SB-XT-RPh----.py")
        commit = os.path.join(cfg_dir, "simulation_params_SB-XT-RPh----_commit.py")
        if not (os.path.exists(base) and os.path.exists(commit)):
            self.skipTest("configs not present")
        a = load_config(base)
        b = load_config(commit)
        self.assertNotEqual(a.DATA_PATH, b.DATA_PATH)
        self.assertIn("POMDP_commit", b.DATA_PATH)
        self.assertNotIn("POMDP_commit", a.DATA_PATH)

    def test_algorithm_is_part_of_the_output_path(self):
        """de and ga results are siblings, never the same directory."""
        from src.config.loader import load_config
        p = os.path.join(ROOT, "data", "simulation_configs",
                         "simulation_params_SB-XT-RPh----.py")
        if not os.path.exists(p):
            self.skipTest("configs not present")
        os.environ["SIM_ALGORITHM"] = "de"
        de = load_config(p).DATA_PATH
        os.environ["SIM_ALGORITHM"] = "ga"
        ga = load_config(p).DATA_PATH
        os.environ["SIM_ALGORITHM"] = "de"
        self.assertNotEqual(de, ga)
        self.assertIn("/de", de)
        self.assertIn("/ga", ga)


if __name__ == "__main__":
    unittest.main()
