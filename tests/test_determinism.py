"""Seeding that has to survive a new interpreter.

Analyses that simulate need a seed derived from something stable. Python salts
the hash of a string per process, so `hash(subject_id)` looks deterministic,
reads as deterministic, and silently changes every run. That defeated the
per-subject seeding in the optimality analysis and made the deposited numbers
irreproducible until it was found.
"""
import os
import subprocess
import sys
import unittest
import zlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class TestStableSeeding(unittest.TestCase):
    def test_builtin_hash_of_a_string_is_not_stable_across_processes(self):
        """The premise. If this ever fails, the guard below can be relaxed."""
        snippet = "print(hash('subject-42'))"
        seen = set()
        for _ in range(4):
            out = subprocess.run([sys.executable, "-c", snippet],
                                 capture_output=True, text=True)
            seen.add(out.stdout.strip())
        self.assertGreater(len(seen), 1,
                           "hash() no longer varies per process on this build")

    def test_crc32_is_stable_across_processes(self):
        snippet = "import zlib; print(zlib.crc32(b'subject-42'))"
        seen = set()
        for _ in range(4):
            out = subprocess.run([sys.executable, "-c", snippet],
                                 capture_output=True, text=True)
            seen.add(out.stdout.strip())
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen.pop(), str(zlib.crc32(b"subject-42")))

    def test_no_script_seeds_from_the_builtin_hash(self):
        """Grep, because the failure is invisible in any single run."""
        offenders = []
        for folder in ("scripts", "src"):
            for dirpath, _dirs, files in os.walk(os.path.join(ROOT, folder)):
                if "__pycache__" in dirpath:
                    continue
                for filename in files:
                    if not filename.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, filename)
                    with open(path, errors="ignore") as handle:
                        for number, line in enumerate(handle, start=1):
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue
                            if "seed(" in stripped and "hash(" in stripped:
                                offenders.append(
                                    f"{os.path.relpath(path, ROOT)}:{number}")
        self.assertEqual(offenders, [], "seed derived from the salted builtin hash")


if __name__ == "__main__":
    unittest.main()
