"""Shared machinery for the two stage runners.

`scripts/generate_figures.py` and `scripts/run_pipeline.py` both run a table of
other scripts, each in its own interpreter, and both need the same three things:
a sensible default width, output that stays readable when stages overlap, and a
failure that says what went wrong rather than only that something did.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def default_jobs():
    """Two thirds of the cores this process may actually use.

    sched_getaffinity reports the allocation on a batch node or inside a
    container, which is the number that matters; os.cpu_count() would report the
    whole machine and oversubscribe it.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:                      # not Linux
        available = os.cpu_count() or 1
    return max(1, available * 2 // 3)


def last_exception(text):
    """Pull the final traceback line out of captured output, for the summary."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        # the last line of a traceback, e.g. "KeyError: 'GEN_TASK'", and the
        # SystemExit("...") form scripts use to refuse missing inputs
        if line[:1].isupper() and ": " in line and not line.startswith("  "):
            return line
    return lines[-1] if lines else "no output"


def run_script(name, script, root, extra_env, header):
    """Run one script in its own interpreter.

    Returns (name, ok, cause). Output is captured rather than inherited so that
    concurrent stages do not interleave, and so a failure can be explained
    instead of reported as a bare exit code.
    """
    env = dict(os.environ)
    env.setdefault("SIM_ALGORITHM", "de")
    env.update(extra_env)
    result = subprocess.run([sys.executable, os.path.join(root, "scripts", script)],
                            cwd=root, env=env, capture_output=True, text=True)

    block = [header]
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        block.append(output.rstrip())
    cause = None
    if result.returncode != 0:
        cause = last_exception(output)
        block.append(f"    FAILED, exit code {result.returncode}: {cause}")
        block.append(f"    reproduce with: python3 scripts/{script}")
    print("\n".join(block), flush=True)
    return name, result.returncode == 0, cause


def run_waves(wanted, jobs, needs, runner):
    """Run `wanted` respecting `needs`, up to `jobs` at a time.

    A prerequisite outside the selected set counts as satisfied, since running
    one section or a single target are both legitimate.
    """
    causes, remaining = {}, list(wanted)
    while remaining:
        ready = [n for n in remaining
                 if all(d in causes or d not in wanted for d in needs.get(n, ()))]
        if not ready:                           # nothing can start; run the rest
            ready = list(remaining)
        if jobs == 1:
            results = [runner(n) for n in ready]
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                results = list(pool.map(runner, ready))
        for name, ok, cause in results:
            causes[name] = None if ok else cause
        remaining = [n for n in remaining if n not in ready]
    return causes
