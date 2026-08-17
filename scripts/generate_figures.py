"""Produce the manuscript's figures and tables, grouped by where they appear.

One entry point instead of thirty. Each function below makes the assets for one
part of the paper, and the sections match the manuscript:

    methods    the task and policy illustrations, which need no fitted data
    results    model comparison, the fits, the GLM comparison, the cost of
               departing from optimality
    appendix   parameter and model recovery, per subject selection, the
               personalisation cost, the symptom scan

Usage (from the repository root):

    python3 scripts/generate_figures.py --list
    python3 scripts/generate_figures.py methods
    python3 scripts/generate_figures.py results appendix
    python3 scripts/generate_figures.py all
    python3 scripts/generate_figures.py --only optimality
    python3 scripts/generate_figures.py all --jobs 8

Everything except `methods` needs fitted results. `--only` takes any target
name from `--list` and runs just that one, which is what you want while
iterating on a single figure.

Targets run in parallel, two thirds of the cores available to the process by
default, which on a shared or batch machine is the allocation rather than the
whole node. `--jobs` overrides it and `--jobs 1` runs them one at a time. Each
target runs its underlying script in a fresh interpreter with the environment
it expects, so a failure in one does not take the rest down, and its output is
captured and printed as one block so parallel runs stay readable. A target that
fails reports its exit code and the exception that caused it rather than only
the code. The exit status is the number of targets that failed.
"""
import argparse
import os
import sys

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
# _staging sits beside this file, so find it from __file__ rather than
# from ROOT, which POMDP_ROOT may point somewhere else entirely
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# a plain module beside the runners, deliberately free of numpy and
# matplotlib so the runner still reports when the analysis stack is broken
from _staging import default_jobs, last_exception, run_script, run_waves  # noqa: E402

# name -> (section, script, extra environment, one line description)
TARGETS = {
    # ---- methods -----------------------------------------------------------
    "policy_illustrations": (
        "methods", "figures/export_policy_illustration_figures.py", {},
        "policy heatmaps, temporal regulation, hazard functions; needs no data"),

    # ---- results -----------------------------------------------------------
    "model_params_table": (
        "results", "figures/export_model_params_table.py", {},
        "the candidate model set and its fitted ranges"),
    "mechanism_ablation": (
        "results", "figures/mechanism_ablation_table.py", {},
        "what each mechanism contributes"),
    "subject_fits": (
        "results", "figures/export_subject_fit_panel.py", {},
        "per subject draw and outcome distributions against the fits"),
    "exaggeration_subgroups": (
        "results", "figures/export_exaggeration_subgroup_figures.py", {},
        "the subjects with the highest fitted exaggeration"),
    "shared_parameters": (
        "results", "figures/shared_parameter_correlation.py", {},
        "short against long horizon parameters, per subject"),
    "optimality_cost": (
        "results", "optimality/export_optimality_cost.py",
        {"INFORMED_PRIOR": "1", "OPTIMAL_ONLY": "1", "OUT_SUFFIX": "_informed"},
        "score every policy in points against the informed normative agent"),
    "optimality": (
        "results", "optimality/export_optimality_figure.py",
        {"OPTIMALITY_SRC": os.path.join(ROOT, "BIC/optimality/optimality_cost_informed.csv")},
        "the cost of departing from optimality figure"),
    "optimality_stats": (
        "results", "optimality/export_optimality_stats.py",
        {"INFORMED_PRIOR": "1",
         "OPTIMALITY_SRC": os.path.join(ROOT, "BIC/optimality/optimality_cost_informed.csv")},
        "every number quoted in the cost of departure paragraph"),
    "glmm_comparison": (
        "results", "glm/export_glmm_comparison_commit.py", {},
        "human against model simulated GLMM coefficients"),
    "glmm_by_horizon": (
        "results", "glm/export_glmm_horizon_comparison.py", {},
        "the same split by horizon condition"),
    "glm_cached": (
        "results", "glm/plot_glm_array_commit_cached.py", {},
        "human against model GLM betas, replayed from the cache"),
    "recency_vs_compulsivity": (
        "results", "glm/export_recency_beta_vs_compulsivity.py", {},
        "the recency regressor against symptom scores"),

    # ---- appendix ----------------------------------------------------------
    "parameter_recovery": (
        "appendix", "recovery/export_parameter_recovery_panel.py", {},
        "true against recovered parameters for the three winners"),
    "model_recovery": (
        "appendix", "recovery/export_model_recovery.py",
        {"MODEL_RECOVERY_TAG": "top5",
         "MODEL_RECOVERY_LISTS": "BIC/_model_recovery_top5_{h}.txt",
         # the manuscript reports the resampled parameter run, whose cells live
         # in their own tree; without this the export reads the superseded pass
         "MODEL_RECOVERY_CELL_DIR": os.path.join(
             ROOT, "data/POMDP_recovery_x4/_model_recovery")},
        "the model recovery confusion matrices"),
    "pxp": (
        "appendix", "comparison/export_pxp_table.py", {},
        "protected exceedance probabilities over the personalised fits"),
    "personalisation_cost": (
        "appendix", "comparison/export_personalization_cost_table.py", {},
        "whether personalising pays for its own selection cost"),
    "fullfit_vs_commit": (
        "appendix", "figures/export_fullfit_vs_commit_params.py", {},
        "full against commit likelihood parameters"),
    "symptom_associations": (
        "appendix", "figures/export_symptom_assoc_figures.py", {},
        "fitted parameters against obsessive compulsive symptom measures"),
    "sensitivity_penalty": (
        "appendix", "sensitivity/export_sensitivity_penalty_table.py", {},
        "how the winning margin moves under a penalty for the fixed temporal "
        "regulation parameters"),
    "sensitivity_sweep": (
        "appendix", "sensitivity/export_sensitivity_sweep_table.py", {},
        "the fixed value sweep: chosen value against the one a direct search "
        "finds, and the BIC cost of the difference"),
    "task_figure": (
        "methods", "figures/export_task_figure.py", {},
        "the task schematic: one evidence sample and the response screen"),
    "human_data": (
        "methods", "figures/export_human_data_figures.py", {},
        "commitment fractions at each evidence state, and four participants "
        "against their own fitted ensemble; the one methods target needing data"),
}

SECTIONS = ("methods", "results", "appendix")

# Targets that must run after another, because they read what it writes:
# export_optimality_cost.py produces the csv the figure and the stats consume.
# Everything else is independent and can run in any order.
NEEDS = {
    "optimality": ("optimality_cost",),
    "optimality_stats": ("optimality_cost",),
}


def run(name):
    """Run one target in its own interpreter. Returns (name, ok, cause)."""
    section, script, extra, description = TARGETS[name]
    header = f"\n=== {name}  ({section})\n    {description}"
    return run_script(name, script, ROOT, extra, header)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="*", default=[],
                        help="any of: " + ", ".join(SECTIONS) + ", all")
    parser.add_argument("--only", metavar="TARGET",
                        help="run a single target by name")
    parser.add_argument("--list", action="store_true",
                        help="show every target and exit")
    parser.add_argument("--jobs", type=int, default=None, metavar="N",
                        help="how many targets to run at once "
                             f"(default {default_jobs()}, two thirds of the "
                             "cores available to this process)")
    args = parser.parse_args()

    if args.list:
        for section in SECTIONS:
            print(f"\n{section}")
            for name, (sec, script, _env, description) in TARGETS.items():
                if sec == section:
                    print(f"  {name:24s} {description}")
                    print(f"  {'':24s}   {script}")
        return 0

    if args.only:
        if args.only not in TARGETS:
            parser.error(f"unknown target {args.only!r}; see --list")
        wanted = [args.only]
    else:
        sections = args.sections or ["all"]
        if "all" in sections:
            sections = list(SECTIONS)
        for section in sections:
            if section not in SECTIONS:
                parser.error(f"unknown section {section!r}; see --list")
        wanted = [n for n, (sec, *_rest) in TARGETS.items() if sec in sections]

    jobs = args.jobs if args.jobs and args.jobs > 0 else default_jobs()
    jobs = min(jobs, len(wanted))
    print(f"{len(wanted)} target(s), {jobs} at a time")

    # In waves, so a target never starts before what it reads.
    causes = run_waves(wanted, jobs, NEEDS, run)

    failed = [n for n in wanted if causes.get(n) is not None]
    print(f"\n{len(wanted) - len(failed)} of {len(wanted)} targets succeeded")
    for name in failed:
        upstream = [d for d in NEEDS.get(name, ()) if d in failed]
        note = f"  (after {', '.join(upstream)} failed)" if upstream else ""
        print(f"  {name}: {causes[name]}{note}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
