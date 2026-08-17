"""Run the analysis pipeline, grouped by stage.

The front door for everything that is not a figure. `scripts/generate_figures.py`
produces the manuscript's assets from finished results; this produces the
results. The groups run in the order the analysis requires:

    preprocess   the .mat files to evidence dictionaries, and the gamma grids
    fit          differential evolution fits, the seeded second pass, the merge
    ensembles    re-simulate every subject from their fitted parameters
    comparison   per-subject-summed BIC, the winners, per subject selection
    recovery     parameter recovery and model recovery
    glm          the human and model regressions the POMDP is compared against
    optimality   score every policy in the task's own points
    sensitivity  sweep the fixed temporal regulation values

Usage (from the repository root):

    python3 scripts/run_pipeline.py --list
    python3 scripts/run_pipeline.py preprocess
    python3 scripts/run_pipeline.py comparison recovery
    python3 scripts/run_pipeline.py --only per_subject_selection
    python3 scripts/run_pipeline.py all --dry-run

Stages run in order, because each reads what the one before it writes. Only
stages explicitly marked independent share a group and can overlap, which is
what `--jobs` widens; the default is two thirds of the cores available to this
process.

Several stages describe one model rather than the whole set, so they need
`SIM_CONFIG_PATH` (or `GEN_TASK`) to say which. Those are skipped with a message
naming what is missing rather than failing halfway in. Fitting all 78 models is
a job array, not a loop here; `slurm/` holds the templates.
"""
import argparse
import os
import sys

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
# _staging sits beside this file, so find it from __file__ rather than
# from ROOT, which POMDP_ROOT may point somewhere else entirely
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# a plain module beside the runners, deliberately free of numpy and
# matplotlib so the runner still reports when the analysis stack is broken
from _staging import default_jobs, run_script, run_waves  # noqa: E402

GROUPS = ("preprocess", "fit", "ensembles", "comparison", "recovery", "glm",
          "optimality", "sensitivity")

# name -> (group, script, extra env, needs these env vars set, description)
STAGES = {
    # ---- preprocess --------------------------------------------------------
    "convert_mat": (
        "preprocess", "preprocess/preprocess_01_convert_mat.py", {}, (),
        "behdat.mat to behdat.pkl"),
    "add_reward": (
        "preprocess", "preprocess/preprocess_02_add_reward.py", {}, (),
        "merge sNdraws8.mat for the reward; everything in points needs this"),
    "build_evidence": (
        "preprocess", "preprocess/preprocess_03_build_evidence.py", {}, (),
        "the evidence dictionaries the fitting reads"),
    "gamma_grids": (
        "preprocess", "preprocess/generate_gamma_grids.py", {}, (),
        "discounted evidence grids, needed only by forgetting models"),

    # ---- fit ---------------------------------------------------------------
    "fit": (
        "fit", "fit/fit_data.py", {}, ("SIM_CONFIG_PATH",),
        "fit one model to all subjects"),
    "fit_forgetting": (
        "fit", "fit/fit_data_forgetting.py", {}, ("SIM_CONFIG_PATH",),
        "the same for a forgetting model, which has no ensemble stage"),
    "fit_forgetting_recovery": (
        "fit", "fit/fit_data_forgetting_recovery.py", {}, ("SIM_CONFIG_PATH",),
        "the recovery pass for a forgetting model, added on top of an existing "
        "fit without rewriting it"),
    "count_violations": (
        "fit", "fit/count_nesting_violations.py", {}, (),
        "nested pairs where the larger model fits worse, which cannot happen "
        "at a true optimum"),
    "build_seeds": (
        "fit", "fit/build_de_seeds.py", {}, (),
        "seeds for the second pass, from the nested model's fit"),
    "merge_runs": (
        "fit", "fit/merge_de_runs.py", {}, (),
        "keep each subject's better fit across the seeded and unseeded runs"),

    # ---- ensembles ---------------------------------------------------------
    "ensembles": (
        "ensembles", "ensembles/regenerate_ensembles.py", {}, (),
        "300 simulated datasets per model from the fitted parameters"),
    "ensembles_forgetting": (
        "ensembles", "ensembles/build_ensemble_forgetting.py", {}, ("SIM_CONFIG_PATH",),
        "the same for a forgetting model"),

    # ---- comparison --------------------------------------------------------
    "model_comparison": (
        "comparison", "comparison/model_comparison_viper.py", {}, (),
        "per-subject-summed BIC across every fitted model"),
    "bic_commit": (
        "comparison", "comparison/compute_bic_commit.py", {}, (),
        "the same for the commit likelihood family"),
    "winners": (
        "comparison", None, {}, (),
        "notebooks/model_comparison.ipynb writes BIC/best_models.json"),
    "per_subject_selection": (
        "comparison", "comparison/per_subject_model_selection.py", {}, (),
        "does the best model structure vary across people"),
    "per_subject_selection_commit": (
        "comparison", "comparison/per_subject_model_selection_commit.py", {}, (),
        "the same under the commit likelihood"),

    # ---- recovery ----------------------------------------------------------
    "parameter_recovery": (
        "recovery", "recovery/recovery_post_analysis.py", {}, ("SIM_CONFIG_PATH",),
        "simulate from each subject's own fit, refit blind, compare"),
    "model_recovery": (
        "recovery", "recovery/model_recovery.py", {}, ("GEN_TASK", "SIM_CONFIG_PATH"),
        "simulate from one model, fit every candidate, see which wins"),

    # ---- glm ---------------------------------------------------------------
    "glm_human": (
        "glm", "glm/glm_magda.py", {}, (),
        "human GLM coefficients, to glm_betas_human.csv"),
    "glmm_human": (
        "glm", "glm/compute_human_glmm.py", {}, (),
        "human GLMM estimates"),
    "glm_simulated": (
        "glm", "glm/glm_multiprocessed_simulate.py", {},
        ("SIM_CONFIG_SHORT_PATH", "SIM_CONFIG_LONG_PATH"),
        "model simulated coefficients, pairing a short and a long horizon model"),
    "glm_combined": (
        "glm", "glm/glm_multiprocessed_combined.py", {}, ("SIM_CONFIG_PATH",),
        "the same for a combined horizon config"),

    # ---- optimality --------------------------------------------------------
    "informed_prior": (
        "optimality", "optimality/informed_prior_optimum.py", {}, (),
        "build the informed benchmark and report how far it departs from the "
        "uniform prior version"),
    "optimality_cost": (
        "optimality", "optimality/export_optimality_cost.py",
        {"INFORMED_PRIOR": "1", "OPTIMAL_ONLY": "1", "OUT_SUFFIX": "_informed"},
        (),
        "score every policy in the task's own points"),
    "fitted_parameters_csv": (
        "optimality", "optimality/export_fitted_parameters_csv.py", {}, (),
        "write the fitted parameters out as csv, which is what results/ holds"),

    # ---- sensitivity -------------------------------------------------------
    # The sweeps are the expensive part and write .npz files; the table and
    # figure that the manuscript carries are made by make_figures afterwards.
    "sensitivity_sweep": (
        "sensitivity", "sensitivity/sensitivity_value_sweep.py", {}, (),
        "hold every subject's fit and sweep each fixed temporal regulation "
        "value, one horizon at a time via SWEEP_HORIZON"),
    "sensitivity_sweep_combined": (
        "sensitivity", "sensitivity/sensitivity_combined_sweep.py", {}, (),
        "the same for the combined horizon winner"),
}

# Stages that read what another stage writes. Everything else in a group is
# ordered only by convention, so it may overlap.
NEEDS = {
    "add_reward": ("convert_mat",),
    "build_evidence": ("add_reward",),
    "gamma_grids": ("build_evidence",),
    "build_seeds": ("count_violations",),
    "merge_runs": ("build_seeds",),
    "winners": ("model_comparison", "bic_commit"),
    "per_subject_selection": ("winners",),
    "per_subject_selection_commit": ("winners",),
    "optimality_cost": ("informed_prior",),
}


def missing_env(stage):
    _group, _script, _extra, requires, _desc = STAGES[stage]
    return [name for name in requires if not os.environ.get(name)]


def make_runner(dry_run):
    def runner(name):
        group, script, extra, _requires, description = STAGES[name]
        header = f"\n=== {name}  ({group})\n    {description}"
        if script is None:
            print(f"{header}\n    not a script: run the notebook, see PIPELINE.txt "
                  "section 5.2", flush=True)
            return name, True, None
        absent = missing_env(name)
        if absent:
            print(f"{header}\n    skipped, needs {' and '.join(absent)} to say "
                  "which model; see PIPELINE.txt", flush=True)
            return name, True, None
        if dry_run:
            shown = " ".join(f"{k}={v}" for k, v in extra.items())
            print(f"{header}\n    would run: {shown} python3 scripts/{script}".replace(
                "    would run:  ", "    would run: "), flush=True)
            return name, True, None
        return run_script(name, script, ROOT, extra, header)
    return runner


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=[],
                        help="any of: " + ", ".join(GROUPS) + ", all")
    parser.add_argument("--only", metavar="STAGE", help="run a single stage by name")
    parser.add_argument("--list", action="store_true",
                        help="show every stage and exit")
    parser.add_argument("--jobs", type=int, default=None, metavar="N",
                        help=f"independent stages to run at once (default "
                             f"{default_jobs()}, two thirds of the cores "
                             f"available to this process)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run, without running it")
    args = parser.parse_args()

    if args.list:
        for group in GROUPS:
            print(f"\n{group}")
            for name, (grp, script, _e, requires, description) in STAGES.items():
                if grp != group:
                    continue
                note = f"   [needs {', '.join(requires)}]" if requires else ""
                print(f"  {name:30s} {description}")
                print(f"  {'':30s}   {script or 'notebooks/model_comparison.ipynb'}{note}")
        return 0

    if args.only:
        if args.only not in STAGES:
            parser.error(f"unknown stage {args.only!r}; see --list")
        wanted = [args.only]
    else:
        groups = args.groups or ["all"]
        if "all" in groups:
            groups = list(GROUPS)
        for group in groups:
            if group not in GROUPS:
                parser.error(f"unknown group {group!r}; see --list")
        wanted = [n for n, (grp, *_rest) in STAGES.items() if grp in groups]

    jobs = args.jobs if args.jobs and args.jobs > 0 else default_jobs()
    jobs = min(jobs, len(wanted))
    print(f"{len(wanted)} stage(s), up to {jobs} at a time, in dependency order")

    causes = run_waves(wanted, jobs, NEEDS, make_runner(args.dry_run))

    failed = [n for n in wanted if causes.get(n) is not None]
    print(f"\n{len(wanted) - len(failed)} of {len(wanted)} stages succeeded")
    for name in failed:
        upstream = [d for d in NEEDS.get(name, ()) if d in failed]
        note = f"  (after {', '.join(upstream)} failed)" if upstream else ""
        print(f"  {name}: {causes[name]}{note}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
