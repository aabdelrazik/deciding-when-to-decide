#!/bin/bash -l
#SBATCH --job-name=verify
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/verify_%j.out

# Everything that checks the release is sound, on a compute node so the timings
# below describe the hardware the results were produced on rather than whatever
# else happens to be running on a login node.
#
#   1  every script imports
#   2  the test suite
#   3  the primer notebook, top to bottom
#   4  one single subject fit, as a timing reference
#
# Submit with (from the repository root):
#   sbatch slurm/run_verification.sh
#
# Set POMDP_DATA_ROOT if the dataset and the fits live outside the checkout;
# each tree is mounted read only over its counterpart here. Without it the
# stages that need data skip rather than fail.
#
#   POMDP_DATA_ROOT=/scratch/pomdp sbatch slurm/run_verification.sh

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

R="${SLURM_SUBMIT_DIR}"
CFG="${R}/data/simulation_configs/simulation_params_SB-XT-RPh----.py"

BINDS="--bind ${R}:${R}"
if [ -z "${POMDP_DATA_ROOT}" ]; then
    # Not an error, but say so: this site does not export the submitting
    # environment by default, so an unset variable here usually means the job
    # was submitted without --export rather than deliberately.
    echo "POMDP_DATA_ROOT is unset, using whatever data the checkout holds"
else
    echo "POMDP_DATA_ROOT=${POMDP_DATA_ROOT}"
    for tree in data/TrHu_NHB_light data/POMDP data/POMDP_commit; do
        if [ -d "${POMDP_DATA_ROOT}/${tree}" ]; then
            mkdir -p "${R}/${tree}"
            BINDS="${BINDS} --bind ${POMDP_DATA_ROOT}/${tree}:${R}/${tree}:ro"
            echo "  bound ${tree}"
        else
            echo "  ${POMDP_DATA_ROOT}/${tree} not found, using the checkout's own"
        fi
    done
fi

# SIM_CONFIG_PATH matters everywhere: the default config is a forgetting model
# that needs the gamma grids, and a model whose likelihood raises returns the
# 1e10 sentinel rather than failing, which reads as a fit that ran and did
# nothing.
RUN="apptainer exec ${BINDS} --pwd ${R} --env PYTHONNOUSERSITE=1 \
     --env SIM_ALGORITHM=de --env SIM_CONFIG_PATH=${CFG} ${R}/pomdp_image.sif"

echo "host=$(hostname)  cores=${SLURM_CPUS_PER_TASK}  start=$(date)"
echo "cpu: $(lscpu | awk -F: '/Model name/{print $2; exit}' | xargs)"
echo

echo "===================================================================="
echo "1  import check"
echo "===================================================================="
t0=$SECONDS
${RUN} python3 - <<'PY'
import importlib.util, os, sys, io, contextlib
sys.path.insert(0, os.getcwd())
os.environ.setdefault("SIM_CONFIG_PATH", os.path.join(
    os.getcwd(), "data/simulation_configs/simulation_params_SB-XT-RPh----.py"))
bad = []
names = []
for dirpath, dirs, files in os.walk("scripts"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    names += [os.path.join(dirpath, f) for f in files if f.endswith(".py")]
names.sort()
for path in names:
    # a script's own directory is on sys.path when it runs, so sibling imports
    # resolve; reproduce that here rather than reporting them as failures
    here = os.path.abspath(os.path.dirname(path))
    added = here not in sys.path
    if added:
        sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location(
        "m_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(mod)
    except SystemExit:
        pass
    except Exception as e:
        bad.append((os.path.relpath(path, "scripts"), type(e).__name__, str(e)[:66]))
    finally:
        if added:
            sys.path.remove(here)
print(f"{len(names) - len(bad)}/{len(names)} scripts import cleanly")
for fn, kind, msg in bad:
    print(f"   {fn:46s} {kind}: {msg}")
PY
echo "  elapsed ${SECONDS}s"

echo
echo "===================================================================="
echo "2  test suite"
echo "===================================================================="
t0=$SECONDS
${RUN} python3 -m unittest discover -s tests
echo "  elapsed $((SECONDS - t0))s"

echo
echo "===================================================================="
echo "3  primer notebook"
echo "===================================================================="
t0=$SECONDS
${RUN} python3 - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read("notebooks/example_usage.ipynb", as_version=4)
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": "notebooks"}},
               allow_errors=True).execute()
code = [c for c in nb.cells if c.cell_type == "code"]
fails = [(n, o["ename"], (o.get("evalue") or "")[:60])
         for n, c in enumerate(code, start=1)
         for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"{len(fails)} of {len(code)} code cells failed")
for n, name, detail in fails:
    print(f"   code cell {n}: {name}: {detail}")
PY
echo "  elapsed $((SECONDS - t0))s"

echo
echo "===================================================================="
echo "4  one subject fit, timing reference"
echo "===================================================================="
t0=$SECONDS
${RUN} python3 - <<'PY'
import os, sys, time
import pandas as pd
sys.path.insert(0, os.getcwd())
from src.pomdp import POMDPFactory

path = "data/TrHu_NHB_light/data_MEG/all_subject_evidence_dicts_short.pkl"
if not os.path.exists(path):
    print("dataset not present, skipped")
else:
    RANGES = {"xi": (0, 1), "tau": (0, 100), "subjective_cost": (-300, 0),
              "patience": (0, 8), "belief_bias": (0.01, 5)}
    evidence = pd.read_pickle(path)
    subject = evidence.index[0]
    model = POMDPFactory("urgency")
    start = time.time()
    best, ll, *_ = model.fit_subject(evidence.loc[subject].to_dict(),
                                     RANGES, subject, "de")
    elapsed = time.time() - start
    if ll <= -1e9:
        print(f"FAILED: the cost function returned its error sentinel after "
              f"{elapsed:.1f}s, so nothing was actually fitted. Check "
              f"SIM_CONFIG_PATH and the job log for [Cost Error].")
        sys.exit(1)
    print(f"subject {subject}: {elapsed:.1f}s at the full DE budget, "
          f"logL {ll:.4f}")
    print("  " + ", ".join(f"{k}={v:.4f}" for k, v in best.items()))
PY
echo "  elapsed $((SECONDS - t0))s"

echo
echo "end=$(date)  total=${SECONDS}s"
