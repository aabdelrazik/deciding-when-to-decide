#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=export
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/export_%j.out

# Runs the analysis/export scripts in scripts/ on a compute node, packing
# several onto the same allocation.
#
# These scripts solve a POMDP per subject per parameter vector, which is tens of
# minutes of work, so they do not belong on the shared login node. They are also
# single-threaded (the container pins OMP/OPENBLAS/MKL/NUMEXPR to one thread),
# so one script per node would leave 127 cores idle; running them concurrently
# costs no extra wall time. Taking the whole node also avoids the site's
# requirement that shared jobs declare a memory limit.
#
#   sbatch --export=ALL,SCRIPTS="a.py b.py",N_REPEATS=50,N_SEEDS=50 \
#          slurm/run_export_analysis.sh
#
# Any environment variable the scripts read (N_REPEATS, N_SEEDS, PILOT,
# SIM_ALGORITHM, POMDP_SEEDS_COMMIT, ...) passes straight through.

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

SCRIPTS="${SCRIPTS:?set SCRIPTS=\"one.py two.py\"}"
ALGO="${SIM_ALGORITHM:-de}"

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "host=$(hostname) cores=${SLURM_CPUS_PER_TASK} start=$(date)"
echo "scripts: ${SCRIPTS}"

pids=()
for SCRIPT in $SCRIPTS; do
    echo "  -> ${SCRIPT}"
    apptainer exec \
        --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
        --pwd ${SLURM_SUBMIT_DIR}/scripts \
        --env "SIM_ALGORITHM=${ALGO}" \
        --env "N_REPEATS=${N_REPEATS:-50}" \
        --env "N_SEEDS=${N_SEEDS:-50}" \
        --env "PILOT=${PILOT:-0}" \
        --env "N_JOBS=${N_JOBS:-1}" \
        --env "OUT_SUFFIX=${OUT_SUFFIX:-}" \
        --env "PYTHONUNBUFFERED=1" \
        --env "OMP_NUM_THREADS=1" \
        --env "OPENBLAS_NUM_THREADS=1" \
        --env "MKL_NUM_THREADS=1" \
        --env "NUMEXPR_NUM_THREADS=1" \
        pomdp_image.sif python3 "$SCRIPT" \
        > "${SLURM_SUBMIT_DIR}/logs/export_${SLURM_JOB_ID}_${SCRIPT%.py}.log" 2>&1 &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "end=$(date)  failed=${fail}/${#pids[@]}"

status=$?
echo "exit=${status}"
exit ${status}
