#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_glm_array
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/glm_array_%A_%a.out
#SBATCH --exclusive=user


# Pairs one fixed "best short" config (from BIC/best_models.json, written by
# model_comparison.ipynb) against every long-horizon config in
# data/simulation_configs/, running glm_multiprocessed_simulate.py once per
# pair as a Slurm array task -- each task gets its own node.
#
# IMPORTANT: never run glm_multiprocessed_simulate.py directly on the login
# node -- it's a heavy multiprocessing job (one process per CPU) and will load
# the shared login node for everyone. Always go through this (or any) sbatch
# script so it lands on a compute node instead.
#
# Submit with (run from the repo root, after `mkdir -p logs`):
#   LONG_FILES=(data/simulation_configs/simulation_params_L*.py)
#   N=${#LONG_FILES[@]}
#   sbatch --array=0-$((N-1))%5 run_glm_array.sh
# (The L* glob is long-horizon-only configs, by the TASK-name convention where
# the first letter is L; combined "C" configs aren't part of a short+long pair.)
# Raise/lower the %5 throttle same as run_array.sh -- see scontrol update
# ArrayTaskThrottle to change a running array's concurrency without resubmitting.

mapfile -t LONG_FILES < <(ls "${SLURM_SUBMIT_DIR}"/data/simulation_configs/simulation_params_L*.py | grep -v '_commit\.py')
LONG_CONFIG="${LONG_FILES[$SLURM_ARRAY_TASK_ID]}"
LONG_NAME=$(basename "$LONG_CONFIG")

SHORT_TASK=$(python3 -c "import json; print(json.load(open('${SLURM_SUBMIT_DIR}/BIC/best_models.json'))['short'])")
SHORT_NAME="simulation_params_${SHORT_TASK}.py"

echo "Array task ${SLURM_ARRAY_TASK_ID}: short=${SHORT_NAME} (fixed, best short model) long=${LONG_NAME}"

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

# --- Job execution ---
# SIM_CONFIG_SHORT_PATH / SIM_CONFIG_LONG_PATH tell glm_multiprocessed_simulate.py
# which pair to use; it reads each config's own RESULTS_PATH/raw_simulations/ and
# never constructs a POMDP itself, so each config's own POMDP_TYPE is irrelevant
# here -- no SIM_CONFIG_PATH/CONFIG mismatch risk like generate_meg_data has.
srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_CONFIG_SHORT_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${SHORT_NAME}" \
    --env "SIM_CONFIG_LONG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${LONG_NAME}" \
    --env "OMP_NUM_THREADS=1" \
    --env "OPENBLAS_NUM_THREADS=1" \
    --env "MKL_NUM_THREADS=1" \
    --env "NUMEXPR_NUM_THREADS=1" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/glm/glm_multiprocessed_simulate.py

status=$?
echo "exit=${status}"
exit ${status}
