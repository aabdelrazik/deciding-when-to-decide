#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_glm_combined_array
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/glm_combined_array_%A_%a.out
#SBATCH --exclusive=user


# Runs glm_multiprocessed_combined.py once per combined-horizon config (TASK
# starting with "C") in data/simulation_configs/, one Slurm array task per
# config, each on its own node. Unlike run_glm_array.sh, there's no separate
# short config to pair -- each combined config already fit short+long together,
# so its own raw_simulations/ already holds the complete dataset per run.
#
# IMPORTANT: never run glm_multiprocessed_combined.py directly on the login
# node -- always submit through this script so it lands on a compute node.
#
# Submit with (run from the repo root, after `mkdir -p logs`):
#   COMBINED_FILES=(data/simulation_configs/simulation_params_C*.py)
#   N=${#COMBINED_FILES[@]}
#   sbatch --array=0-$((N-1))%5 run_glm_combined_array.sh
# (Or just use ./submit_glm_combined_array.sh.)

mapfile -t COMBINED_FILES < <(ls "${SLURM_SUBMIT_DIR}"/data/simulation_configs/simulation_params_C*.py | grep -v '_commit\.py')
COMBINED_CONFIG="${COMBINED_FILES[$SLURM_ARRAY_TASK_ID]}"
COMBINED_NAME=$(basename "$COMBINED_CONFIG")

echo "Array task ${SLURM_ARRAY_TASK_ID}: combined=${COMBINED_NAME}"

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

# --- Job execution ---
srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${COMBINED_NAME}" \
    --env "OMP_NUM_THREADS=1" \
    --env "OPENBLAS_NUM_THREADS=1" \
    --env "MKL_NUM_THREADS=1" \
    --env "NUMEXPR_NUM_THREADS=1" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/glm/glm_multiprocessed_combined.py

status=$?
echo "exit=${status}"
exit ${status}
