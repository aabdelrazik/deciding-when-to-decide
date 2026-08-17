#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_dec
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/ocd_pomdp_dec_%A_%a.out
#SBATCH --exclusive=user

# Refit with differential evolution, each subject started from the best model
# nested inside this one (BIC/_de_seeds.pkl, built by scripts/fit/build_de_seeds.py).
# The unseeded run left 19 nesting violations; probing the worst of them showed
# the shortfall was where the search started, not the DE settings: seeding took
# that pair from -625.1 logL to +0.0 with no subject violating.
#
# Config list is ordered short, then long, then combined, so the cheap models
# finish first.
#
# --- Submit with (from the repo root): ---
#   N=$(wc -l < BIC/_de_run_configs_commit.txt)
#   sbatch --array=0-$((N-1))%15 slurm/run_array_de_seeded.sh

mapfile -t CONFIG_FILES < "${SLURM_SUBMIT_DIR}/BIC/_de_run_configs_commit.txt"
CONFIG_NAME="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${CONFIG_NAME} (seeded DE)"

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${CONFIG_NAME}" \
    --env "SIM_ALGORITHM=de" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/fit/fit_data.py

status=$?
echo "exit=${status}"
exit ${status}
