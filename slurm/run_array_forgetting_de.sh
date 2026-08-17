#!/bin/bash -l
#SBATCH --job-name=forget_de
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/forget_de_%A_%a.out
#SBATCH --exclusive=user

# Forgetting models with DE against the corrected gamma grids. The previous
# grids were built by discounting each gamma on top of the previous one's
# output, so only the first grid point held valid data and gamma was pinned
# there; validation on SB-XTGRPhC--- gave a genuine spread over all 16 points
# and +1103 BIC at identical k.
#
# Run unseeded, matching that validation: the only nested neighbours available
# hold gamma at 1.0, so seeding would start every subject at "no forgetting"
# and prejudge the distribution.
#
# Ordered short, then long, then combined.
mapfile -t CONFIG_FILES < "${SLURM_SUBMIT_DIR}/BIC/_de_run_configs_forgetting.txt"
CONFIG_NAME="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"
echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${CONFIG_NAME} (forgetting, DE)"

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${CONFIG_NAME}" \
    --env "SIM_ALGORITHM=de" \
    --env "SIM_N_JOBS=64" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/fit/fit_data_forgetting.py

status=$?
echo "exit=${status}"
exit ${status}
