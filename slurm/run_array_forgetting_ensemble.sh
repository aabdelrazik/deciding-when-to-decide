#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_forgetting_ensemble
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/ocd_pomdp_forgetting_ensemble_%A_%a.out
#SBATCH --exclusive=user


# Runs scripts/ensembles/build_ensemble_forgetting.py, one Slurm array task per
# forgetting config (POMDP_TYPE="forgetting") in data/simulation_configs/.
# Builds the 300-run ensemble simulation cache (raw_simulations/,
# ensemble_metrics_summary.csv, ensemble_distribution_data.pkl) from each
# config's already-fitted results.pkl -- no refitting, no parameter
# recovery, just the ensemble simulations GLM/GLMM scripts need downstream.
#
# Submit with (run from the repo root, after `mkdir -p logs`):
#   FORGETTING_FILES=$(grep -l 'POMDP_TYPE="forgetting"' data/simulation_configs/simulation_params_*.py | grep -v '_commit\.py')
#   N=$(echo "$FORGETTING_FILES" | wc -l)
#   sbatch --array=0-$((N-1))%5 run_array_forgetting_ensemble.sh

mapfile -t CONFIG_FILES < <(grep -l 'POMDP_TYPE="forgetting"' "${SLURM_SUBMIT_DIR}"/data/simulation_configs/simulation_params_*.py | grep -v '_commit\.py')
CONFIG_FILE="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"
CONFIG_NAME=$(basename "$CONFIG_FILE")

echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${CONFIG_NAME}"

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${CONFIG_NAME}" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/ensembles/build_ensemble_forgetting.py

status=$?
echo "exit=${status}"
exit ${status}
