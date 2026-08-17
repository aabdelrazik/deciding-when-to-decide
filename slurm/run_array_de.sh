#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_de
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/ocd_pomdp_de_%A_%a.out
#SBATCH --exclusive=user

# Refit the non-forgetting candidate models with differential evolution instead
# of the GA. The GA violated nesting in 51 of 212 checkable model pairs (a model
# whose search space contains another's fitted worse than it, which is
# impossible at an optimum); DE satisfies nesting for 20/20 subjects on the
# short, long and combined pathological pairs and reaches an identical optimum
# from independent seeds.
#
# SIM_ALGORITHM=de sends output to data/POMDP/<TASK>/de/... , leaving the
# existing ga/ fits (the ones behind the manuscript) untouched.
#
# --- Submit with (from the repo root, after `mkdir -p logs`): ---
#   N=$(wc -l < BIC/_de_run_configs.txt)
#   sbatch --array=0-$((N-1))%8 slurm/run_array_de.sh

# CONFIG_LIST names the file holding one config filename per line, so a
# subset can be run without editing the tracked list.
mapfile -t CONFIG_FILES < "${SLURM_SUBMIT_DIR}/${CONFIG_LIST:-BIC/_de_run_configs.txt}"
CONFIG_NAME="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${CONFIG_NAME} (differential evolution)"

# POMDP_DATA_ROOT points at whatever holds the dataset when it lives outside
# the checkout; the fits are written into the checkout either way.
BINDS="--bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}"
if [ -n "${POMDP_DATA_ROOT}" ] && [ -d "${POMDP_DATA_ROOT}/data/TrHu_NHB_light" ]; then
    mkdir -p "${SLURM_SUBMIT_DIR}/data/TrHu_NHB_light"
    BINDS="${BINDS} --bind ${POMDP_DATA_ROOT}/data/TrHu_NHB_light:${SLURM_SUBMIT_DIR}/data/TrHu_NHB_light:ro"
    echo "dataset bound from ${POMDP_DATA_ROOT}"
fi

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    ${BINDS} \
    --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/${CONFIG_NAME}" \
    --env "SIM_ALGORITHM=de" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/fit/fit_data.py

status=$?
echo "exit=${status}"
exit ${status}
