#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_sensitivity_sweep
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/sensitivity_sweep_%x_%j.out

# Runs scripts/sensitivity/sensitivity_value_sweep.py for one horizon: value-sweep
# sensitivity check of the temporal-regulation parameters that the winning
# model fixes rather than fits, parallelized across subjects. Pure forward
# evaluation (value_iteration + log_likelihood at each candidate value using
# each subject's already-fitted other parameters) -- no refitting.
#
# The horizon, the model, its family and which parameters are fixed all come
# from BIC/best_models.json and the config, so this does not need editing when
# a winner changes.
#
# Submit one per horizon from the repo root:
#   SWEEP_HORIZON=short sbatch --job-name=sweep_short slurm/run_sensitivity_value_sweep.sh
#   SWEEP_HORIZON=long  sbatch --job-name=sweep_long  slurm/run_sensitivity_value_sweep.sh

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

: "${SWEEP_HORIZON:=short}"
: "${SIM_ALGORITHM:=de}"

ENV_ARGS=(
    --env "OMP_NUM_THREADS=1"
    --env "OPENBLAS_NUM_THREADS=1"
    --env "MKL_NUM_THREADS=1"
    --env "NUMEXPR_NUM_THREADS=1"
    --env "SWEEP_HORIZON=${SWEEP_HORIZON}"
    --env "SIM_ALGORITHM=${SIM_ALGORITHM}"
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
)

# invoke python3 directly: the container's login shell (bash -lc) sources a
# profile that exits non-zero here, killing the step before the script runs
srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    "${ENV_ARGS[@]}" \
    "${SLURM_SUBMIT_DIR}/pomdp_image.sif" python3 scripts/sensitivity/sensitivity_value_sweep.py

status=$?
echo "exit=${status}"
exit ${status}
