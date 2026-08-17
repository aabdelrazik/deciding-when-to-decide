#!/bin/bash -l

# --- Slurm properties ---
#SBATCH --job-name=ocd_pomdp_sensitivity_combined
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/sensitivity_combined_%j.out
#SBATCH --exclusive=user

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmed.abdelrazik@tuebingen.mpg.de

# Runs scripts/sensitivity/sensitivity_combined_sweep.py: value-sweep sensitivity check
# of the combined-horizon winning model's (CB--TGRPhCL--) fixed
# urgency_coefficient (phi_min) and urgency_slope (k), parallelized across
# subjects. Pure forward evaluation (value_iteration + log_likelihood at
# each candidate value, using each subject's already-fitted other
# parameters) -- no refitting/optimization.
#
# Submit with (run from the repo root, after `mkdir -p logs`):
#   sbatch slurm/run_sensitivity_combined_sweep.sh

ENV_ARGS=(
    --env "OMP_NUM_THREADS=1"
    --env "OPENBLAS_NUM_THREADS=1"
    --env "MKL_NUM_THREADS=1"
    --env "NUMEXPR_NUM_THREADS=1"
)

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    "${ENV_ARGS[@]}" \
    "${SLURM_SUBMIT_DIR}/pomdp_image.sif" python3 scripts/sensitivity/sensitivity_combined_sweep.py

status=$?
echo "exit=${status}"
exit ${status}
