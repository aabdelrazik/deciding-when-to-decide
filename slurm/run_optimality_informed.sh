#!/bin/bash -l
#SBATCH --job-name=optimality_informed
#SBATCH --nodes=1
#SBATCH --time=0-06:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/optimality_informed_%j.out

# Cost-of-departure benchmark recomputed with the ideal observer given the
# task's true prior over the generative probability. Writes
# optimality_cost_informed.csv alongside the published file rather than over it.

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_ALGORITHM=de" \
    --env "INFORMED_PRIOR=1" \
    --env "OPTIMAL_ONLY=1" \
    --env "OUT_SUFFIX=_informed" \
    --env "Q_GRID=400" \
    --env "N_JOBS=${N_JOBS:-64}" \
    --env "OMP_NUM_THREADS=1" \
    --env "OPENBLAS_NUM_THREADS=1" \
    --env "MKL_NUM_THREADS=1" \
    --env "NUMEXPR_NUM_THREADS=1" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/optimality/export_optimality_cost.py

status=$?
echo "exit=${status}"
exit ${status}
