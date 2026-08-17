#!/bin/bash -l
#SBATCH --job-name=informed_prior
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/informed_prior_%j.out

# Rebuilds the cost-of-departure benchmark with an ideal observer that knows
# the task's true generative prior over q, and compares it against the
# published uniform-prior benchmark on the same card sequences.

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_ALGORITHM=de" \
    --env "Q_GRID=400" \
    --env "OMP_NUM_THREADS=1" \
    --env "OPENBLAS_NUM_THREADS=1" \
    --env "MKL_NUM_THREADS=1" \
    --env "NUMEXPR_NUM_THREADS=1" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/optimality/informed_prior_optimum.py

status=$?
echo "exit=${status}"
exit ${status}
