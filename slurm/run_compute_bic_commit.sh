#!/bin/bash -l
#SBATCH --job-name=biccommit
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/biccommit_%j.out

# Re-evaluates the best models' already-fitted parameters under
# log_likelihood_commit, so the notebook can build a GLM-vs-POMDP BIC on the
# same target the GLM predicts. No refitting.

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true
echo "host=$(hostname) start=$(date)"
srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_ALGORITHM=de" --env "SIM_N_JOBS=105" \
    --env "OMP_NUM_THREADS=1" --env "OPENBLAS_NUM_THREADS=1" --env "MKL_NUM_THREADS=1" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/comparison/compute_bic_commit.py
echo "end=$(date)"

status=$?
echo "exit=${status}"
exit ${status}
