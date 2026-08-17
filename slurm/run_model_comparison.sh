#!/bin/bash -l
#SBATCH --job-name=modcomp
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/modcomp_%j.out

# Regenerates the manuscript's BIC tables and best_models.json by executing the
# model-comparison notebook headlessly against the DE fits. NB selects which
# notebook (model_comparison.ipynb / model_comparison_commit.ipynb).

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true
echo "notebook=${NB}  host=$(hostname) start=$(date)"

apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_ALGORITHM=de" \
    --env "SIM_N_JOBS=105" \
    --env "OMP_NUM_THREADS=1" --env "OPENBLAS_NUM_THREADS=1" --env "MKL_NUM_THREADS=1" \
    pomdp_image.sif bash -lc "cd ${SLURM_SUBMIT_DIR}/notebooks && python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output ${NB} ${NB}"
echo "nbconvert exit=$?"
echo "end=$(date)"

status=$?
echo "exit=${status}"
exit ${status}
