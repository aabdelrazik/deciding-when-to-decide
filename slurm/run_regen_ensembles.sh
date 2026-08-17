#!/bin/bash -l
#SBATCH --job-name=regen_ens
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/regen_ens_%A_%a.out
#SBATCH --exclusive=user

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

# Re-simulate ensembles for the models the GLM/GLMM analysis draws on, using the
# merged parameters. Only the winners are needed: the paper simulates from each
# subject's own best-fitting short and long model, not from the whole candidate set.
TASKS=(SBEXT-RPh---- LBE-T-RPhCL--)
T="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "regenerating ensembles for $T"
srun apptainer exec \
    --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
    --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/simulation_params_${T}.py" \
    --env "SIM_ALGORITHM=de" \
    --env "SIM_N_JOBS=64" \
    --pwd "${SLURM_SUBMIT_DIR}" \
    pomdp_image.sif python3 scripts/ensembles/regenerate_ensembles.py
