#!/bin/bash -l
#SBATCH --job-name=gamma_grids
#SBATCH --nodes=1
#SBATCH --time=0-03:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/gamma_grids_%A_%a.out

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

# Regenerate the per-gamma discounted evidence grids, one array task per
# horizon. The previous notebook loop discounted each gamma on top of the
# previous one's output, so only the first grid point was ever valid.
CFGS=(SB-XTGRPhC--- LB--TGRPhCL-- CB--TGRPhCL--)      # short, long, combined
CFG="${CFGS[$SLURM_ARRAY_TASK_ID]}"
echo "horizon config: $CFG"
srun apptainer exec \
  --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
  --env "SIM_CONFIG_PATH=${SLURM_SUBMIT_DIR}/data/simulation_configs/simulation_params_${CFG}.py" \
  pomdp_image.sif bash -lc "cd ${SLURM_SUBMIT_DIR}/notebooks && python ${SLURM_SUBMIT_DIR}/scripts/preprocess/generate_gamma_grids.py"
