#!/bin/bash -l
#SBATCH --job-name=make_figures
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --partition general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/make_figures_%j.out

# Produce the manuscript figures for one or more sections. Some targets refit
# GLMMs, so this belongs on a compute node rather than a login node.
#
# Submit with (from the repository root):
#   SECTIONS="results" sbatch slurm/run_generate_figures.sh
#   SECTIONS="methods results appendix" sbatch slurm/run_generate_figures.sh
#
# The fitted ensembles run to a few hundred gigabytes, so they usually sit on a
# scratch filesystem rather than inside the checkout. Point POMDP_DATA_ROOT at
# whatever holds them and each tree is mounted over its counterpart here.
# Slurm at this site does not pass the submitting environment to the job, so
# name the variable in --export or it arrives empty:
#
#   sbatch --export=ALL,POMDP_DATA_ROOT=/scratch/pomdp,SECTIONS=results \
#          slurm/run_generate_figures.sh

# Site specific. Apptainer may already be on PATH, or be called
# singularity, or carry a different version; do not abort if the module
# system has none of them.
module load apptainer/1.3.6 2>/dev/null || module load apptainer 2>/dev/null || true

# Read only where nothing writes, so a figure job cannot damage the fits.
# data/POMDP_commit is writable because the GLMM comparison writes its table
# beside the commit likelihood fits it describes.
RO_TREES="data/TrHu_NHB_light data/POMDP data/POMDP_recovery_x4"
RW_TREES="data/POMDP_commit"

BINDS="--bind ${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}"
if [ -z "${POMDP_DATA_ROOT}" ]; then
    echo "POMDP_DATA_ROOT is unset, using whatever data the checkout holds"
else
    echo "POMDP_DATA_ROOT=${POMDP_DATA_ROOT}"
    for tree in ${RO_TREES} ${RW_TREES}; do
        case " ${RO_TREES} " in *" ${tree} "*) mode=":ro";; *) mode="";; esac
        if [ -d "${POMDP_DATA_ROOT}/${tree}" ]; then
            mkdir -p "${SLURM_SUBMIT_DIR}/${tree}"
            BINDS="${BINDS} --bind ${POMDP_DATA_ROOT}/${tree}:${SLURM_SUBMIT_DIR}/${tree}${mode}"
            echo "  bound ${tree}${mode:- (writable)}"
        else
            echo "  ${POMDP_DATA_ROOT}/${tree} not found, using the checkout's own"
        fi
    done
fi

srun apptainer exec \
    ${BINDS} \
    --pwd "${SLURM_SUBMIT_DIR}" \
    --env "PYTHONNOUSERSITE=1" \
    --env "SIM_ALGORITHM=de" \
    --env "OMP_NUM_THREADS=1" \
    --env "OPENBLAS_NUM_THREADS=1" \
    --env "MKL_NUM_THREADS=1" \
    pomdp_image.sif python3 scripts/generate_figures.py ${SECTIONS:-all}

status=$?
echo "exit=${status}"
exit ${status}
