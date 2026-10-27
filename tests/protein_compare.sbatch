#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victor.yeomsong@aalto.fi
#SBATCH --job-name=protein_compare
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=protein_compare_%j.out
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h100-80g,gpu-a100-80g

# Paired Protein comparison (tests/protein_compare.py): Adam vs Fisher + cost-dim linear term.
# Usage: sbatch tests/protein_compare.sbatch ENV [extra protein_compare.py args...]
#   sbatch tests/protein_compare.sbatch breakout --runs 80 --cap 30
#   sbatch tests/protein_compare.sbatch pacman --runs 40 --cap 300 \
#       --extra='--sweep.train.ent_coef.max=0.5'
# Resumable: resubmitting the same command skips finished sweeps and evals.
# For one job per sweep in parallel, use tests/protein_compare_array.sh instead.
ENV=${1:-breakout}
shift
ARGS=${*@Q}
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then  # array task: one sweep (tests/protein_compare_array.sh)
    ARGS+=" --task $SLURM_ARRAY_TASK_ID"
fi

srun -A ellis_users --cpus-per-task=$SLURM_CPUS_PER_TASK apptainer exec --nv \
    --bind /scratch/work/yeomv1/projects/puffer/my_fork/PufferLib:/puffertank/pufferlib \
    /scratch/work/yeomv1/venvs/puffer_img/puffertank/puffertank_sandbox/ bash -c "
    set -e
    export CUDA_HOME=/usr/local/cuda
    export OMP_NUM_THREADS=\$(nproc)
    cd /puffertank/pufferlib
    echo \"CPUs visible in container: \$(nproc)\"
    python3 tests/protein_compare.py $ENV $ARGS
"
