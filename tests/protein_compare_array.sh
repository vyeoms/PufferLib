#!/bin/bash
# Parallel protein_compare: a build job, one array task per sweep, then an evals + summary job.
# Run on the login node from the repo root:
#   tests/protein_compare_array.sh ENV [protein_compare.py args...]
#   SWEEP_SBATCH='--time=48:00:00 --cpus-per-task=32 --mem=64G' \
#       tests/protein_compare_array.sh nethack --sweeps 3 --runs 20 --cap 7200 --eval-seeds 1 --eval-max 4
# SWEEP_SBATCH: extra sbatch options for the sweep and eval jobs (EVAL_SBATCH overrides for evals).
# A sweep killed mid-run restarts from scratch on resubmit, so give array tasks enough --time.
set -e
N=$(python3 tests/protein_compare.py "$@" --num-tasks)
B=$(sbatch --parsable --time=02:00:00 tests/protein_compare.sbatch "$@" --build-only)
A=$(sbatch --parsable $SWEEP_SBATCH --dependency=afterok:$B --array=0-$((N - 1)) \
    tests/protein_compare.sbatch "$@" --no-build)
E=$(sbatch --parsable ${EVAL_SBATCH:-$SWEEP_SBATCH} --dependency=afterany:$A \
    tests/protein_compare.sbatch "$@" --no-build)
echo "build $B, sweeps $A (array 0-$((N - 1))), evals + summary $E"
