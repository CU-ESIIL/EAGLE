#!/bin/bash
# Score a run's saved checkpoints (and a random-init baseline) on the config's evaluation tasks
# (scripts/litePT/eval_checkpoints.py); writes <run>/eval_checkpoints.csv and TensorBoard scalars.
#   sbatch scripts/litePT/slurm/eval_checkpoints.sbatch ssl_s_2gpu_2
#   sbatch scripts/litePT/slurm/eval_checkpoints.sbatch ssl_s_2gpu_2 --steps 11000
#SBATCH --job-name=eagle-evalckpt
#SBATCH --account=bio260075p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/stage_local.sh nlcd ofo snapshot_usa lfrdb bbs
RUN="$1"; shift
python scripts/litePT/eval_checkpoints.py "$EAGLE_SCRATCH/runs/$RUN" --init --tb "$@"
