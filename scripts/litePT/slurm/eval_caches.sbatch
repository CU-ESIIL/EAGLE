#!/bin/bash
# Fetch the cookie caches of all evaluation tasks in the pre-training config and pack them for staging
# (scripts/litePT/build_eval_caches.py). Re-run after an evaluation table changes.
#   sbatch scripts/litePT/slurm/eval_caches.sbatch                    # all tasks
#   sbatch scripts/litePT/slurm/eval_caches.sbatch --tasks ofo bbs    # some tasks
#   sbatch scripts/litePT/slurm/eval_caches.sbatch --retry-errors --workers 16   # retry failed fetches (S3 read
#       errors are often transient at high concurrency)
#SBATCH --job-name=eagle-evalcache
#SBATCH --account=bio260075p
#SBATCH --partition=RM-shared
#SBATCH --ntasks-per-node=32
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh
python scripts/litePT/build_eval_caches.py --workers 48 "$@"
