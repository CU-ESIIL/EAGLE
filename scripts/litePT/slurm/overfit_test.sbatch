#!/bin/bash
# Sanity check of the SSL training workflow: train on a tiny fixed pool of cookies (1 GPU) and
# confirm the loss (and the KL part of it) decreases without collapse. ~20 min on an H100.
#   sbatch scripts/litePT/slurm/overfit_test.sbatch [extra --opts ...]
#SBATCH --job-name=eagle-overfit
#SBATCH --account=bio260075p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh
RUN=overfit_test_${SLURM_JOB_ID:-local}
python scripts/litePT/train_ssl.py --config scripts/litePT/configs/ssl_litept_s.py --opts \
  data_mode=cache "cache_dirs=['$EAGLE_SCRATCH/cache/sites']" max_samples=8 batch_size=4 num_workers=7 \
  total_steps=1500 steps_per_epoch=1500 log_every=10 ckpt_every=100000 run_name=$RUN "$@"
python scripts/litePT/plot_ssl_log.py "$EAGLE_SCRATCH/runs/$RUN"
