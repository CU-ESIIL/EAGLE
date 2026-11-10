#!/bin/bash
# Full SSL pre-training on one 8-GPU H100 node, streaming 3DEP squares inside the DataLoader workers
# (data_mode="pool", eagle_als.stream_pool). Resubmit the same command to continue a run: it resumes
# from the latest checkpoint in $EAGLE_SCRATCH/runs/<run_name> with a fresh square sequence.
#   sbatch scripts/litePT/slurm/pretrain.sbatch                          # run ssl_s_pool_v1
#   sbatch scripts/litePT/slurm/pretrain.sbatch run_name=ssl_s_pool_v2 total_steps=80000
#   sbatch -p GPU-shared --gpus=h100-80:2 --cpus-per-task=26 scripts/litePT/slurm/pretrain.sbatch \
#       run_name=ssl_s_2gpu                                              # 2 H100s, same batch 128
#   sbatch -p GPU-shared --gpus=h100-80:1 --cpus-per-task=13 -t 1:00:00 scripts/litePT/slurm/pretrain.sbatch \
#       run_name=ssl_s_test batch_size=16 total_steps=200 log_every=10   # 1-GPU smoke test
# Arguments are config overrides (key=value); run_name (default ssl_s_pool_v1) is also read here.
# Environment variables do not reach the job on Bridges-2, so pass settings as arguments.
# Any number of GPUs works: train_ssl.py keeps the global batch_size (128) and sets grad_accum
# (at most 8 samples per GPU per micro-batch) and DataLoader workers (from the CPU cores per GPU) to
# fit the job. On GPU-shared ask for 13 CPUs per GPU (an H100 node has 104 cores for 8 GPUs).
# The run stops cleanly with a checkpoint 30 min before the time limit.
#SBATCH --job-name=eagle-pretrain
#SBATCH --account=bio260075p
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gpus=h100-80:8
#SBATCH --time=48:00:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[pretrain] $(hostname), job ${SLURM_JOB_ID:-local}, LOCAL=${LOCAL:-unset}"
# environment and evaluation cookies onto node-local disk (/ocean can stall on many small reads)
source scripts/litePT/stage_local.sh sites nlcd
trap 'rm -rf /dev/shm/eagle_pool_*' EXIT

NGPU=$(nvidia-smi -L | wc -l)
RUN=ssl_s_pool_v1
for arg in "$@"; do
  case "$arg" in
    run_name=*) RUN="${arg#run_name=}" ;;
  esac
done
LIMIT_H=$(squeue -h -j "${SLURM_JOB_ID}" -o %l | awk -F'[-:]' '{ if (NF == 4) print $1 * 24 + $2 + $3 / 60; else if (NF == 3) print $1 + $2 / 60; else print $1 / 60 }')
MAX_H=$(awk -v h="$LIMIT_H" 'BEGIN { print (h > 1 ? h - 0.5 : h * 0.8) }')
nvidia-smi -L
echo "[pretrain] run $RUN: $NGPU GPUs, $(nproc) CPUs, max_hours $MAX_H"

python -m torch.distributed.run --standalone --nproc-per-node "$NGPU" scripts/litePT/train_ssl.py \
  --config scripts/litePT/configs/ssl_litept_s.py \
  --opts data_mode=pool run_name="$RUN" max_hours=$MAX_H \
         "cache_dirs=['${EAGLE_LOCAL_CACHE:-$EAGLE_SCRATCH/cache}/sites']" "$@"
python scripts/litePT/plot_ssl_log.py "$EAGLE_SCRATCH/runs/$RUN" || true
