#!/bin/bash
# Full SSL pre-training on one 8-GPU H100 node, streaming 3DEP squares inside the DataLoader workers
# (data_mode="pool", eagle_als.stream_pool). Resubmit the same command to continue a run: it resumes
# from the latest checkpoint in $EAGLE_SCRATCH/runs/<run_name> with a fresh square sequence.
#   sbatch scripts/litePT/slurm/pretrain.sbatch                          # run ssl_s_pool_v1
#   RUN_NAME=ssl_s_test BATCH=16 sbatch -p GPU-shared --gpus=h100-80:1 --cpus-per-task=12 -t 1:00:00 \
#       scripts/litePT/slurm/pretrain.sbatch total_steps=200 log_every=10   # 1-GPU smoke test
# Extra arguments are config overrides (key=value). Per-GPU batch is 8 (16 runs out of memory), so
# grad_accum = 128 / (8 x GPUs). The run stops cleanly with a checkpoint 30 min before the time limit.
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
source scripts/litePT/stage_local.sh sites
trap 'rm -rf /dev/shm/eagle_pool_*' EXIT

NGPU=$(nvidia-smi -L | wc -l)
PER_GPU=8
BATCH=${BATCH:-128}  # global batch (ForPT); override for smoke tests
ACCUM=$(( BATCH / (NGPU * PER_GPU) ))
if [ "$ACCUM" -lt 1 ]; then ACCUM=1; fi
LIMIT_H=$(squeue -h -j "${SLURM_JOB_ID}" -o %l | awk -F'[-:]' '{ if (NF == 4) print $1 * 24 + $2 + $3 / 60; else if (NF == 3) print $1 + $2 / 60; else print $1 / 60 }')
MAX_H=$(awk -v h="$LIMIT_H" 'BEGIN { print (h > 1 ? h - 0.5 : h * 0.8) }')
RUN=${RUN_NAME:-ssl_s_pool_v1}
nvidia-smi -L
echo "[pretrain] run $RUN: $NGPU GPUs, batch $BATCH = $NGPU x $PER_GPU x accum $ACCUM, max_hours $MAX_H"

python -m torch.distributed.run --standalone --nproc-per-node "$NGPU" scripts/litePT/train_ssl.py \
  --config scripts/litePT/configs/ssl_litept_s.py \
  --opts data_mode=pool run_name="$RUN" batch_size=$BATCH grad_accum=$ACCUM max_hours=$MAX_H \
         "cache_dirs=['${EAGLE_LOCAL_CACHE:-$EAGLE_SCRATCH/cache}/sites']" "$@"
python scripts/litePT/plot_ssl_log.py "$EAGLE_SCRATCH/runs/$RUN" || true
