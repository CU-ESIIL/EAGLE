#!/bin/bash
# ForPT embeddings of the cached eval cookies (04_embed_forpt.py) on one NCSA Delta A100. Arguments go to the script;
# it is resumable (cookies already in a shard are skipped), so a time-out just needs a resubmit.
#   sbatch scripts/baseline_eval/embed_forpt_delta.sbatch --limit 1500 --shard-size 100           # speed test
#   sbatch -p gpuA100x4 -t 12:00:00 scripts/baseline_eval/embed_forpt_delta.sbatch               # full run
# GPU utilisation is sampled every 10 s into the job log (lines starting with "#gpu") to show whether the GPU or the
# CPU-side preprocessing limits the rate. Env: envs/litept-gpu (see its pixi.toml). AGENTS.md: ask before > 30 GPU-h.
#SBATCH --job-name=forpt-embed
#SBATCH --account=bjud-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/work/hdd/bjud/sammlapp/benchmarking/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
W=/work/hdd/bjud/sammlapp/benchmarking
echo "[forpt_delta] $(date) $(hostname) job $SLURM_JOB_ID: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 10 | sed -u 's/^/#gpu /' &
SMI=$!
ulimit -c 0
~/.pixi/bin/pixi run --manifest-path envs/litept-gpu/pixi.toml env PYTHONPATH=src OMP_NUM_THREADS=4 \
    python scripts/baseline_eval/04_embed_forpt.py --cache "$W/als_cache" --out "$W/embeddings/forpt" \
    --weights "$W/forpt_weights/epoch_30.pth" --io-workers "$((SLURM_CPUS_PER_TASK - 2))" "$@"
kill $SMI 2>/dev/null || true
echo "[forpt_delta] $(date) done"
