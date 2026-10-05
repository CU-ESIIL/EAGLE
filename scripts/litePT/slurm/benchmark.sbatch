#!/bin/bash
# GPU memory / throughput benchmark (and smoke test) for one GPU.
#   sbatch scripts/litePT/slurm/benchmark.sbatch                       # H100 (default)
#   sbatch --gpus=v100-32:1 --cpus-per-task=5 scripts/litePT/slurm/benchmark.sbatch
# Extra args are passed to benchmark_memory.py (e.g. --batch-sizes 2 4 --opts ...).
#SBATCH --job-name=eagle-bench
#SBATCH --account=bio260075p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
nvidia-smi
python scripts/litePT/benchmark_memory.py --config scripts/litePT/configs/ssl_litept_s.py \
  --workers "${SLURM_CPUS_PER_TASK:-5}" --out "$EAGLE_SCRATCH/benchmarks/ssl_s_${GPU_NAME}_${SLURM_JOB_ID}" "$@"
