#!/bin/bash
# Streaming plan A (fetch in DataLoader workers) vs plan B (producer + shared buffer) on one CPU node.
# Each run measures 8 minutes after the first batch; both designs use 32 fetch processes.
# Unconstrained runs give the CPU-limited maximum; --rate 40 imitates GPUs consuming 40 samples/s
# (64 cores' share of an H100 node at ~64 samples/s per 104 cores), leaving spare CPU to fetching.
#   sbatch scripts/litePT/slurm/stream_compare.sbatch          # all five runs
#   sbatch scripts/litePT/slurm/stream_compare.sbatch capped   # only the 40 samples/s runs
# A cold Python environment on /ocean can take minutes to import, so the job warms it up first.
# Results: $EAGLE_SCRATCH/benchmarks/stream_compare_<jobid>/results.jsonl
#SBATCH --job-name=eagle-stream-compare
#SBATCH --account=bio260075p
#SBATCH --partition=RM-shared
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:45:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh
OUT=$EAGLE_SCRATCH/benchmarks/stream_compare_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
nproc; free -g; df -h /dev/shm "${LOCAL:-/tmp}"
time python -c "import torch, geopandas, pdal, litept.model; from eagle_als.stream import SquareSampler; SquareSampler()"
run() { python scripts/litePT/profile_stream_compare.py --minutes 8 --out "$OUT" "$@" 2>&1 | grep -v Warning; }
if [ "${1:-all}" != capped ]; then
  run --design A --workers 32 --seed 1
  run --design B --workers 32 --producer-workers 32 --seed 2
fi
run --design A --workers 32 --rate 40 --seed 3
run --design B --workers 12 --producer-workers 32 --rate 40 --seed 4
run --design A --workers 32 --rate 40 --uses 12 --seed 5 --tag A_w32_r40_uses12
rm -rf /dev/shm/eagle_cmp_* "${LOCAL:-/tmp}"/eagle_cmp_buffer_*
cat "$OUT/results.jsonl"
