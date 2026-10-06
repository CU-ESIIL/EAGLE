#!/bin/bash
# Streaming benchmark on a CPU node: producer throughput and S3 error rate vs. worker count, then
# DataLoader throughput of StreamingChunkDataset on the resulting (static) buffer.
#   sbatch scripts/litePT/slurm/producer_sweep.sbatch [workers ...]   (default: 12 24 48)
# Results: $EAGLE_SCRATCH/benchmarks/producer_sweep_<jobid>/
#SBATCH --job-name=eagle-producer-sweep
#SBATCH --account=bio260075p
#SBATCH --partition=RM-shared
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:30:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh
if [ $# -gt 0 ]; then WORKERS=("$@"); else WORKERS=(12 24 48); fi
OUT=$EAGLE_SCRATCH/benchmarks/producer_sweep_${SLURM_JOB_ID:-local}
BUF=${LOCAL:-/tmp}/eagle_buffer_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
echo "LOCAL=${LOCAL:-unset} buffer root $BUF"; df -h "$(dirname "$BUF")"; nproc; free -g
vmstat -t 15 > "$OUT/vmstat.txt" &
VMSTAT=$!
for W in "${WORKERS[@]}"; do
  echo "=== producer, $W workers ==="
  python -m eagle_als.stream --buffer "$BUF/w$W" --state-dir "$OUT/w$W" --workers "$W" \
    --max-shards 100000 --seed 0 --max-hours 0.12
  du -sh "$BUF/w$W"
done
LAST=${WORKERS[-1]}
echo "=== dataloader on buffer w$LAST ==="
python scripts/litePT/profile_dataloader.py --config scripts/litePT/configs/ssl_litept_s.py \
  --opts data_mode=stream stream_dir="$BUF/w$LAST" stream_usage_dir="$OUT/usage" \
  --stages 60 --workers 1 4 8 16 --loader-samples 96 --out "$OUT/loader"
kill $VMSTAT || true
rm -rf "$BUF"
