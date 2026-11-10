#!/bin/bash
# Fetch the 3DEP cookie of every benchmark site into the shared cache (resumable; safe to rerun at any time).
# Runs detached in tmux so it survives the terminal / agent session:
#   tmux new-session -d -s als_cache "bash scripts/baseline_eval/run_lidar_cache.sh"
#   tmux attach -t als_cache            # watch (detach: Ctrl-b d)
#   tail -f $BENCH/als_cache/logs/cache_*.log
# Progress: python -c "from eagle_als.cache import read_manifest; print(read_manifest('$BENCH/als_cache').status.value_counts())"
# Sites come from 03_lidar_sites.py (priority order: small tasks first, SatBird last). Extra args go to eagle_als.cache.
set -uo pipefail
cd "$(dirname "$0")/../.."
BENCH=/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking
OUT=$BENCH/als_cache
mkdir -p "$OUT/logs"
LOG="$OUT/logs/cache_$(date +%Y%m%d_%H%M%S).log"
export PROJ_NETWORK=OFF PYTHONPATH=src OMP_NUM_THREADS=1
echo "[run_lidar_cache] $(date) start, log $LOG" | tee -a "$LOG"
# 12 workers: each fetch is mostly network wait (~6-20 s) plus ~2.5 s CPU; the machine has 4 CPUs.
~/.pixi/bin/pixi run --manifest-path envs/eagle-bench/pixi.toml python -m eagle_als.cache \
    --sites "$OUT/sites.parquet" --out "$OUT" --id-col als_site_id --workers 12 "$@" 2>&1 \
    | grep --line-buffered -v "PDAL Error" >> "$LOG"
echo "[run_lidar_cache] $(date) exit ${PIPESTATUS[0]}" | tee -a "$LOG"
