#!/bin/bash
# Fetch the 3DEP cookie of every benchmark site into the shared cache (resumable; safe to rerun at any time).
# Runs detached in tmux so it survives the terminal / agent session:
#   tmux new-session -d -s als_cache "bash scripts/baseline_eval/run_lidar_cache.sh"
#   tmux attach -t als_cache            # watch (detach: Ctrl-b d)
#   tail -f $BENCH/als_cache/logs/cache_*.log
# Progress: python -c "from eagle_als.cache import read_manifest; print(read_manifest('$BENCH/als_cache').status.value_counts())"
# Sites come from 03_lidar_sites.py (priority order: small tasks first, SatBird last). Extra args go to eagle_als.cache.
#
# Platform settings (environment variables; defaults = CyVerse):
#   BENCH     cache root (sites table at $BENCH/als_cache/sites.parquet)
#   EAGLE_ENV prefix of a conda-style env with python-pdal; if unset, the pixi env envs/eagle-bench is used
#   WORKERS   parallel fetches (CyVerse: 12 on 4 CPUs; each fetch is ~6-20 s network wait + ~2.5 s CPU)
# NCSA Delta (32-CPU job; env built with the miniforge3-python module, see CLAUDE.md):
#   BENCH=/work/hdd/bjud/sammlapp/benchmarking EAGLE_ENV=/work/hdd/bjud/sammlapp/envs/eagle-bench WORKERS=64 \
#     bash scripts/baseline_eval/run_lidar_cache.sh --max-hours 7
set -uo pipefail
# PDAL occasionally aborts a worker (eagle_als.cache records it and moves on); Delta allows unlimited core dumps,
# which wrote 1.5-2.9 GB per crash into the working directory. Do not write them.
ulimit -c 0
cd "$(dirname "$0")/../.."
BENCH=${BENCH:-/home/jovyan/data-store/data/iplant/home/shared/esiil/als/benchmarking}
WORKERS=${WORKERS:-12}
OUT=$BENCH/als_cache
mkdir -p "$OUT/logs"
LOG="$OUT/logs/cache_$(date +%Y%m%d_%H%M%S).log"
export PROJ_NETWORK=OFF PYTHONPATH=src OMP_NUM_THREADS=1
# cap PDAL readers.ept threads per fetch (default would be one per hardware thread of the node; see eagle_als.fetch)
export EAGLE_PDAL_THREADS=${EAGLE_PDAL_THREADS:-4}
if [ -n "${EAGLE_ENV:-}" ]; then
    export PROJ_DATA=$EAGLE_ENV/share/proj GDAL_DATA=$EAGLE_ENV/share/gdal
    PY=("$EAGLE_ENV/bin/python")
else
    PY=(~/.pixi/bin/pixi run --manifest-path envs/eagle-bench/pixi.toml python)
fi
echo "[run_lidar_cache] $(date) start on $(hostname), $WORKERS workers, log $LOG" | tee -a "$LOG"
"${PY[@]}" -m eagle_als.cache \
    --sites "$OUT/sites.parquet" --out "$OUT" --id-col als_site_id --workers "$WORKERS" "$@" 2>&1 \
    | grep --line-buffered -v "PDAL Error" >> "$LOG"
echo "[run_lidar_cache] $(date) exit ${PIPESTATUS[0]}" | tee -a "$LOG"
