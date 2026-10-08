#!/bin/bash
# Process shards of the unit table, one 128-cpu node per array task.
#   N=40; sbatch --array=0-$((N-1))%10 --export=ALL,N_SHARDS=$N,UNITS=units.parquet,OUT=/scratch/$USER/coreg_out slurm_node.sbatch
# %10 = at most 10 nodes at a time (be polite to the S3/Planetary Computer endpoints). Resubmit the same command to resume.
#SBATCH --job-name=coreg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/coreg_%A_%a.log

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
WORKERS=${WORKERS:-25}      # concurrent units per node; memory ~8 GB each at 1 m (lower WORKERS on small-memory nodes)
JOBS=${JOBS:-5}             # processes inside a unit (WORKERS x JOBS ~ 128)
# environment: pixi env of this repo (python-pdal, rasterio, ...) -- adjust to the cluster's module/conda setup
# source /path/to/pixi-env/activate  OR  module load ...
PY=${PY:-python}
$PY run_shard.py "$UNITS" "$OUT" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${N_SHARDS:-1}" --workers "$WORKERS" --jobs "$JOBS" ${EXTRA:-}
