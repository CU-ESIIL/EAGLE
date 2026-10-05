#!/bin/bash
# Fetch + validate lidar for labeled datasets and write *_als_available tables.
#   sbatch scripts/litePT/slurm/check_sources.sbatch bbs butterflies
#   sbatch scripts/litePT/slurm/check_sources.sbatch --table path/to/vegbank.parquet --name vegbank --lat-col lat --lon-col lon
#SBATCH --job-name=eagle-check
#SBATCH --account=bio260075p
#SBATCH --partition=RM-shared
#SBATCH --ntasks-per-node=32
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh
if [[ "${1:-}" == --* ]]; then
  python scripts/litePT/check_lidar_sources.py --workers 48 "$@"
else
  for ds in "$@"; do
    python scripts/litePT/check_lidar_sources.py --dataset "$ds" --workers 48
  done
fi
