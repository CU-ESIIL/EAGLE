#!/bin/bash
# Extract AlphaEarth embeddings for all benchmark tasks (network-bound, no GPU; about 1 TB read from
# Source Cooperative, ~100 MB written). Resumable: resubmit the same command after a time-out.
#   sbatch scripts/baseline_eval/extract_aef.sbatch                  # all tasks
#   sbatch scripts/baseline_eval/extract_aef.sbatch --tasks nlcd ofo_trees   # arguments go to 01_extract_aef.py
# Needs python with rasterio, pyproj, geopandas, pandas, pyarrow, scikit-learn, joblib, scipy, tabulate
# (see scripts/baseline_eval/README.md, "Environment"). Check partition/account names for your allocation.
#SBATCH --job-name=aef-extract
#SBATCH --account=bio260075p
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/ocean/projects/bio260075p/sammlapp/eagle/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/litePT/env.sh            # puts src/ on PYTHONPATH; swap for the env that has rasterio
export PROJ_NETWORK=OFF
python scripts/baseline_eval/01_extract_aef.py --workers 32 "$@"
python scripts/baseline_eval/02_linear_probe.py
