#!/bin/bash
# lidR stdmetrics (+ terrain) of every cached eval cookie (05_lidr_metrics.py) on an NCSA Delta CPU node, one
# single-threaded Rscript per worker. Arguments go to the script; it is resumable (a time-out just needs a resubmit).
#   sbatch scripts/baseline_eval/lidr_metrics_delta.sbatch --radius-m task      # task radii, capped at 100 m
# Cost: ~1.2 core-s per 50 m cookie, ~7 core-s per 100 m cookie (2026-11-12): ~125 core-h for the task radii.
# 96 cores x 3 h = 288 CPU-h at most (AGENTS.md: ask before > 300 CPU-h). Batches of 10 cookies keep memory
# (100 m cookies: up to ~1M points each) well under the node's RAM.
#SBATCH --job-name=lidr-metrics
#SBATCH --account=bjud-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bjud/sammlapp/benchmarking/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
W=/work/hdd/bjud/sammlapp/benchmarking
echo "[lidr_delta] $(date) $(hostname) job $SLURM_JOB_ID: $*"
ulimit -c 0
~/.pixi/bin/pixi run --manifest-path envs/eagle-bench/pixi.toml env PYTHONPATH=src \
    python scripts/baseline_eval/05_lidr_metrics.py --cache "$W/als_cache" --out "$W/features/lidr_stdmetrics" \
    --workers "$((SLURM_CPUS_PER_TASK - 4))" --batch 10 "$@"
echo "[lidr_delta] $(date) done"
