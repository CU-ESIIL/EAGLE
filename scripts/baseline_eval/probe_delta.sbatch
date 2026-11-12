#!/bin/bash
# Linear probes (02_linear_probe.py) as a batch job on an NCSA Delta CPU node. Arguments go to 02_linear_probe.py;
# run all tasks of a model in one job (the script rewrites results/<model>/summary.* with only the tasks it runs).
#   sbatch scripts/baseline_eval/probe_delta.sbatch --naip-clay /work/hdd/bjud/sammlapp/benchmarking/naip_clay --model naip_clay
#   sbatch scripts/baseline_eval/probe_delta.sbatch --embeddings <dir> --model <name>
# Budget: 16 cores x 4 h = 64 CPU-hours at most (AGENTS.md: ask the user before jobs above 300 CPU-hours). A full NAIP
# Clay probe (1024-d, all tasks) is an estimated 25-45 CPU-hours; small requests start sooner (backfill).
# Results are written to results/<model>/ in the repo after every task, so a time-out keeps finished tasks.
#SBATCH --job-name=eagle-probe
#SBATCH --account=bjud-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bjud/sammlapp/benchmarking/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[probe_delta] $(date) $(hostname) job $SLURM_JOB_ID: $*"
~/.pixi/bin/pixi run --manifest-path envs/eagle-bench/pixi.toml env PYTHONPATH=src \
    python scripts/baseline_eval/02_linear_probe.py --n-jobs "$SLURM_CPUS_PER_TASK" "$@"
echo "[probe_delta] $(date) done"
