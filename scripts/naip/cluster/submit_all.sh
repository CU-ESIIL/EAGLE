#!/bin/bash
# plan -> run (array of nodes) -> merge, chained with dependencies.   usage: ./submit_all.sh WORKDIR [plan_units.py args...]
set -euo pipefail
W=${1:?workdir}; shift
mkdir -p "$W"; cd "$(dirname "$0")"
python plan_units.py "$W/units.parquet" "$@"
N=$(python - <<PY
import pandas as pd; n=len(pd.read_parquet("$W/units.parquet")); print(max(1, min(200, -(-n // 400))))   # ~400 units per node-task
PY
)
JID=$(sbatch --parsable --array=0-$((N-1))%10 --export=ALL,N_SHARDS=$N,UNITS="$W/units.parquet",OUT="$W/out" slurm_node.sbatch)
sbatch --dependency=afterany:$JID --job-name=coreg_merge --wrap="python $(pwd)/merge_db.py $W/out $W/db --units $W/units.parquet"
echo "submitted array $JID with $N shards"
