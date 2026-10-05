# Source this file to use the LitePT GPU environment (works on CPU and GPU nodes):
#   source scripts/litePT/env.sh
# The pixi environment is defined in envs/litept-gpu/pixi.toml and installed on /ocean.
export EAGLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EAGLE_SCRATCH="${EAGLE_SCRATCH:-/ocean/projects/bio260075p/sammlapp/eagle}"
export CONDA_PREFIX="$(ls -d /ocean/projects/bio260075p/sammlapp/pixi_envs/eagle-litept-gpu-*/envs/default | head -1)"
export PATH="$CONDA_PREFIX/bin:$PATH"
case $- in *u*) _eagle_nounset=1; set +u ;; *) _eagle_nounset=0 ;; esac  # activate scripts use unset vars
for f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do . "$f"; done  # sets PROJ_DATA, GDAL_DATA, ...
[ "$_eagle_nounset" = 1 ] && set -u; unset _eagle_nounset
export PYTHONPATH="$EAGLE_ROOT/src:$EAGLE_ROOT/scripts/litePT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
