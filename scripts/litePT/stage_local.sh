# Source at the start of a slurm job instead of env.sh: copies what the job reads onto node-local
# disk ($LOCAL) and activates the Python environment from there.
#   source scripts/litePT/stage_local.sh [cache ...]      # e.g. source scripts/litePT/stage_local.sh sites
#
# /ocean reads large files quickly but can stall for minutes to an hour on many small reads (Python
# imports from the ~72k-file environment), so the environment and the evaluation cookie caches are
# shipped as single archives, built once with pack_for_local.sh:
#   $EAGLE_ENV_ARCHIVE                 -> $LOCAL/eagle_env/default   (exported as EAGLE_ENV_PREFIX)
#   $EAGLE_SCRATCH/cache/<cache>.tar   -> $LOCAL/eagle_cache/<cache> (EAGLE_LOCAL_CACHE=$LOCAL/eagle_cache)
# Without $LOCAL, or with an archive missing, it falls back to the copies on /ocean.
_eagle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_eagle_ocean_env="$(ls -d /ocean/projects/bio260075p/sammlapp/pixi_envs/eagle-litept-gpu-*/envs/default | head -1)"
export EAGLE_SCRATCH="${EAGLE_SCRATCH:-/ocean/projects/bio260075p/sammlapp/eagle}"
EAGLE_ENV_ARCHIVE="${EAGLE_ENV_ARCHIVE:-/ocean/projects/bio260075p/sammlapp/pixi_envs/eagle-litept-gpu-env.tar.zst}"

if [ -n "${LOCAL:-}" ] && [ -d "$LOCAL" ] && [ -f "$EAGLE_ENV_ARCHIVE" ]; then
  if [ ! -f "$LOCAL/eagle_env/.complete" ]; then
    _t=$(date +%s)
    rm -rf "$LOCAL/eagle_env" && mkdir -p "$LOCAL/eagle_env"
    "$_eagle_ocean_env/bin/zstd" -d -q -T0 -c "$EAGLE_ENV_ARCHIVE" | tar -x -C "$LOCAL/eagle_env" \
      && touch "$LOCAL/eagle_env/.complete"
    echo "[stage] environment -> $LOCAL/eagle_env in $(( $(date +%s) - _t ))s"
  fi
  if [ -f "$LOCAL/eagle_env/.complete" ]; then export EAGLE_ENV_PREFIX="$LOCAL/eagle_env/default"; fi
else
  echo "[stage] no \$LOCAL or no $EAGLE_ENV_ARCHIVE: using the environment on /ocean"
fi
source "$_eagle_dir/env.sh"

if [ -n "${LOCAL:-}" ] && [ -d "$LOCAL" ]; then
  export EAGLE_LOCAL_CACHE="$LOCAL/eagle_cache"
  for _c in "$@"; do
    if [ -d "$EAGLE_LOCAL_CACHE/$_c" ]; then continue; fi
    if [ ! -f "$EAGLE_SCRATCH/cache/$_c.tar" ]; then echo "[stage] missing $EAGLE_SCRATCH/cache/$_c.tar"; continue; fi
    _t=$(date +%s)
    mkdir -p "$EAGLE_LOCAL_CACHE" && tar -xf "$EAGLE_SCRATCH/cache/$_c.tar" -C "$EAGLE_LOCAL_CACHE"
    echo "[stage] cache $_c -> $EAGLE_LOCAL_CACHE/$_c in $(( $(date +%s) - _t ))s"
  done
fi
unset _eagle_dir _eagle_ocean_env _t _c
