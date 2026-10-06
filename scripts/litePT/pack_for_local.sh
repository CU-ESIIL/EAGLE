#!/bin/bash
# Build the single-file archives that stage_local.sh extracts onto node-local disk in each job.
# Re-run after changing the environment (pixi install) or rebuilding a cookie cache.
#   bash scripts/litePT/pack_for_local.sh env            # -> pixi_envs/eagle-litept-gpu-env.tar.zst
#   bash scripts/litePT/pack_for_local.sh cache sites    # $EAGLE_SCRATCH/cache/sites -> cache/sites.tar
set -euo pipefail
ENVS=/ocean/projects/bio260075p/sammlapp/pixi_envs
EAGLE_SCRATCH="${EAGLE_SCRATCH:-/ocean/projects/bio260075p/sammlapp/eagle}"
case "${1:-}" in
  env)
    src="$(ls -d $ENVS/eagle-litept-gpu-*/envs | head -1)"
    tar -C "$src" -cf - default | "$src/default/bin/zstd" -T8 -3 -q -o "$ENVS/eagle-litept-gpu-env.tar.zst.tmp"
    mv "$ENVS/eagle-litept-gpu-env.tar.zst.tmp" "$ENVS/eagle-litept-gpu-env.tar.zst"
    ls -la "$ENVS/eagle-litept-gpu-env.tar.zst" ;;
  cache)
    for c in "${@:2}"; do  # cookies are already compact binary arrays; plain tar extracts fastest
      tar -C "$EAGLE_SCRATCH/cache" -cf "$EAGLE_SCRATCH/cache/$c.tar.tmp" "$c"
      mv "$EAGLE_SCRATCH/cache/$c.tar.tmp" "$EAGLE_SCRATCH/cache/$c.tar"
      ls -la "$EAGLE_SCRATCH/cache/$c.tar"
    done ;;
  *) echo "usage: $0 env | cache <name> ..."; exit 1 ;;
esac
