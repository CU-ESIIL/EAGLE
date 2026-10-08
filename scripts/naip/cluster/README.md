# NAIP <-> 3DEP coregistration database (cluster workflow)

Goal: a precomputed lookup. Given a lat/lon, return the 3DEP projects and NAIP surveys that cover it, and for each pairing a
calibration (the displacement field of the NAIP relative to the lidar) with QC. Method and evidence:
`scripts/naip/COREGISTRATION_NOTES.md`; model code: `src/eagle_als/coreg/`.

## Design in one paragraph
The NAIP-vs-lidar error is a smooth field (constant in flat open land, roughly linear across a few km in hilly terrain, plus a
height-dependent relief lean), so it is estimated once per **cell** (5 km UTM-km grid square, fitted on the cell + 500 m margin so
neighbours overlap) and **3DEP project**, for every NAIP survey (vintage) covering the cell. The lidar of a cell is streamed once and
re-used for all vintages. Units are independent JSON-writing tasks, so the run is resumable and embarrassingly parallel.

```
plan_units.py   3DEP registry (+ AOI / plot points)  ->  units.parquet        (unit = cell x project)
run_shard.py    units.parquet shard                  ->  OUT/done/<unit>.json  (one per unit; failures in OUT/failed/)
slurm_node.sbatch / submit_all.sh                       one 128-cpu node per array task, chained plan -> run -> merge
refine_plan.py  sparse results                       ->  extra cells where neighbouring fits disagree
merge_db.py     OUT/done/*.json                      ->  DB/{calibrations,coverage,units,failures}.parquet + meta.json
src/eagle_als/coreg/lookup.py  CalibrationDB(DB).lookup(lat, lon)  -> JSON-able dict
```

## Run it
```bash
cd scripts/naip/cluster
# (a) demand-driven: only cells containing our plots (+1 ring), newest lidar only -- do this first
python plan_units.py $W/units.parquet --points-file ../../../datasets/OFO_trees/plots_w_als.gpkg --k-ring 1 --newest-only
# (b) sparse wall-to-wall first pass (every 3rd cell each way), later refine
python plan_units.py $W/units.parquet --stride 3 --newest-only
# submit (edit PY/env in slurm_node.sbatch for the cluster; WORKERS x JOBS ~ 128)
N=$(python -c "import pandas as pd;print(-(-len(pd.read_parquet('$W/units.parquet'))//400))")
sbatch --array=0-$((N-1))%10 --export=ALL,N_SHARDS=$N,UNITS=$W/units.parquet,OUT=$W/out slurm_node.sbatch
# or everything chained: ./submit_all.sh $W --stride 3 --newest-only
python merge_db.py $W/out $W/db --units $W/units.parquet     # any time; re-run after resuming
python refine_plan.py $W/units.parquet $W/out $W/units_refine.parquet --stride 3   # then run_shard on the new table
PYTHONPATH=../../../src python -m eagle_als.coreg.lookup $W/db 40.0 -105.28
```
Resume = resubmit the same command (finished units are skipped). `--retry-failed` clears failures of the shard first.

## What a lookup returns
`lidar`: 3DEP projects containing the point (year, QL, EPT url). `naip`: for each survey the scene ids/dates/GSD and, per project,
`status/quality/model/r2/sd_xy_m/sun`, `ground_shift_m` (dx, dy east/north of the NAIP rel. to lidar at that point, h = 0) and
`by_height_m` (for 10 m, 25 m features: the lean). Correct NAIP by moving it by **-d**. `eval_displacement(record, E, N, h)`
evaluates any point/height. Quality: `good` (R2 >= 0.10 and bootstrap sd <= 0.5 m), `fair`, `poor` -- thresholds are provisional.
Cells not in the DB report `cell_not_in_database`; NAIP surveys with a failed/poor fit are listed but flagged, never silently dropped.

## Cost (measured on a laptop, scaled -- treat as +-2x)
Smoke test: one 2.4 km Iowa cell (QL2), 2 NAIP surveys, 4 processes: 241 s wall (lidar 46 s, ~100 s per survey = ~400 cpu-s per
survey). A production cell is 6 km x 6 km (5.6x the pixels): ~0.6 cpu-h per survey, ~0.2-1 cpu-h of lidar streaming (QL1 dense
urban is the expensive end: ~30 M points/km2). With ~6 surveys per cell that is roughly **4-5 cpu-h per unit**, memory ~8 GB per
unit at 1 m. The registry (computed with `plan_units.py`): **415,157 units on 283,684 distinct cells, 2,187 projects**
(2008-2024; 2018-2019 dominate).

| plan | units | cpu-h | 128-cpu node-days |
|---|---|---|---|
| everything | 415 k | ~1.9 M | ~600 |
| newest lidar per cell | 284 k | ~1.3 M | ~420 |
| newest lidar, stride 3 (sparse pass) | ~32 k | ~145 k | ~47 |
| newest lidar, stride 2 | ~71 k | ~320 k | ~100 |
| plots only (N cells + 1 ring) | ~9 N | ~40 N | -- |
Levers not yet validated: NAIP surveys >= 2015 only (about -40 %), 2 m analysis grid (`--res 2`, about 3-4x cheaper fits; accuracy
at 2 m must be checked), fitting every other block (2x). Recommended order: plots-first, then a stride-3 sparse pass, then
`refine_plan.py`.

## Operational notes / not yet exercised
- Tested here: plan, run_shard (1 worker x 4 jobs), merge, lookup on one real cell. **Not tested**: SLURM scripts, 25 workers per node,
  thousands of units, network behaviour from a cluster.
- Network: ~100 concurrent readers hit USGS S3 (us-west-2) and Planetary Computer; keep `%10` node limits, expect throttling.
  SAS tokens expire in ~1 h, so NAIP hrefs are signed at read time (not at search time). A cluster on AWS us-west-2 gets the lidar
  fastest; NAIP could be read from the `naip-analytic` requester-pays bucket instead.
- Failure handling: a failing NAIP survey is recorded (`ok: false`) without losing the other surveys; a failing unit goes to `failed/`.
- Small cells overfit: the 2.4 km smoke cell chose an 'affine' model for NAIP 2021 and extrapolated to (-1.85, -0.42) m at a point
  where the 5 km analysis gave about (-0.4, 0.1). Production cells (>= 5 km) avoid this; do not shrink them without raising `cv_gain`.
- Cell seams: neighbouring cells overlap by 500 m but lookups use the containing cell only (no blending); NAIP seamlines inside a
  cell are not modelled. Edge distance is returned (`dist_to_cell_edge_m`) so a client can blend later.
- QC thresholds are provisional until the landmark (held-out building) evaluation is done.
- Version/provenance: every unit stores the git short hash, NAIP item ids, grid/resolution; changing the model => new DB directory.
