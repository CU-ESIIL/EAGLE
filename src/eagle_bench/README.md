# `eagle_bench`

Reusable pieces of the benchmark suite (no torch needed). Used by `scripts/baseline_eval/`.

- `tasks.py`: one loader per task (`load_task(name)`) returning a `Task`: locations, year, the dataset's own
  train/test split and targets (multiclass, multilabel occurrence, regression). 15 tasks, listed in `TASKS`.
- `aef.py`: AlphaEarth annual embeddings at points from the public Source Cooperative COGs (`locate`, `extract`,
  `load_points`, `matrix_for`). Uses the survey year, or the nearest year with imagery.
- `naip_clay.py`: Clay v1.5 NAIP chip embeddings (source.coop `clay-v1-5-naip-2`) averaged over a disc around a point; file lookup by quarter-quad arithmetic, nothing cached but the pooled sums.
- `probes.py`: `evaluate(task, X, min_count=None)` (min_count = rows per class in each split for it to be scored; 1 = all classes, 20 = stable ones) fits a linear probe (logistic or ridge) on train rows and scores test rows.

To benchmark another embedding, build an `(n_rows, d)` matrix aligned to `task.df` (key on `als_site_id` for
lidar cookies, or `lat`/`lon`) and call `probes.evaluate`. Add a task by writing a loader in `tasks.py`.
