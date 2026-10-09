# Snapshot USA per-location evaluation set, species with >= 20 locations in each split ("-20")

The stricter variant of the per-location tables in [../SNAPSHOT_USA_eval/](../SNAPSHOT_USA_eval/README.md) (read its "Per-location targets" section for how locations are pooled, the one-lidar-product-per-location rule, columns and caveats). The rows, split and lidar match are **identical**; only the species differ.

**Species rule:** wild species (not human or domestic) detected at **at least 20 locations in the train split and at least 20 in the test split**: **39 species** (of 269 wild species detected). The lowest counts are 25 train / 23 test locations (moose 26/26, white-footed mouse 30/24, American badger 31/26, grey wolf 29/29); the most common are white-tailed deer (1,282 train, 1,324 test), northern raccoon and eastern gray squirrel.

| file | contents |
|---|---|
| `location_occupancy.parquet` | 3,409 locations (1,715 train, 1,694 test): location columns plus `occ__<species>` (0/1) for the 39 species |
| `location_detection_rate.parquet` | same rows with `rate__<species>` (independent events per 100 camera-nights, pooled over the location's deployments) |
| `location_species_summary.csv` | all 276 species detected: locations present in train and test, prevalence, `included` (True for the 39) |

54 locations have none of the 39 species (all zeros). The 39 species are a subset of the 45 eligible species of the deployment table; six of those fall below 20 locations in one split once deployments are pooled (pronghorn, Gambel's quail, common raven, white-tailed jackrabbit, long-tailed weasel, western gray squirrel).

Regenerate: `pixi run python data_cleaning/snapshot_usa/04_location_targets.py --min-locations 20`. Any other N writes `datasets/SNAPSHOT_USA_eval-N/`.
