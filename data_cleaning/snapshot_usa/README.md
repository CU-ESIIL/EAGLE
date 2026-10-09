# Snapshot USA camera traps: species occupancy and encounter-rate pipeline

Scripts that read the Snapshot USA 2019-2023 camera-trap tables, count each species' detections per camera deployment, keep deployments with a precise location and enough effort, match each to the 3DEP lidar product closest in time (within 3 years), and write the evaluation tables: one row per deployment, and one row per camera location (separate occupancy and detection-rate tables) in [datasets/SNAPSHOT_USA_eval/](../../datasets/SNAPSHOT_USA_eval/README.md), plus the stricter per-location set [datasets/SNAPSHOT_USA_eval-20/](../../datasets/SNAPSHOT_USA_eval-20/README.md). Columns, targets and caveats are documented there.

## Raw data

`datasets/raw/snapshotUSA_2019-2023/` (gitignored), unmodified, downloaded from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.k0p2ngfhn#methods)

- `ssusa_finaldeployments.csv`: 9,679 camera deployments (Year, Project, array, site, dates, `Survey_Nights`, lat/lon, habitat, development level, feature type)
- `ssusa_finalsequences.csv`: 987,979 annotated detection sequences (taxon, group size, sex, age)

Both files are the Snapshot USA / eMammal compilation; the files themselves record no download date.

## Run order

Run from the repo root with `pixi run python data_cleaning/snapshot_usa/<script>` (about 30 s in total). Stage 4 also runs as `... 04_location_targets.py --min-locations 20` for the "-20" set.

| script | does | writes |
|---|---|---|
| `01_ingest_and_count.py` | reads both tables, joins them, counts sequences and independent events per (deployment, species) | `datasets/raw/snapshotUSA_2019-2023/derived/*.parquet` |
| `02_filter_and_match_als.py` | coordinate, effort and ALS filters | `derived/deployments_matched.parquet`, `catalog/filter_counts_stage2.csv`, `catalog/year_gap_distribution.csv` |
| `03_split_and_export.py` | targets for eligible species, 50% spatial hold-out, figures | `datasets/SNAPSHOT_USA_eval/`: `snapshot_usa_eval.parquet` (one row per deployment, targets for 45 species), `species_events_long.parquet`, `species_summary.csv`, `figures/` |
| `04_location_targets.py` | pools deployments into one row per location with one lidar product; separate occupancy and detection-rate tables; species present in >= 1 location of each split (`--min-locations N` writes `datasets/SNAPSHOT_USA_eval-N/`, e.g. 20) | `datasets/SNAPSHOT_USA_eval/location_occupancy.parquet`, `location_detection_rate.parquet`, `location_species_summary.csv`; with `--min-locations 20`, the same three files in `datasets/SNAPSHOT_USA_eval-20/` |

## Rules

| rule | why |
|---|---|
| join sequences to deployments on **(Year, Deployment_ID)** | `Deployment_ID` is reused in a later year for 46 sites; joining on the ID alone puts 4,710 sequences outside their deployment's dates. With the pair, every sequence is inside its dates; 69 sequences have no deployment and are dropped |
| species = `Genus` + `Species` | records identified only to genus, family or class (Bird, Rodent, `Peromyscus Species`, `Sciuridae Family`, `Animal`, `Unknown`) are not species targets; they are counted in `n_seq_unresolved` per deployment (70,150 sequences) |
| independent events | sequences of one species at one deployment less than 30 min after the previous one's end are merged (`EVENT_GAP_MIN`); 891,883 species sequences become 607,336 events |
| >= 4 decimals in lat and lon (source text) | 301 deployments are rounded to 1-3 decimals (100 m to 10 km) |
| >= 7 survey nights | 297 deployments run 1-6 nights, too short for presence/absence (`MIN_NIGHTS`) |
| 3DEP footprint and `year_diff_AWS <= 3` | same +/-3 year rule as the other datasets, relative to the year of `Start_Date` |
| humans and domestic animals flagged, not scored | `Homo sapiens`, cattle, dog, cat, horse, donkey, goat, sheep (`HUMAN_DOMESTIC`) |
| species targets for deployments: >= 20 deployments present in both splits | 45 species (`MIN_PRESENT_PER_SPLIT` in stage 3) |
| location = `location_id`, coordinates rounded to 4 decimals (~11 m) | stage 4 pools the deployments of a site into one row |
| one 3DEP product per location | the product matched by most of the location's deployments (ties: smaller mean year gap); only deployments on it are pooled, so a label always refers to its row's lidar. 22 deployments at 12 locations are left out |
| occupancy = detected in any pooled deployment; detection rate = summed events / summed survey nights x 100 | pooled over the location's deployments (up to 6, up to 5 years) |
| species at location level: wild, present at >= N locations in each split | N = 1 gives 159 species, N = 20 gives 39 (`--min-locations`) |

## Results

| step | deployments left |
|---|---|
| all | 9,679 |
| >= 4 decimals | 9,378 |
| >= 7 survey nights | 9,081 |
| covered by 3DEP | 8,022 |
| within +/-3 yr of lidar | **4,761** (3,409 locations) |

| location table (stage 4) | rows | species |
|---|---|---|
| `SNAPSHOT_USA_eval` (N = 1) | 3,409 locations (1,715 train, 1,694 test), 4,739 pooled deployments | 159 |
| `SNAPSHOT_USA_eval-20` (N = 20) | same locations | 39 |

Unlike most evaluation sets the +/-3 year rule keeps 59% of covered deployments (surveys are 2019-2023, lidar is recent); the gap distribution of covered deployments is in `catalog/year_gap_distribution.csv` (0: 914, 1: 1,601, 2: 1,140, 3: 1,106, 4: 910, 5: 724, ...).
