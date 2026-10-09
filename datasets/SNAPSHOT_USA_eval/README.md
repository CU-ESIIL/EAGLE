# Snapshot USA species occupancy and encounter-rate evaluation table for lidar representations

Camera-trap deployments from Snapshot USA (2019-2023) matched to 3DEP lidar collected within 3 years of the survey. The targets are, for each species, **presence/absence (occupancy)** and **encounter rate** at the deployment. Cleaning code and rules: [data_cleaning/snapshot_usa/](../../data_cleaning/snapshot_usa/README.md).

## Files

| file | contents |
|---|---|
| `snapshot_usa_eval.parquet` | one row per deployment (4,761 rows, 114 columns): metadata, effort, lidar match, split, and the targets of the 45 eligible species |
| `species_events_long.parquet` | every detected species in every matched deployment (only presences): `deployment_key`, `species`, `common_name`, `n_sequences`, `n_events`, `encounter_rate_per_100n`. A species missing for a deployment is an absence |
| `species_summary.csv` | one row per species detected (276): class, `is_human_or_domestic`, sequences, events, deployments and locations, deployments present and prevalence in train and test, `eligible` |
| `location_occupancy.parquet` | **one row per camera location** (3,409), one matched lidar product each, with `occ__<species>` (0/1) for 159 species: see "Per-location targets" |
| `location_detection_rate.parquet` | the same rows with `rate__<species>` (events per 100 camera-nights) for the same species |
| `location_species_summary.csv` | per species at the location level: locations present in train and test, prevalence, `included` |
| `figures/` | `01_split_map.png`, `02_species_presence_by_split.png` |

## Columns of `snapshot_usa_eval.parquet`

| column | meaning |
|---|---|
| `deployment_key` | `<Year>\|<Deployment_ID>`; unique. The raw `deployment_id` alone is not unique across years |
| `ssusa_year`, `Project`, `Camera_Trap_Array`, `Site_Name` | source identifiers (160 arrays; Snapshot USA 2019-2023 plus a few partner projects) |
| `lat`, `lon`, `n_decimals`, `location_id` | WGS84 camera location, decimals in the source text (>= 4), and the site id (coordinates rounded to 4 decimals). 2,629 locations have one deployment, 780 have 2-6 (re-surveys, other years or seasons) |
| `year`, `Start_Date`, `End_Date`, `Survey_Nights` | survey year (year of the start date), dates and effort (7-142 nights, median 36) |
| `Habitat`, `Development_Level`, `Feature_Type` | observer-recorded: Forest 3,574 / Grassland 684 / Anthropogenic 280 / Desert 148 / Wetland 62 / Chaparral 13; Wild 1,786 / Rural 1,733 / Suburban 922 / Urban 320. `Feature_Type` (trail, road, water source, ...) is null for 71% of rows |
| `n_species_wild` | wild species identified to species in the deployment (mean 4.9, 1.3% have none): a possible richness target |
| `n_seq_unresolved` | sequences not identified to species (birds, rodents, genus or family level): larger values make a species' absence less certain |
| `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS`, `als_site_id` | 3DEP project covering the camera, its collection year, the absolute gap to the survey year (0-3: 914, 1,601, 1,140, 1,106 deployments) and `<product>__<lat>_<lon>` (3,595 unique) |
| `test_split` | True for 2,381 rows, assigned by whole 0.1-degree lat/lon blocks (371 blocks): no location has deployments in both splits |
| `occ__<species>` | **presence (1) / absence (0)** of the species in the deployment |
| `rate__<species>` | **encounter rate**: independent events (30-min rule) per 100 camera-nights; 0 when absent |

## Targets: which species

`occ__` and `rate__` columns exist for the **45 eligible species**: wild, and detected in at least 20 deployments of both train and test (`MIN_PRESENT_PER_SPLIT` in `03_split_and_export.py`). Prevalence ranges from 79% (white-tailed deer) and 50% (raccoon) in the common species to about 1% (long-tailed weasel, Gambel's quail) in the rarest, so pick species by prevalence in `species_summary.csv`. 276 species were detected overall (269 wild); the rest are in the long table only. To score a species outside the 45, build its target from the long table (`present = deployment_key in species_events_long`, rate over `Survey_Nights`).

## Per-location targets

`04_location_targets.py` pools the deployments of each site (`location_id`, ~11 m) into one row, in two separate tables with identical location columns: `location_occupancy.parquet` (`occ__<species>`) and `location_detection_rate.parquet` (`rate__<species>`).

- **One lidar product per location.** The product matched by most of the location's deployments (ties: smaller mean year gap) is used, and only deployments matched to it are pooled, so every label refers to the lidar of its row. 12 locations had deployments on a second product; 22 of the 4,761 deployments are left out (4,739 pooled). Columns: `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS_mean`, `year_diff_AWS_max`, `als_site_id` (from the mean coordinates; 10 locations share an id with a neighbour, so use `location_id` as the key).
- **Occupancy** is 1 if the species was detected in any pooled deployment. **Detection rate** is the summed independent events divided by the summed survey nights, times 100. Effort columns: `n_deployments` (1 for 2,629 locations, up to 6), `total_survey_nights`, `first_year`, `last_year` (654 locations span several years; occupancy of a multi-year location is "ever detected", which is easier to be 1 than for a one-deployment location, so keep `total_survey_nights` in mind).
- **Species included:** wild species detected at at least one location in **each** of train and test (`MIN_LOCATIONS_PER_SPLIT = 1`): **159** of 269 wild species (110 are found in only one split). This is a low bar: 73 included species have fewer than 5 locations in train or test, and 120 fewer than 20, so scores for them are noisy or undefined (one test positive). Filter further on `n_locations_train` and `n_locations_test` in `location_species_summary.csv` (`--min-locations 20` gives the 39-species set in [../SNAPSHOT_USA_eval-20/](../SNAPSHOT_USA_eval-20/README.md)).
- **Split:** 1,715 train and 1,694 test locations (by 0.1-degree block, the same assignment as the deployment table).
- **Other columns:** `lat`, `lon` (mean of pooled deployments), `Habitat`, `Development_Level`, `Project`, `Camera_Trap_Array` (most common value), `n_species_wild`.

## Caveats

- **Absence is "not detected".** A camera sees a small area; small, arboreal, aerial and nocturnal species (birds, rodents, flying squirrels) are detected unreliably, and detection depends on camera placement (trail, road, water) and effort. Encounter rate is the usual index of use but is not abundance. Compare models on the same effort; `Survey_Nights` and `n_seq_unresolved` are in the table.
- **Range effects.** Most of the signal for a rare species is geographic (the species is not found in that region), so a lidar embedding can score well by encoding location rather than habitat. A useful check is to restrict a species to the deployments within its range (for example the ecoregions where it occurs) or compare against a latitude/longitude-only baseline.
- **Seasonal and annual differences.** Surveys are mostly late summer to autumn. Species and survey effort differ by year and project; the test split holds out places, not years.
- **Events.** One sequence is a burst of images and not an independent visit; 30-minute independence is a convention. `n_sequences` is in the long table.
- **Turkey.** `Meleagris gallopavo` combines wild and domestic turkey labels. `Sus scrofa` is feral hog.
- **Clustering.** 4,761 deployments in 3,409 locations and 371 blocks; many arrays are dense grids of cameras that share one lidar product (141 products in total) and nearby cookies overlap. Test scores on neighbouring blocks are not independent.
- **Footprint matching only.** 3DEP footprints can include gaps; no point-cloud availability check was run. The `eagle_als` streaming check can be run on `deployment_key`/`lat`/`lon`/`als_site_id`.
- **Leaf-on.** Surveys and lidar flights are in different seasons; the AWS registry has `leaf_on` for the lidar if you need to filter on it.

## Regenerating

```
pixi run python data_cleaning/snapshot_usa/01_ingest_and_count.py
pixi run python data_cleaning/snapshot_usa/02_filter_and_match_als.py
pixi run python data_cleaning/snapshot_usa/03_split_and_export.py
```
