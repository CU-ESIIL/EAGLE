# SatBird USA-winter evaluation set

One row per eBird hotspot of the original SatBird USA-winter dataset (47,688 hotspots, 1,181 lidar products) with the encounter rate of 670 bird species (December-January checklists, 2010-2023) and the 3DEP lidar product collected in 2010-2023 (the label years; rule below). Built by `data_cleaning/satbird/` from [sammlapp/SatBird-labels](https://huggingface.co/datasets/sammlapp/SatBird-labels) (SatBird, Teng et al. 2023, CC BY-NC 4.0; eBird terms of use apply). Coordinates, labels and split are SatBird's own.

## Tables

All share the same rows and split. Every species present at >= 1 hotspot in each of train and test is published (587 of 670); the minimum count is an evaluation setting (467 species have >= 20 hotspots in each of train and test; see `location_species_summary.csv`).

| target | table | columns |
|---|---|---|
| encounter rate (0-1) | `location_encounter_rate.parquet` | `enc__<Genus_species>` |
| occupancy (0/1; rate > 0) | `location_occupancy.parquet` | `occ__<Genus_species>` |

## Columns

`hotspot_id` (eBird locId, unique), `lat`, `lon` (as published by SatBird), `year` (2022 = reference only), `state_code`, `county_code`, `num_complete_checklists`, `num_different_species` (SatBird), `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` (always 0), `label_year_min`, `label_year_max` (2010, 2023), `als_site_id`, `split` (`train` / `val` / `test`: SatBird's), `test_split` (True = test only), then the species columns. Species are named by scientific name (SatBird's `species_list.txt`).

## Caveats

- **Split is SatBird's** (DBSCAN 5 km clusters, random 70/15/15), so scores are comparable in design to published SatBird numbers, though not identical (hotspots without lidar are removed). After the lidar filter: 34,327 train / 7,174 val / 6,187 test (test is 13%). `test_split` is False for val; the benchmark task drops val.
- **Pooled labels and the lidar rule.** Each rate pools a season's complete checklists over 2010-2023, so there is no survey date. A hotspot is matched to the latest 3DEP product collected in 2010-2023; the lidar can be up to 13 years from some checklists. Filter `collection_year_AWS` for a tighter match.
- 5,673 of 53,361 hotspots (11%) have no usable lidar (no 3DEP footprint, or only products outside 2010-2023).
- Encounter rates are 0 where the species is outside its eBird range map (SatBird's correction). Labels reflect birding effort and are noisy for rare species at hotspots with few checklists.
- Source CRS of the coordinates is not stated by SatBird (eBird lat/lon); registry marks it `assumed`.
- Nearby hotspots are common; SatBird's cluster split keeps hotspots within 5 km on one side, but hotspots further apart can still be close.
- `location_occupancy.parquet` is registered as `satbird_winter` and is in `squares.EVAL_SITES`. The cookie cache is not built; 47,688 cookies is above the ~2,000 suggested for an in-training task, so subsample before adding an `eval_tasks` entry.

Figures: `figures/01_map_split.png`, `02_species_hotspots_per_split.png`.
