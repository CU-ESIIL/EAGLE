# SatBird USA-summer evaluation set

One row per eBird hotspot (48,048 hotspots, 1,320 lidar products) with the encounter rate of 670 candidate bird species and the 3DEP lidar product closest collected in 2010-2023 (the years the labels cover; rule below). Built by `data_cleaning/satbird/` (see its README for sources, filters and counts).

## The four variants

| | encounter rate (0-1) | occupancy (0/1) |
|---|---|---|
| **all**: species present at >= 1 hotspot in each of train and test (618 species, 279 songbirds) | `SATBIRD_eval/location_encounter_rate.parquet` (`enc__spNNN`) | `SATBIRD_eval/location_occupancy.parquet` (`occ__spNNN`) |
| **-20**: >= 20 hotspots in each of train and test (537 species, 253 songbirds) | `SATBIRD_eval-20/location_encounter_rate.parquet` | `SATBIRD_eval-20/location_occupancy.parquet` |

All four share the same rows and split. Occupancy = encounter rate > 0 (reported on at least one complete checklist). `location_species_summary.csv` lists all 670 targets with hotspots present in train/test, prevalence, mean rate, `is_songbird` and `included`. Recommended: score the "-20" sets, with the "all" sets for completeness; use mean AUROC / average precision per species for occupancy and per-species correlation or MAE for encounter rate.

## Columns

`hotspot_id` (eBird locId, unique), `hotspot_name`, `lat`, `lon` (WGS84, eBird's current coordinates), `year` (2022 = paired Sentinel-2 year, a reference only), `label_year_min`, `label_year_max` (2010, 2023: the pooled checklist years), `state_code`, `county_code`, `n_checklists_all_time` (eBird, all seasons), `shift_m_vs_2023` (distance to SatBird's 2023 coordinates where known, else null), `product_name_AWS`, `collection_year_AWS`, `year_diff_AWS` (always 0: collected inside 2010-2023), `als_site_id`, `block` (1-degree cell), `test_split` (True = test, 30%), then the species columns.

## Caveats

- **Species have no names.** Names ship only in the 152 GB Drive archive. Columns are `spNNN`, the index in SatBird's 670-vector; map them via the Drive's species list if needed.
- **Pooled labels and the lidar rule.** Each encounter rate pools complete June-July checklists from **2010 to 2023**, so there is no survey date. A hotspot is matched to the **latest 3DEP product whose collection year is in 2010-2023** (any year in the range counts as a match; `year_diff_AWS` = 0 for all rows). `collection_year_AWS` therefore runs 2010-2023 (median 2018), and the lidar can be up to 13 years from some of the checklists. Labels are time-averaged: they describe the long-run bird community at the site, and habitat change between checklists and the lidar adds noise. For a tighter match, filter `collection_year_AWS` (e.g. >= 2018).
- **Only 39% of SatBird's hotspots remain** (55,399 of 122,593 are still public eBird hotspots; 48,048 of them have lidar inside the label years). This is a subset of SatBird, and scores are not comparable to published SatBird numbers (also because the split is our own 1-degree block split).
- Coordinates are today's eBird values; 1,257 hotspots that moved > 100 m were dropped, but about 14,000 could not be checked.
- Nearby hotspots are common (parks, lakes); the block split keeps neighbours on the same side.
- Labels reflect birding effort and are noisy for rare species at hotspots with few checklists (min. 55 complete checklists).
- `datasets/SATBIRD_eval/location_occupancy.parquet` is in `squares.EVAL_SITES`. The cookie cache is not built yet; 48,048 cookies is above the ~2,000 suggested for an in-training task, so subsample before adding an `eval_tasks` entry.

Figures: `figures/01_map_split.png`, `02_species_hotspots_per_split.png`.
