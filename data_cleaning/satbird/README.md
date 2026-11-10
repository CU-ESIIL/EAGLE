# SatBird (USA winter, summer pending): eBird hotspot encounter rates

SatBird (Teng et al. 2023, NeurIPS D&B, [arXiv 2311.00936](https://arxiv.org/abs/2311.00936)) labels eBird hotspots in the USA with the encounter rate of 670 bird species: the share of the hotspot's complete checklists of one season (USA winter: December-January; USA summer: June-July; 2010-2023) that report the species. We use **only the original dataset's own data**: hotspot coordinates, labels and train/valid/test split all come from SatBird. Nothing is joined from eBird, and no split is made here.

## Raw data

`datasets/raw/satbird/SatBird-labels/<season>/` (gitignored), from [huggingface.co/datasets/sammlapp/SatBird-labels](https://huggingface.co/datasets/sammlapp/SatBird-labels) (CC BY-NC 4.0), which repackages the non-image files of SatBird's Google Drive archives unmodified. Provenance, sizes and sha256: `catalog/download_log.csv` and `SOURCE.md` there. Per season:

- `hotspots.parquet`: one row per hotspot: `hotspot_id` (eBird locId), `lat`, `lon`, county/state, checklist counts, bioclim/soil values, SatBird's `split` (`train`/`valid`/`test`) and `probs` (the 670 encounter rates, in the order of `species_list.txt`).
- `species_list.txt` (scientific names), SatBird's `train/valid/test_split.csv`.

`USA_winter`: 53,361 hotspots (38,241 train / 8,323 valid / 6,797 test). `USA_summer` is not in the repo yet; when it is, run the same scripts with `USA_summer`.

## Run order

`pixi run python data_cleaning/satbird/<script> [USA_winter|USA_summer]` from the repo root (about 1 min).

| script | does |
|---|---|
| `01_download.py` | `hf download` of the label files (or `--source` an existing snapshot), download log |
| `02_filter_and_match_als.py` | valid coordinates, match to a 3DEP product collected in 2010-2023 (the label years), gap counts |
| `03_export.py` | tables with SatBird's split, species summary, figures |

## Cleaning rules (USA winter)

`catalog/filter_counts_USA_winter.csv`:

| step | hotspots | dropped |
|---|---|---|
| SatBird hotspots | 53,361 | |
| valid coordinates | 53,361 | 0 |
| covered by 3DEP with a collection year | 48,008 | 5,353 |
| a covering product collected inside 2010-2023 | 47,688 | 320 |

Final split: **34,327 train / 7,174 valid / 6,187 test**. Gap table (`catalog/year_gap_counts_USA_winter.csv`, per split): 47,688 / 47,769 / 47,783 / 47,871 / 48,005 / 48,005 hotspots for N = 0..5. The rule barely matters, so the eval table uses N = 0.

No hotspot is dropped for coordinates, precision or checklist count: SatBird already required enough complete checklists, and eBird hotspot coordinates are map-placed. The only drops are missing lidar.

## Split

SatBird's own, as published (hotspots clustered with DBSCAN at 5 km, clusters assigned at random 70/15/15; winter hotspots that are also in the summer set keep their summer split). It is kept in `split` (`train`, `val`, `test`) and as bool `test_split` (True = test; val is False). The benchmark task uses train and test and ignores val. Because the lidar filter removes hotspots unevenly, the test share is 13%, not the repo default of 50%; this is deliberate (the published split is kept, per the ingestion protocol).

## Lidar matching rule (pooled labels)

The labels pool one season's checklists over **2010-2023**, so a hotspot has no single survey date and the usual "lidar within 3 years" rule does not apply:

- A 3DEP product is a **valid match if its collection year is anywhere in 2010-2023** (`LABEL_YEARS` in stage 2).
- If several products cover the hotspot, the **latest collection year inside the window is used**.
- `year_diff_AWS` = 0 when inside the window, else years to the nearest edge. Only 0 is kept.
- `year` = 2022 is a reference only (the Sentinel-2 year of the original dataset). `label_year_min` / `label_year_max` are stored in every row.
- **Consequence:** the lidar can be up to 13 years from some of the checklists. Labels are time-averaged, so they describe long-run bird communities; habitat change between checklists and the lidar adds noise. Filter `collection_year_AWS` for a tighter match.

## Outputs

`datasets/SATBIRD_USA_winter_eval/` (`SATBIRD_USA_summer_eval/` later): `location_encounter_rate.parquet`, `location_occupancy.parquet` and `location_species_summary.csv`. See the README there.
