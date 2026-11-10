# SatBird (USA summer): eBird hotspot encounter rates

SatBird (Teng et al. 2023, NeurIPS D&B, [arXiv 2311.00936](https://arxiv.org/abs/2311.00936)) labels 122,593 eBird hotspots in the USA with the encounter rate of 670 bird species: the share of complete June-July checklists (2010-2023) at the hotspot that report the species. The raw dataset (Sentinel-2 images, split CSVs with coordinates) is a single 154.7 GB `USA_summer.tar.gz` on [Google Drive](https://drive.google.com/drive/folders/1sm_VS2HtmwtIer3fqXpOs-WVv8e3bPIu); we do not need the images, so it is **not** downloaded.

## Raw data

`datasets/raw/satbird/` (gitignored). Provenance, sizes and sha256: `catalog/download_log.csv` and `SOURCE.md` there.

- `satbird_usa_summer_targets.pkl`, `songbird_indices.npy`, `nonsongbird_indices.npy`: from [huggingface.co/datasets/cisosdm/SatBird](https://huggingface.co/datasets/cisosdm/SatBird) (CC BY-NC 4.0). The pickle is `{hotspot_id: float vector of 670}` and has **no coordinates and no species names**.
- `ebird_hotspots/US-<state>.json`: eBird API v2 `ref/hotspot/US-<state>`, one bulk call per state and DC (51 calls; needs a free `EBIRD_API_KEY`). This is the only source of coordinates.
- `companion_splits/` (optional, cross-check only): split CSVs of two HF datasets by the same group (`cisosdm/SatBirdxsPlotOpen`, `.../SatBirdxSatButterfly`), which carry SatBird's 2023 hotspot coordinates for 87,939 hotspots. Used only to drop hotspots whose eBird coordinates moved; never as a coordinate source. Delete the folder and stage 2 skips the check.

## Run order

`pixi run python data_cleaning/satbird/<script>` from the repo root (about 1 min in total).

| script | does |
|---|---|
| `01_download.py` | targets from Hugging Face, hotspots from the eBird API, download log |
| `02_filter_and_match_als.py` | join coordinates, filter, match to a 3DEP product collected in 2010-2023 (the label years), gap counts |
| `03_split_and_export.py` | 1-degree block split (30% test), four target variants, figures |

## Cleaning rules

`catalog/filter_counts.csv`:

| step | hotspots | dropped |
|---|---|---|
| SatBird targets | 122,593 | |
| still a public eBird hotspot | 55,399 | 67,194 |
| valid coordinates | 55,399 | 0 |
| not moved > 100 m since 2023 (where checkable) | 54,142 | 1,257 |
| covered by 3DEP with a collection year | 48,256 | 5,886 |
| a covering product collected inside the label years 2010-2023 | 48,048 | 208 |

Gap table (`catalog/year_gap_counts.csv`): hotspots kept when the lidar may be up to N years outside 2010-2023, N = 0..5: **48,048** / 48,126 / 48,136 / 48,184 / 48,254 / 48,254. The rule barely matters beyond N = 0, so the eval tables use N = 0.

## Lidar matching rule (pooled labels)

SatBird labels pool the complete June-July checklists of **2010-2023**, so a hotspot has no single survey date and the usual "lidar within 3 years of the survey" rule does not apply. Instead:

- A 3DEP product is a **valid match if its collection year is anywhere in 2010-2023** (`LABEL_YEARS` in stage 2).
- If several products cover the hotspot, the **latest collection year inside the window is used** (newest lidar: closest to the most recent checklists and to the 2022 Sentinel-2 image). This is the preference rule; it is applied per hotspot and gives one product per row.
- `year_diff_AWS` = 0 when the product is inside the window; otherwise years to the nearest window edge. Only 0 is kept in the eval tables.
- `year` = 2022 is the paired Sentinel-2 year and is only a reference, not a survey date. `label_year_min` / `label_year_max` (2010 / 2023) are stored in every row.
- **Consequence:** the lidar can be up to 13 years apart from some of the checklists (collection years in the eval table run 2010-2023, median 2018). Labels are time-averaged, so they describe long-run bird communities at the site, not the state of the habitat on one date; forest change, fire or clearing between a checklist and the lidar adds label noise. Users who want a tighter match can filter `collection_year_AWS` themselves (e.g. >= 2018).

Why:

- **Public hotspot.** eBird returns no coordinates for the 67,194 SatBird ids that it has since merged, hidden or turned into personal locations (single-hotspot lookups return an empty body). They cannot be recovered without the Drive archive.
- **Moved.** eBird coordinates are today's. Of the 40,287 hotspots also in the 2023 CSVs, 1,257 moved more than 100 m (up to 23 km), so their labels describe a different place. 14,000 hotspots cannot be checked and are kept.
- **Year.** The labels pool 2010-2023, so there is no survey year; see "Lidar matching rule" above.
- No coordinate-precision filter: eBird hotspot coordinates are placed on a map and have 6-7 decimals. SatBird already dropped hotspots with fewer than 55 complete checklists.

## Outputs

`datasets/SATBIRD_eval/` ("all" species) and `datasets/SATBIRD_eval-20/` (species present at >= 20 hotspots in each of train and test), each with `location_encounter_rate.parquet` and `location_occupancy.parquet`. See the README in `datasets/SATBIRD_eval/`.
