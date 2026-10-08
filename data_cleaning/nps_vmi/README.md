# NPS Vegetation Mapping Inventory (VMI): crawl and harmonization pipeline

Scripts that find, download and harmonize the vegetation plot data behind https://www.nps.gov/im/vmi-products.htm. Outputs land in [datasets/NPS_VMI/](../../datasets/NPS_VMI/README.md); read [DATASET_CONCERNS.md](DATASET_CONCERNS.md) before interpreting any park's data.

## How the data is found

Park pages and IRMA DataStore profile pages are JavaScript-rendered, so the pipeline uses the IRMA REST API instead of scraping:

park list page -> park page (`vmi-xxxx.htm`, contains the IRMA project `ReferenceId`) -> `https://irmaservices.nps.gov/datastore/v8/rest/Profile/<id>` -> project's `products` -> the "Field data for the Vegetation Mapping Inventory Project of ... - Open Format Data Package" -> its CSV files (`https://irma.nps.gov/DataStore/DownloadFile/<fileId>`).

## Run order

Run from the repo root with `pixi run python data_cleaning/nps_vmi/<script>`. Every step is rerunnable; downloads and API responses are cached in `datasets/raw/nps_vmi/` (gitignored, ~400 MB).

| script | does | writes |
|---|---|---|
| `01_catalog.py` | park list -> projects -> products -> data-package file lists; logs every HTTP access | `catalog/parks.csv`, `products.csv`, `files.csv`, `access_log.csv` |
| `02_download.py` | downloads the data-package CSVs (skips Access `MSys*` tables and the shared `xPLANTS_lu`) | `datasets/raw/nps_vmi/<UNIT>_<id>/`, `catalog/download_log.csv` |
| `03_profile.py` | row counts and columns of every downloaded table | `catalog/package_profile.csv` |
| `04_harmonize_v4.py` | PLOTS v3/v4-schema packages -> full-detail tables | `datasets/NPS_VMI/vmi_plot_events.parquet`, `vmi_plot_species.parquet` |
| `05_build_core.py` | stacks all three schema families into one core schema | `datasets/NPS_VMI/vmi_core_events.parquet`, `vmi_core_species.parquet`, `catalog/core_build_notes.csv` |
| `06_status_and_concerns.py` | per-park access/harmonization status and generated concerns | `catalog/package_status.csv`, `DATASET_CONCERNS.md` |

Stage 5 also resolves community labels against `datasets/vegbank/USNVC_catalog_full_taxonomy.csv` (see the column table in the data README). Label sources: PLOTS `NVC_Elcode`/`Classified_Code`/`Primary_Code`, NER `NVC.ELCODE` + classified names, NCPN `tblFinalAssociationNames*`, `tblFinalClassification`, `tbl_Final_Classification`, `tblVegetation.CEGLCode`/`FinalNVCName`/`Associations`, `tblAAVegetation.Associations`. Not yet used: the 12 non-harmonized custom packages, and free-text `freeform_community_name` is not matched to NVC.

`coords.py` is shared by stages 4 and 5 (coordinate repair).

## Catalog files

- `package_status.csv`: one row per park page. `status` is `harmonized`, `downloaded, NOT yet harmonized (custom schema)`, `no open-format data package (only: ...)`, `park page not available (404)`, `park page has no IRMA project link` or `IRMA project returned no products`. Also has event and species counts, year range and georeferencing counts. This is the table of which data products were accessible.
- `access_log.csv`: every HTTP request in stage 1 with its status (a few park pages 404 and some profile calls returned 500).
- `download_log.csv`: result of every file download. `core_build_notes.csv`: things stage 5 skipped or dropped.

## Schema families harmonized

| family (`schema_family`) | tables used | parks |
|---|---|---|
| `plots_v4` | `tPlots`, `tPlotEvents`, `tPlotEventSpecies`, `tSpecies` (PLOTS v3.x/v4, sometimes with prefixed file names) | 71 |
| `ncpn_tbl` | `tblPlotLocation`, `tblPlotDetails`, `tblVegetation(Details)` and the `tblAA*` equivalents; species by TSN | 14 |
| `ner_plots` | `Plots`, `Plots-Species`, `AA_Observations`, `AA-Species` | 6 |
| `plots_v3_aa` | `tAA` / `tAAEvents` accuracy-assessment points in PLOTS v3-schema packages (events only, no species; 2 packages with a differently shaped `tAAEvents`, APPA and NATR, are skipped) | 46 |

Not harmonized yet (12 packages, downloaded and profiled): Alaska parks (ALAG, ANIA, DENA, GAAR, KEFJ, KLGO, SITK, WRST, YUCH), BIHO and CIRO (`aadata`/`plotdata` tables), SCPN/WUPA.

## `qa_flags` meanings (events)

| flag | meaning |
|---|---|
| `utm_zone_inferred` | zone missing in source; chose the zone that lands the point inside the park/project bounding box |
| `utm_zone_corrected` | zone label put the point outside the bounding box; a neighboring zone fits |
| `utm_xy_swapped` | northing was stored in the X column |
| `latlon_swapped_lon_sign_assumed`, `lon_sign_assumed` | lat/lon in (lat, lon) order or with the western longitude sign lost |
| `ll_but_metadata_says_utm` | coordinates are lat/lon though project metadata says UTM (informational) |
| `outside_park_bbox` | still more than 0.25 degrees from the IRMA bounding box (e.g. bundled sub-sites such as BEOL's `SAND.*`) |
| `no_coordinates` / `utm_zone_missing` | no usable location; lon/lat are null |
| `implausible_date_nulled` | survey year outside 1975-2025 (source typo); date set to null |

Species rows carry `cover_pct_source`: `reported` (percent in source), `reported(Total_Plot_Cover)`, `class_midpoint` (park lookup table), `class_midpoint_parsed_from_range` (class code was itself a range such as `1-5`), `class_midpoint_assumed_ncpn_scheme` (park lookup had no percentages; standard NCPN scheme assumed), or null (no cover recorded, mostly presence-only accuracy-assessment points).

## Requirements

`pandas`, `pyarrow`, `pyproj`, `shapely` and `tabulate` (all in the pixi environment); network access for stages 1-2 only.
