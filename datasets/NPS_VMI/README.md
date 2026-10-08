# NPS Vegetation Mapping Inventory (VMI) plot data

Vegetation classification and accuracy-assessment (AA) plots from the NPS VMI program, harmonized from the open-format data packages in IRMA DataStore (https://www.nps.gov/im/vmi-products.htm). Raw downloads live in `datasets/raw/nps_vmi/` (gitignored). Code: `data_cleaning/nps_vmi/` (run `01_` to `06_` in order). **Read `data_cleaning/nps_vmi/DATASET_CONCERNS.md` before interpreting any park.**

## Files

| file | rows | contents |
|---|---|---|
| `vmi_core_events.parquet` | ~76k | one row per plot / accuracy-assessment (AA) point visit, all harmonized families. ALS-matching columns come first (see below), then NVC resolution, then raw coordinates and `qa_flags` |
| `vmi_core_species.parquet` | ~700k | species x stratum records: event_key, species_name, plants_symbol, tsn, stratum, cover_code, `cover_pct`, `cover_pct_source`, qa_flags (plots only; the v3 AA points have no species lists) |
| `vmi_plot_events.parquet`, `vmi_plot_species.parquet` | | richer PLOTS v3/v4-schema-only tables (all original tPlots/tPlotEvents columns incl. ground cover, strata heights) |

Join species to events with `event_key`.

## Key caveats (details in DATASET_CONCERNS.md)

- Cover is a percent where reported, otherwise a class midpoint; always check `cover_pct_source`.
- Coordinates were repaired where zone, X/Y order or longitude sign were wrong; see `qa_flags`.
- Not harmonized yet: 12 custom-schema packages (9 Alaska parks, BIHO, CIRO, SCPN/WUPA). ~38 parks have no open-format data package.

## Columns for matching to ALS (`vmi_core_events.parquet`)

| column | meaning |
|---|---|
| `community_name` | NVC community name, or null. Scientific name of the NVC unit that `nvc_code` resolves to in the current USNVC catalog (`datasets/vegbank/USNVC_catalog_full_taxonomy.csv`); if the code does not resolve, the source's own classified name when it gave one |
| `freeform_community_name` | the field crew's provisional community description, as written (not NVC, many parks) |
| `year` | survey year (null if missing or implausible) |
| `lat_wgs84`, `lon_wgs84` | decimal degrees; reprojected from NAD83/WGS84 UTM or taken from lat/lon fields. Repairs are listed in `qa_flags` |
| `location_accuracy_flag` | `as_recorded`, `repaired` (zone/axis/sign fixed), `suspect_outside_park` (more than 0.25 deg from the IRMA park box) or `missing` (null coordinates). It does **not** express GPS precision (raw GPS error is in the source tables) |
| `nvc_code`, `nvc_code_source` | the assigned code and where it came from: `final_classification`, `aa_field_call` (AA mapper's first call, lower confidence) or `name_match_to_catalog` |
| `nvc_resolved`, `nvc_level`, `nvc_common_name`, `nvc_macrogroup`, `nvc_group`, `nvc_division` | catalog resolution and roll-up to coarser NVC levels; null when `nvc_resolved` is false (park-local, provisional or retired codes) |
| `community_name_source` | `nvc_catalog`, `nvc_catalog(aa_field_call)` or `source_classified_name` |

Typical filter for labeled, trustworthy points: `community_name.notna() & location_accuracy_flag.isin(['as_recorded','repaired'])`. Add `nvc_code_source != 'aa_field_call'` to drop the lower-confidence AA calls, and use `nvc_macrogroup` / `nvc_group` for coarse classes.
