import pandas as pd
import sys
sys.path.append('../../src/streaming/3dep/')
sys.path.append('../../src')
from eagle_als.sites import add_site_ids
from eagle_als.splits import spatial_split
from get_als import get_nearest_year_product
import numpy as np

plots = pd.read_parquet('../../datasets/NPS_VMI/vmi_core_events.parquet')
plots.rename(columns={'lat_wgs84':'latitude','lon_wgs84':'longitude'}, inplace=True)
print("Filtering plots with community name and location accuracy as recorded or repaired.")
filtered_plots = plots[plots.community_name.notna() & plots.location_accuracy_flag.isin(['as_recorded','repaired'])]
print(f"Filtered from {len(plots)} to {len(filtered_plots)} ({len(filtered_plots)/len(plots):.2f})")

def find(row):
    try:
        result = get_nearest_year_product(row['latitude'], row['longitude'],int(row['year']) )
        assert result is not None, "No product found"
        return result
    except Exception as e:
        return {'product_name_AWS': None, 'collection_year_AWS': None, 'year_diff_AWS': None}

matched_als = filtered_plots.apply(find, axis=1, result_type='expand')
print(f"Matched {matched_als.product_name_AWS.notna().sum()} plots to ALS products.")
filtered_plots = filtered_plots.join(matched_als)

matched = filtered_plots[filtered_plots['year_diff_AWS']<=3]
len(matched), len(matched)/len(filtered_plots)

# select cols
matched=matched[['event_key', 'package', 'unit_code', 'plot_code',
        'latitude', 'longitude', 'elevation_m',
        'nvc_macrogroup','nvc_group', 'nvc_division', 'community_name', 'freeform_community_name',
       'year', 'event_date', 'product_name_AWS', 'collection_year_AWS','year_diff_AWS',
       'plot_type', 'plot_shape', 'plot_area_m2']]

# blocked train/test split (eagle_als.splits: 0.25 degree blocks, ~50% test, stable)

# note: splitting results in some Divisions only in test split (6) or only in train split (1)
matched['test_split'] = spatial_split(matched, 'latitude', 'longitude')[0].to_numpy()
class_cnts = matched.groupby(['test_split','nvc_division']).size().unstack(level=0,fill_value=0).sort_index().rename(columns={False:'train',True:'test'})

matched = add_site_ids(matched, 'latitude', 'longitude')  # cookie id (key in the cookie cache)
matched.to_parquet('../../datasets/NPS_VMI/vmi_classification_task.parquet', index=False)