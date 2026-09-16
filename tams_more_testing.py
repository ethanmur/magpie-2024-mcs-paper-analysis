#%%

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import shapely
import pandas as pd
import geopandas as gpd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from synoptic_overview import calc_latlon, get_xy_from_latlon

path_project = Path(__file__).parent.parent


#%%
# load data here
path_tb   = str(path_project) + "/data/tams.nc"
goes = xr.open_dataset(path_tb)

path_cs   = str(path_project) + "/data/tams_tracked.parquet"
tams = gpd.read_parquet(path_cs)



# %%
# pull data just for polygons of interest
tams_782 = tams[tams['mcs_id'] == 782]

geom = tams_782.iloc[0].geometry

gs = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs('EPSG:32663')
area_km2 = gs.area.values[0] / 1e6

# %%
