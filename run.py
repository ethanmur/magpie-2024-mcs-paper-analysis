#%%
import os
from pathlib import Path
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd 
import regionmask
import shapely
import xarray as xr

# import local code functions to create figures
path_project = Path(__file__).parent.parent
path_code = str(path_project) + "/code/"
sys.path.append(path_code)
import synoptic_overview
import dropsonde_profiles
import tdr_analysis
import supplemental_calcs
import tdr_analysis_new_cross
import kinematic_partitions

 #%%
# make the plots here!

# Figure 1
# intro overview: GOES + full P-3 flight track + dropsonde locations
dropsonde_profiles.goes_flight_overview()


#%%

# Figure 2
synoptic_overview.combined_figure()


#%%

# Figure 3
# Plot evolution of southern and eastern convection and development of the cold pool
# new, streamlined plot
synoptic_overview.combined_cold_pools_wide_ir_only()


#%%
# Figure 4
# provided by Jeff- no code to add

# Figure 5
# TAMS analysis
start_time = pd.Timestamp('2024-07-15 14:30:00')
supplemental_calcs.plot_tams_stats_nice(start_time=start_time)



#%%
# Figure 6
# plot dropsonde profiles ahead of and behind cold pool / convective complex. also add satellite view
# for context!
dropsonde_profiles.cold_pool_first_dropsondes_goes()


#%%
# Figure 7
# 3x3 TDR radar figure: reflectivity + winds at 2km and 6km, rainfall classification
tdr_analysis.tdr_radar_figure(cross_endpoints_1=None, cross_endpoints_2=None)


#%%

# Figure 8: TDR statistics
tdr_analysis.plot_tdr_statistics()

#%%
# Figure 9: kinematic/reflectivity CFADs partitioned by precip classification
# (weak echo + stratiform vs. shallow/moderate/deep convective), pooling all
# 3 flight passes, plus per-leg CFADs to see how properties evolve pass to pass
kinematic_partitions.complete_partitions_contour_nice()


#%%
# Figure 10: TDR curtain from 2nd pass
filename = '240715I1_1919_xy.nc'
ce = (10.6, -54.3, 10.25, -53.6)
out_name = 'figure10.nc'
tdr_analysis.plot_tdr_curtains(convective_axis_deg=20, add_barbs=True, add_vort=True, x_axis='distance', cross_endpoints=ce, passind=0)


#%%
# Figure 11: skew Ts and fl distributions
dropsonde_profiles.composite_plots()
kinematic_partitions.complete_flightlevel_partitions_nice()

#%%
# Figure 12: sal + buffer zone time series, distribitions, spatial plots, etc
kinematic_partitions.flightlevel_buffer_sal_dists()
kinematic_partitions.flightlevel_buffer_sal_timeseries(t_start=19.25, t_end=20.0)
kinematic_partitions.flightlevel_dists_spatial()
dropsonde_profiles.sal()

#%%
# Figure 13
# convective mixing with convective downdrafts and buffer zone
dropsonde_profiles.convective_mixing()

#%%
# Figure 14
# goes + skew t plots of unfavorable outflow interactions with convection
dropsonde_profiles.outflow()






#%%
# # Supplemental calculations: Figure out size of cells, IR BT contours, etc
supplemental_calcs.calc_tdr_size()

supplemental_calcs.calc_tdr_mean_flow()


# supplemental_calcs.plot_dbz_area()          # 10 dBZ threshold → tdr_dbz_area_10dbz.png
# supplemental_calcs.plot_dbz_area(dbz_threshold=20)  # 20 dBZ → tdr_dbz_area_20dbz.png

