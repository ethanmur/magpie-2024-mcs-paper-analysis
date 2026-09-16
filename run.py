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
synoptic_overview.combined_figure()


#%%

# Figure 2
# Plot evolution of southern and eastern convection and development of the cold pool
synoptic_overview.combined_cold_pools()



#%%
# Figure 3
# plot dropsonde profiles ahead of and behind cold pool / convective complex. also add satellite view
# for context!
dropsonde_profiles.cold_pool_first_dropsondes_goes()

#%%
# change cross section endpoints here!
# original
ce=(10.55, -54.3, 10.15, -53.7)
# nudged north
ce=(10.65, -54.3, 10.25, -53.7)

# use two different cross sections for different passes!
ce1=(10.65, -54.3, 10.25, -53.7)
ce1=(10.6, -54.3, 10.25, -53.6)

ce2=(11.1, -54.5, 10.7, -53.8)
ce2=(11.3, -54.15, 10.8, -53.5)
ce2=(11.3, -54.2, 10.85, -53.3)



#%%
# Figure 4
# 3x3 TDR radar figure: reflectivity + winds at 2km and 6km, rainfall classification
# tdr_analysis.tdr_radar_figure(cross_endpoints_1=ce1, cross_endpoints_2=ce2)
tdr_analysis.tdr_radar_figure(cross_endpoints_1=None, cross_endpoints_2=None)


#%%

# Figure 4.5: TDR statistics
tdr_analysis.plot_tdr_statistics()


#%%

# # Figures 5 and 6: TDR curtains from 2nd and 3rd passes
# tdr_analysis.plot_tdr_curtains()
# tdr_analysis.plot_tdr_curtains(convective_axis_deg=20)

# build cross section datasets here
#passind=0
for passind in [0,1]:
    if passind == 0:
        filename = '240715I1_1919_xy.nc'
        ce = ce1
    if passind == 1:
        filename = '240715I1_2030_xy.nc'
        ce = ce2
    out_name = f'tdr_curtain_2160_across_pass_{passind}.nc'

    # tdr_analysis_new_cross.build_cross_curtain(lat1=ce[0], lon1=ce[1], lat2=ce[2], lon2=ce[3], passind=passind, tdr_file=filename)


    # figure 5.5: TDR curtain across the axis, rather than along
    tdr_analysis_new_cross.plot_tdr_curtains_across(convective_axis_deg=20, add_barbs=True, add_vort=True, x_axis='distance', 
                                                    cross_endpoints=ce, cross_name=out_name, passind=passind)



    # tdr_analysis_new_cross.plot_tdr_curtains_across(convective_axis_deg=None, add_barbs=False, add_vort=True, x_axis='distance', 
    #                                                 cross_endpoints=ce, cross_name=out_name, passind=passind)

    tdr_analysis.plot_tdr_curtains(convective_axis_deg=20, add_barbs=True, add_vort=True, x_axis='distance', cross_endpoints=ce, passind=passind)

    # tdr_analysis.plot_tdr_curtains(convective_axis_deg=None, add_barbs=False, add_vort=True, x_axis='distance', cross_endpoints=ce, passind=passind)




#%%
# Figure 6.5: kinematic/reflectivity CFADs partitioned by precip classification
# (weak echo + stratiform vs. shallow/moderate/deep convective), pooling all
# 3 flight passes, plus per-leg CFADs to see how properties evolve pass to pass

#kinematic_partitions.complete_partitions(convective_axis_deg=20, per_leg=True, wind_units='m/s')

kinematic_partitions.complete_partitions_contour(wind_units='m/s', per_leg=True, convective_axis_deg=20)

#%%

kinematic_partitions.complete_flightlevel_partitions(match_method='manual', resolution='upscale')
kinematic_partitions.complete_flightlevel_partitions(match_method='curtain', resolution='upscale')

kinematic_partitions.complete_flightlevel_partitions_nice()



#%%

# Figure 7
# Composite Skew-T profiles: stratiform vs convective dropsondes
dropsonde_profiles.composite_plots()

# Figure 8
# GOES + flight-level RH | Skew-T sonde 18 | SAL composite
dropsonde_profiles.sal()

# Figure 9:
# convective mixing with outer environment
dropsonde_profiles.convective_mixing()

# Figure 10
# goes + skew t plots of unfavorable outflow interactions with convection
dropsonde_profiles.outflow()


#%%
# # Supplemental calculations: Figure out size of cells, IR BT contours, etc
supplemental_calcs.calc_tdr_size()

supplemental_calcs.calc_tdr_mean_flow()


# supplemental_calcs.plot_dbz_area()          # 10 dBZ threshold → tdr_dbz_area_10dbz.png
# supplemental_calcs.plot_dbz_area(dbz_threshold=20)  # 20 dBZ → tdr_dbz_area_20dbz.png




#%%
# # IR data analysis

# manual definition: IR cold cloud-top coverage at TDR pass times (3 subplots)
# kinda clunky but effective visuals tbh!
# supplemental_calcs.plot_ir_bt_edges()

# # plot tams outlines for my case!

# original function: tams outlines for 3 TDR passes, as above. use standard bt thresholds
# supplemental_calcs.find_tams_info(ctt_threshold_c=-35.0, ctt_core_threshold_c=-54.0)

# automatic analysis pipeline
# run tracking algorithm
#supplemental_calcs.load_goes_to_nc()        # slow — loads 43 GOES files, saves tams.nc

supplemental_calcs.run_tams_tracking()


 # %%
supplemental_calcs.find_tams_synoptic(plot_type='zoom')
#supplemental_calcs.find_tams_synoptic(plot_type='mid')




# %%
# single MCS track + stats
supplemental_calcs.plot_single_mcs_track(mcs_id=782, include_companions=False)



# %%
# supplemental figure here
start_time = pd.Timestamp('2024-07-15 14:30:00')
supplemental_calcs.plot_tams_stats_nice(start_time=start_time)
# %%
