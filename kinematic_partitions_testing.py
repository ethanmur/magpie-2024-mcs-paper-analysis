#%%
"""
Spatial sanity check for the flight-level <-> TDR classification matching
used in kinematic_partitions.complete_flightlevel_partitions().

For each pass: plot the TDR classification field, the full flight track
within that pass's time window, and which flight points actually matched a
classification. Diagnostic only — nothing is saved.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import kinematic_partitions as kp

class_ds = xr.open_dataset(kp.CLASS_PATH + kp.CLASS_FILE)
fl = xr.open_dataset(kp.FLIGHT_LEVEL_PATH + kp.FLIGHT_LEVEL_FILE, decode_times=False)
fl_lat  = fl['LATref'].values
fl_lon  = fl['LONref'].values
fl_time = kp.build_time_array(fl, axis_type='decimal')


def check_pass(tdr_file, window):
    """Plot classification field + full flight track + matched points for one pass."""
    pd_ = kp._load_pass(tdr_file, class_ds)

    t0, t1 = window
    seg = (fl_time >= t0) & (fl_time <= t1)
    codes = kp._nearest_classification(fl_lat[seg], fl_lon[seg], pd_)
    matched = np.isfinite(codes)

    plt.figure(figsize=(6, 6))
    plt.pcolormesh(pd_['lon'], pd_['lat'], pd_['class'].transpose(), cmap='viridis',
                   vmin=-0.5, vmax=4.5, shading='auto')
    plt.plot(fl_lon[seg], fl_lat[seg], color='k', lw=1, label='full flight track')
    plt.scatter(fl_lon[seg][matched], fl_lat[seg][matched], color='red', s=8,
               label='matched to classification')
    plt.title(f'{tdr_file}  window={window}  matched={matched.sum()}/{seg.sum()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='upper right', fontsize=8)


#%% Pass 1
check_pass(kp.PASS_FILES[0], kp.PASS_TIME_WINDOWS[0])

#%% Pass 2
check_pass(kp.PASS_FILES[1], kp.PASS_TIME_WINDOWS[1])

#%% Pass 3
check_pass(kp.PASS_FILES[2], kp.PASS_TIME_WINDOWS[2])

# %%
