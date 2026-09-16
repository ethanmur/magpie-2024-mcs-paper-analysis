#%%

import os
import sys
from pathlib import Path
from datetime import datetime
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import pandas as pd
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

# project paths (mirrors synoptic_overview.py)
path_project = Path(__file__).parent.parent
path_figures = str(path_project) + "/figures/"
path_data    = str(path_project) + "/data/"
path_code    = str(path_project) + "/code/"
sys.path.append(path_code)

import synoptic_overview


# ---------------------------------------------------------------------------
# Skew-T builder
# ---------------------------------------------------------------------------

def make_skewt(fig, subplot_spec, filepath, filename,
               panel_label=None, xlims=[15, 30], ylims=[1020, 680],
               add_leg=True):
    """
    Build a Skew-T Log-P diagram on the given GridSpec subplot position.

    Parameters
    ----------
    fig          : matplotlib Figure
    subplot_spec : GridSpec SubplotSpec (e.g. gs[0, 1])
    filepath     : directory containing the dropsonde NC file
    filename     : NC filename
    panel_label  : single character for the upper-left panel label, e.g. 'b'
    xlims        : [Tmin, Tmax] temperature axis limits (deg C)
    ylims        : [pmax, pmin] pressure axis limits (hPa)

    Returns
    -------
    skew : MetPy SkewT object (access axes via skew.ax)
    """
    label_fs = plt.rcParams['axes.labelsize']
    tick_fs  = plt.rcParams['xtick.labelsize']

    # --- load and coarsen data ------------------------------------------------
    data        = xr.open_dataset(filepath + filename)
    data_coarse = data.coarsen(time_since_launch=2, boundary='trim').mean()
    p  = data_coarse.pres
    T  = data_coarse.tdry
    Td = data_coarse.dp
    u  = data_coarse.u_wind
    v  = data_coarse.v_wind

    # --- create SkewT on the given GridSpec position --------------------------
    skew = SkewT(fig, rotation=45, subplot=subplot_spec)

    lw = 2.0
    skew.plot(p, T,  'r', label='T',      lw=lw)
    skew.plot(p, Td, 'b', label='$T_d$',  lw=lw)

    # wind barbs at log-spaced pressure levels
    interval = np.logspace(2, 3, 40) * units.hPa
    idx = mpcalc.resample_nn_1d(p * units.hPa, interval)
    skew.plot_barbs(pressure=p[idx], u=u[idx], v=v[idx])
    skew.ax.collections[-1].set_clip_on(False)

    skew.ax.set_ylim(ylims)
    skew.ax.set_xlim(xlims)

    # --- reference lines ------------------------------------------------------
    lw2 = 0.7
    skew.ax.axvline(0 * units.degC, linestyle='--', color='blue', alpha=0.3)
    skew.plot_dry_adiabats(lw=lw2,   alpha=0.3)
    skew.plot_moist_adiabats(lw=lw2, alpha=0.3)
    skew.plot_mixing_lines(lw=lw2,   alpha=0.3)

    # shaded isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90,  50, 8)
    for i in range(8):
        skew.shade_area(y=[1100, 50], x1=x1[i], x2=x2[i],
                        color='gray', alpha=0.075, zorder=1)

    # --- labels and ticks -----------------------------------------------------
    skew.ax.set_xlabel('Temperature (°C)', fontsize=label_fs)
    skew.ax.set_ylabel('Pressure (hPa)',        fontsize=label_fs)
    skew.ax.tick_params(axis='both', which='both', labelsize=tick_fs)
    if add_leg:
        skew.ax.legend(fancybox=False, shadow=False,
                    facecolor='w', edgecolor='k', framealpha=1,
                    loc='upper right',
                    fontsize=plt.rcParams['legend.fontsize'], ncol=2)

    # --- panel label (upper-left) ---------------------------------------------
    if panel_label is not None:
        skew.ax.text(0.05, 0.95, f'({panel_label})',
                     transform=skew.ax.transAxes, ha='left', va='top',
                     fontsize=label_fs,
                     bbox=dict(facecolor='white', alpha=1.,
                               edgecolor='k', linewidth=1.2, pad=4))

    return skew


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def cold_pool_first_dropsondes_goes():
    """
    Figure 3: 2 x 2 layout.
      (a) upper-left:  GOES-16 RGB + full P-3 flight track + aircraft symbol
      (b) upper-right: GOES-16 RGB + wind barbs + dropsonde-4/6 circles
      (c) lower-left:  Skew-T of dropsonde 4
      (d) lower-right: Skew-T of dropsonde 6

    Saves to figures/cold_pool_thermo.png
    """
    lons = (-58.5, -52.5)
    lats = (8,     13)

    # --- bump font sizes for this figure only (restored at the end) -----------
    _orig_rc = {k: plt.rcParams[k] for k in
                ['axes.labelsize', 'axes.titlesize', 'xtick.labelsize',
                 'ytick.labelsize', 'legend.fontsize']}
    plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 15,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 11})

    # --- paths ----------------------------------------------------------------
    path_goes  = str(path_project) + "/data/goes-level2-cold pool dropsondes/"
    path_drops = str(path_project) + "/data/dropsondes/"

    # --- load GOES data -------------------------------------------------------
    goes_files = synoptic_overview.pull_files_helper(path_goes)
    print(f'GOES dropsonde file: {goes_files[0]}')
    ds = synoptic_overview.data_subset(path_goes, goes_files[0], lats, lons)

    # --- identify dropsonde files (0-based index, sorted alphabetically) ------
    drop_files   = synoptic_overview.pull_files_helper(path_drops)
    sonde_4_file = drop_files[4]
    sonde_6_file = drop_files[6]
    print(f'Sonde 4: {sonde_4_file}')
    print(f'Sonde 6: {sonde_6_file}')

    # --- figure sizing --------------------------------------------------------
    asp_goes     = synoptic_overview._panel_aspect(lons, lats)
    goes_height  = 4.0              # inches
    goes_width   = goes_height * asp_goes
    skewt_size   = goes_height      # square-ish

    fig_width  = 2 * goes_width
    fig_height = goes_height + skewt_size

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[goes_width, goes_width],
                  height_ratios=[goes_height, skewt_size],
                  hspace=0.5, wspace=0.3)

    goes_end_time = ds.time_bounds.values[1]
    pc = ccrs.PlateCarree()

    # --- panel (a): GOES RGB + full flight track + aircraft symbol -----------
    ax_a = fig.add_subplot(gs[0, 0], projection=pc)
    # omit panel_label here — drawn manually at high zorder after all overlays
    synoptic_overview.test_plot(ds, channel='rgb', lats=lats, lons=lons,
                                ax=ax_a, fig=fig,
                                show_colorbar=False,
                                show_lat=True, show_lon=True,
                                show_title_banner=False,
                                show_time=False,
                                panel_label=None)
    # test_plot's show_lat branch sets ylabel but skips xlabel; add it manually
    ax_a.set_xlabel("Longitude ($\degree$)", fontsize=plt.rcParams['axes.labelsize'])

    # load flight-level data
    fl_path = str(path_project) + "/data/flight-level/20240715I1_A.nc"
    fl_data = xr.open_dataset(fl_path, decode_times=False)
    x_time  = build_time_array(fl_data, axis_type='decimal')
    x_time  = xr.DataArray(x_time, coords={'Time': x_time})
    fl_data = fl_data.assign_coords(Time=x_time)

    fl_time_dec = fl_data.Time.values
    lats_fl_all = fl_data['LATref'].values
    lons_fl_all = fl_data['LONref'].values
    valid = np.isfinite(lats_fl_all) & np.isfinite(lons_fl_all)

    # full flight track
    ax_a.plot(lons_fl_all[valid], lats_fl_all[valid],
              color='k', lw=1.5, transform=pc, zorder=5)

    # colored segments for radar passes (decimal UTC hours)
    radar_passes = [([17.8, 18.3], 'red'),
                    ([18.95, 19.45], 'blue'),
                    ([20.2, 20.65], 'w')]
    for (t0, t1), color in radar_passes:
        mask = (fl_time_dec >= t0) & (fl_time_dec <= t1) & valid
        if np.any(mask):
            ax_a.plot(lons_fl_all[mask], lats_fl_all[mask],
                      color=color, lw=2.5, transform=pc, zorder=6)

    # dropsonde dots for every sonde whose release location falls within domain
    drop_path  = str(path_project) + "/data/dropsondes/"
    drop_files_all = synoptic_overview.pull_files_helper(drop_path)
    for df in drop_files_all:
        try:
            d = xr.open_dataset(drop_path + df)
            ref_lon = float(d.reference_lon.values)
            ref_lat = float(d.reference_lat.values)
            if (lons[0] <= ref_lon <= lons[1] and lats[0] <= ref_lat <= lats[1]):
                ax_a.scatter(ref_lon, ref_lat, c='k', marker='o', s=25,
                             transform=pc, zorder=7)
        except Exception:
            pass

    # # aircraft symbol at GOES scan time — add_artist returns the object so we
    # # can set its zorder above everything else
    # synoptic_overview.overlay_flight_data(
    #     ax_a, goes_end_time, lons, lats,
    #     sonde_indices=None,
    #     fl_interval_min=999999,   # effectively skips wind barbs
    #     linec='k')
    # # bring the most-recently-added artist (the AnnotationBbox plane) to front
    # if ax_a.artists:
    #     ax_a.artists[-1].set_zorder(15)

    # panel label drawn last so it sits above all overlays
    label_fs = plt.rcParams['axes.labelsize']
    ax_a.text(0.05, 0.95, '(a)',
              transform=ax_a.transAxes, ha='left', va='top',
              fontsize=label_fs, zorder=20,
              bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                        linewidth=1.2, pad=4))

    # --- panel (b): GOES RGB + wind barbs + sonde circles --------------------
    ax_b = fig.add_subplot(gs[0, 1], projection=pc)
    synoptic_overview.test_plot(ds, channel='rgb', lats=lats, lons=lons,
                                ax=ax_b, fig=fig,
                                show_colorbar=False,
                                show_lat=False, show_lon=True,
                                show_title_banner=False,
                                show_time=True,
                                panel_label='b',
                                label_x=0.05, label_y=0.95,
                                time_x=0.95, time_y=0.95)
    # test_plot's show_lat branch sets ylabel but skips xlabel; add it manually
    ax_a.set_xlabel("Longitude ($\degree$)", fontsize=plt.rcParams['axes.labelsize'])

    synoptic_overview.overlay_flight_data(
        ax_b, goes_end_time, lons, lats,
        sonde_indices=[4, 6],
        sonde_labels=[4, 6],
        fl_interval_min=5,
        linec='k',
        drop_fs=13,
        label_color='white')

    # --- panel (c): Skew-T dropsonde 4 ---------------------------------------
    skew4 = make_skewt(fig, gs[1, 0], path_drops, sonde_4_file, panel_label='c', xlims=[15,29.95], ylims=[1020,680],
                       add_leg=False)
    skew4.ax.set_title('Dropsonde 4')

    # --- panel (d): Skew-T dropsonde 6 ---------------------------------------
    skew6 = make_skewt(fig, gs[1, 1], path_drops, sonde_6_file, panel_label='d', ylims=[1020,680],
                       add_leg=False)
    skew6.ax.set_title('Dropsonde 6')

    # --- save -----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'cold_pool_thermo.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight')
    plt.close(fig)
    plt.rcParams.update(_orig_rc)
    print(f'Saved {outfile}')



# ---------------------------------------------------------------------------
# Compositing helper: bin individual sondes on GPS altitude, return mean + all
# ---------------------------------------------------------------------------

def _composite_helper(data, sonde_indices, binsize=24.):
    """
    Bin dropsonde data from `all_sondes_20240715I1.nc` onto a shared GPS
    altitude grid and return both the composite mean and the individual
    binned profiles.

    Parameters
    ----------
    data          : xarray Dataset (all_sondes_20240715I1.nc)
    sonde_indices : list of int dropsonde indices (0-based)
    binsize       : altitude bin width in metres (default 24 m)

    Returns
    -------
    p_mean, T_mean, Td_mean, u_mean, v_mean : composite mean DataArrays
    p_all, T_all, Td_all : lists of individual binned DataArrays (one per sonde)
    """
    subset = data.isel(dropsonde=sonde_indices)

    alt_max = float(subset.gpsalt.max())
    altitude_bins = np.arange(0, binsize * np.ceil(alt_max / binsize) + 0.01, binsize)

    def bin_var(var):
        return var.groupby_bins(subset['gpsalt'], bins=altitude_bins).mean()

    # composite mean across all selected sondes
    p_mean  = bin_var(subset['pres'])
    T_mean  = bin_var(subset['tdry'])
    Td_mean = bin_var(subset['dp'])
    u_mean  = bin_var(subset['u_wind'])
    v_mean  = bin_var(subset['v_wind'])

    # individual sondes (for thin background lines)
    p_all, T_all, Td_all = [], [], []
    for si in sonde_indices:
        s = data.isel(dropsonde=si)
        p_all.append( s['pres'].groupby_bins( s['gpsalt'], bins=altitude_bins).mean())
        T_all.append( s['tdry'].groupby_bins( s['gpsalt'], bins=altitude_bins).mean())
        Td_all.append(s['dp'].groupby_bins(   s['gpsalt'], bins=altitude_bins).mean())

    return p_mean, T_mean, Td_mean, u_mean, v_mean, p_all, T_all, Td_all


# ---------------------------------------------------------------------------
# Skew-T builder for composites (accepts pre-loaded arrays)
# ---------------------------------------------------------------------------

def make_composite_skewt(fig, subplot_spec, p_mean, T_mean, Td_mean,
                         u_mean, v_mean, p_all, T_all, Td_all,
                         panel_label=None, title=None,
                         xlims=[-10, 35], ylims=[1020, 400],
                         add_leg=True):
    """
    Build a composite Skew-T on a GridSpec SubplotSpec.

    Plots individual sonde profiles as thin, semi-transparent lines behind
    the thick composite mean.

    Returns
    -------
    skew : MetPy SkewT object
    """
    label_fs = plt.rcParams['axes.labelsize']
    tick_fs  = plt.rcParams['xtick.labelsize']

    skew = SkewT(fig, rotation=45, subplot=subplot_spec)

    lw_ind  = 0.8    # individual profile linewidth
    alpha_ind = 0.30  # individual profile alpha
    lw_mean = 2.0    # composite mean linewidth

    # --- individual profiles (thin, semi-transparent) -------------------------
    for p_i, T_i, Td_i in zip(p_all, T_all, Td_all):
        skew.plot(p_i, T_i,  'r', lw=lw_ind, alpha=alpha_ind, zorder=1)
        skew.plot(p_i, Td_i, 'b', lw=lw_ind, alpha=alpha_ind, zorder=1)

    # --- composite mean (thick, opaque) ---------------------------------------
    skew.plot(p_mean, T_mean,  'r', lw=lw_mean, label='T',     zorder=3)
    skew.plot(p_mean, Td_mean, 'b', lw=lw_mean, label='$T_d$', zorder=3)

    # --- wind barbs at log-spaced pressure levels ----------------------------
    interval = np.logspace(2, 3, 40) * units.hPa
    idx = mpcalc.resample_nn_1d(p_mean * units.hPa, interval)
    skew.plot_barbs(pressure=p_mean[idx], u=u_mean[idx], v=v_mean[idx])
    skew.ax.collections[-1].set_clip_on(False)

    # --- axes limits and reference lines -------------------------------------
    skew.ax.set_ylim(ylims)
    skew.ax.set_xlim(xlims)

    lw2 = 0.7
    skew.ax.axvline(0 * units.degC, linestyle='--', color='blue', alpha=0.3)
    skew.plot_dry_adiabats(lw=lw2,   alpha=0.3)
    skew.plot_moist_adiabats(lw=lw2, alpha=0.3)
    skew.plot_mixing_lines(lw=lw2,   alpha=0.3)

    # shaded isotherms
    x1 = np.linspace(-100, 40, 8)
    x2 = np.linspace(-90,  50, 8)
    for i in range(8):
        skew.shade_area(y=[1100, 50], x1=x1[i], x2=x2[i],
                        color='gray', alpha=0.075, zorder=1)

    # --- labels ---------------------------------------------------------------
    skew.ax.set_xlabel('Temperature (°C)', fontsize=label_fs)
    skew.ax.set_ylabel('Pressure (hPa)',   fontsize=label_fs)
    skew.ax.tick_params(axis='both', which='both', labelsize=tick_fs)
    if add_leg:
        skew.ax.legend(fancybox=False, shadow=False,
                    facecolor='w', edgecolor='k', framealpha=1,
                    loc='upper right', fontsize=plt.rcParams['legend.fontsize'],
                    ncol=2)

    if title is not None:
        skew.ax.set_title(title, fontsize=label_fs)

    if panel_label is not None:
        skew.ax.text(0.05, 0.95, f'({panel_label})',
                     transform=skew.ax.transAxes, ha='left', va='top',
                     fontsize=label_fs,
                     bbox=dict(facecolor='white', alpha=1.,
                               edgecolor='k', linewidth=1.2, pad=4))
    return skew


# ---------------------------------------------------------------------------
# Figure 7: composite Skew-Ts for stratiform and convective dropsondes
# ---------------------------------------------------------------------------

# make composite plots for all stratiform and convective dropsondes, as IDed by the TDR!
# manually select cases of interest for ease of use here first
# stratiform: 8, 14, 15, 17, 22, 23, 25
# convective: 9, 16, 24
def composite_plots():
    """
    Figure 7: 1 x 2 Skew-T composite layout.
      (a) Stratiform dropsonde composite  (indices 8, 14, 15, 17, 22, 23, 25)
      (b) Convective dropsonde composite  (indices 9, 16, 24)

    Each panel shows individual sonde profiles (thin, semi-transparent) behind
    the composite mean (thick, opaque).  Saves to figures/composite_dropsondes.png
    """
    strat_indices = [8, 14, 15, 17, 22, 23, 25]
    conv_indices  = [9, 16, 24]

    data_file = path_data + "all_sondes_20240715I1.nc"
    data = xr.open_dataset(data_file)

    # --- composite arrays for each group -------------------------------------
    (p_s, T_s, Td_s, u_s, v_s,
     p_s_all, T_s_all, Td_s_all) = _composite_helper(data, strat_indices)

    (p_c, T_c, Td_c, u_c, v_c,
     p_c_all, T_c_all, Td_c_all) = _composite_helper(data, conv_indices)

    # --- figure sizing: two square-ish SkewT panels side by side -------------
    panel_size = 5.0   # inches (width = height for a SkewT)
    fig = plt.figure(figsize=(panel_size * 2 + 0.5, panel_size))

    gs = GridSpec(1, 2, figure=fig, wspace=0.45)

    make_composite_skewt(fig, gs[0, 0],
                         p_s, T_s, Td_s, u_s, v_s,
                         p_s_all, T_s_all, Td_s_all,
                         panel_label='a',
                         title='Stratiform',
                         xlims=[15,30],
                         ylims=[1013., 660.])

    make_composite_skewt(fig, gs[0, 1],
                         p_c, T_c, Td_c, u_c, v_c,
                         p_c_all, T_c_all, Td_c_all,
                         panel_label='b',
                         title='Convective',
                         add_leg=False,
                         xlims=[15,30],
                         ylims=[1013., 660.])

    # --- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'composite_dropsondes.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f'Saved {outfile}')



# convert from seconds since start of flight to either utc hours ('decimal') or actual datetime objects ('datetime').
# this code would need to be generalized for other cases! issues:
# 1) times that wrap past 24 UTC might lead to an error?
# 2) two seconds must be manually added to end of time series! this will likely vary per flight
# code taken from magpie_data_testing.py
def build_time_array(data, axis_type='decimal'):
    # figure out number of seconds from start to end of flight
    time_gap = data.TimeInterval
    str_start  = time_gap[:time_gap.find('-')]
    str_end = time_gap[time_gap.find('-')+1:]

    fmt = "%H:%M:%S"
    dt_start = datetime.strptime(str_start.strip(), fmt)
    dt_end = datetime.strptime(str_end.strip(), fmt)
    seconds_diff = int((dt_end - dt_start).total_seconds())

    if axis_type=='decimal':
        # create a new time array with hours as floats
        start_hour = dt_start.hour + dt_start.minute / 60 + dt_start.second / 3600
        # do +3, not +1, to account for extra 2 timesteps
        retval = np.round(start_hour + np.arange(seconds_diff + 3) / 3600, 5)
    elif axis_type=='datetime':
        # initialize a new time array with 1s intervals
        retval = pd.to_datetime(np.arange(seconds_diff + 3), unit='s', origin=dt_start)

    return retval


# ---------------------------------------------------------------------------
# Figure 8: GOES + flight-level RH | Skew-T sonde 18 | SAL composite
# ---------------------------------------------------------------------------

def sal():
    """
    Figure 8: 1 x 3 layout.
      (a) GOES-16 RGB (goes-level2-tdr index 1, ~19:00 UTC) overlaid with
          flight-level relative humidity scatter (18:30–19:30 UTC, ~2-min
          subsampling) and dropsonde-18 location marker.
          Colorbar for RH plotted to the right of panel (a).
      (b) Skew-T of dropsonde index 18 (from dropsondes/ folder).
      (c) Composite Skew-T of SAL dropsondes (indices 0,1,2,27,28,29,30).

    Saves to figures/figure_8.png
    """
    lons = (-55.5, -52.5)
    lats = (9,     12.5)

    path_goes  = str(path_project) + "/data/goes-level2-tdr/"
    path_drops = str(path_project) + "/data/dropsondes/"
    path_fl    = str(path_project) + "/data/flight-level/"
    fl_file    = "20240715I1_A_small.nc"

    # --- GOES data (index 1) -------------------------------------------------
    goes_files = synoptic_overview.pull_files_helper(path_goes)
    ds_goes    = synoptic_overview.data_subset(path_goes, goes_files[1], lats, lons)

    # --- dropsonde index 18 --------------------------------------------------
    drop_files   = synoptic_overview.pull_files_helper(path_drops)
    sonde_18_file = drop_files[18]
    print(f'Sonde 18: {sonde_18_file}')

    drop18    = xr.open_dataset(path_drops + sonde_18_file)
    ref_lon18 = float(drop18.reference_lon.values)
    ref_lat18 = float(drop18.reference_lat.values)

    # --- flight-level RH data ------------------------------------------------
    fl_data  = xr.open_dataset(path_fl + fl_file, decode_times=False)
    fl_time  = build_time_array(fl_data, axis_type='decimal')  # decimal UTC hours

    rh  = fl_data['HUM_REL.d'].values
    lat = fl_data['LATref'].values
    lon = fl_data['LONref'].values

    # trim to relevant flight times
    t_start, t_end = 18.8, 20.5
    time_mask = (fl_time >= t_start) & (fl_time <= t_end)

    # subsample to ~every 2 minutes at 1 Hz → every 120 points
    subsample = 60 # 120
    idx_full = np.where(time_mask)[0]
    idx_sub  = idx_full[::subsample]

    rh_sub  = rh[idx_sub]
    lat_sub = lat[idx_sub]
    lon_sub = lon[idx_sub]

    # keep only finite values within domain
    valid = (np.isfinite(rh_sub) & np.isfinite(lat_sub) & np.isfinite(lon_sub) &
             (lon_sub >= lons[0]) & (lon_sub <= lons[1]) &
             (lat_sub >= lats[0]) & (lat_sub <= lats[1]))

    # --- SAL composite -------------------------------------------------------
    sal_indices = [0, 1, 2, 27, 28, 29, 30]
    all_data    = xr.open_dataset(path_data + "all_sondes_20240715I1.nc")
    (p_sal, T_sal, Td_sal, u_sal, v_sal,
     p_sal_all, T_sal_all, Td_sal_all) = _composite_helper(all_data, sal_indices,)

    # --- figure sizing -------------------------------------------------------
    asp          = synoptic_overview._panel_aspect(lons, lats)
    panel_height = 4.5
    goes_width   = panel_height * asp
    skewt_width  = panel_height

    fig_width = goes_width + 2 * skewt_width
    fig = plt.figure(figsize=(fig_width, panel_height))

    gs = GridSpec(1, 3, figure=fig,
                  width_ratios=[goes_width, skewt_width, skewt_width],
                  wspace=0.55)

    # --- panel (a): GOES RGB + flight-level RH scatter -----------------------
    pc      = ccrs.PlateCarree()
    ax_goes = fig.add_subplot(gs[0, 0], projection=pc)

    synoptic_overview.test_plot(ds_goes, channel='rgb', lats=lats, lons=lons,
                                ax=ax_goes, fig=fig,
                                show_colorbar=False,
                                show_lat=True, show_lon=True,
                                show_title_banner=False,
                                show_time=False,
                                panel_label='a',
                                label_x=0.05, label_y=0.95,
                                time_x=0.95, time_y=0.95)

    # flight-level RH scatter
    sc = ax_goes.scatter(lon_sub[valid], lat_sub[valid], c=rh_sub[valid],
                         cmap='RdYlBu', vmin=0, vmax=100,
                         s=36, zorder=10, transform=pc,
                         edgecolors='none')

    # colorbar to the right of panel (a)
    label_fs = plt.rcParams['axes.labelsize']
    cb = fig.colorbar(sc, ax=ax_goes, orientation='vertical',
                      fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label('Relative Humidity (%)', fontsize=label_fs)

    # dropsonde 18 marker
    nudge = 0.1
    if (lons[0] <= ref_lon18 <= lons[1]) and (lats[0] <= ref_lat18 <= lats[1]):
        ax_goes.scatter(ref_lon18, ref_lat18, c='w', marker='o', s=40,
                        transform=pc, zorder=100)
        ax_goes.text(ref_lon18 + nudge, ref_lat18 + nudge, '18',
                     color='w', fontsize=10, fontweight='bold',
                     transform=pc, zorder=1000)

    # --- panel (b): Skew-T dropsonde 18 --------------------------------------
    skew18 = make_skewt(fig, gs[0, 1], path_drops, sonde_18_file,
                        panel_label='b', xlims=[10, 30], ylims=[1020, 645],
                        add_leg=False)
    skew18.ax.set_title('Dropsonde 18', fontsize=label_fs)

    # --- panel (c): SAL composite Skew-T ------------------------------------
    skew_sal = make_composite_skewt(fig, gs[0, 2],
                                    p_sal, T_sal, Td_sal, u_sal, v_sal,
                                    p_sal_all, T_sal_all, Td_sal_all,
                                    panel_label='c',
                                    title='SAL Composite',
                                    xlims=[-15, 30], ylims=[1020, 360])

    # --- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'sal.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f'Saved {outfile}')


# ---------------------------------------------------------------------------
# Figure 10: 2 x 2 — GOES | Skew-T 26 // Skew-T 21 | Skew-T 22
# ---------------------------------------------------------------------------

def outflow():
    """
    Figure 10: 2 x 2 layout.
      (a) upper-left  : GOES-16 RGB (last file in goes-level2-tdr) with white
                        markers for dropsonde locations 21, 22, and 26.
      (b) upper-right : Skew-T of dropsonde index 26.
      (c) lower-left  : Skew-T of dropsonde index 21.
      (d) lower-right : Skew-T of dropsonde index 22.

    Saves to figures/outflow_profiles.png
    """
    lons = (-55.5, -52.5)
    lats = (9,     12.5)

    path_goes  = str(path_project) + "/data/goes-level2-tdr/"
    path_drops = str(path_project) + "/data/dropsondes/"

    # --- GOES data (last file in folder) -------------------------------------
    goes_files = synoptic_overview.pull_files_helper(path_goes)
    ds_goes    = synoptic_overview.data_subset(path_goes, goes_files[-1], lats, lons)

    # --- dropsonde files at indices 21, 22, and 26 ---------------------------
    drop_files    = synoptic_overview.pull_files_helper(path_drops)
    sonde_21_file = drop_files[21]
    sonde_22_file = drop_files[22]
    sonde_26_file = drop_files[26]
    print(f'Sonde 21: {sonde_21_file}')
    print(f'Sonde 22: {sonde_22_file}')
    print(f'Sonde 26: {sonde_26_file}')

    def _ref_coords(fname):
        ds = xr.open_dataset(path_drops + fname)
        return float(ds.reference_lon.values), float(ds.reference_lat.values)

    ref_lon21, ref_lat21 = _ref_coords(sonde_21_file)
    ref_lon22, ref_lat22 = _ref_coords(sonde_22_file)
    ref_lon26, ref_lat26 = _ref_coords(sonde_26_file)

    # --- figure sizing -------------------------------------------------------
    # Use equal column widths so all four panels are the same size.
    # Cartopy handles the GOES panel's geographic aspect ratio internally.
    panel_height = 4.5
    skewt_width  = panel_height

    fig_width  = skewt_width * 2
    fig_height = panel_height * 2
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(2, 2, figure=fig,
                  wspace=0.35, hspace=0.1)

    label_fs = plt.rcParams['axes.labelsize']
    nudge    = 0.1
    skewt_xlims = [10, 30]
    skewt_ylims = [1020, 645]

    # --- panel (a): GOES RGB + dropsonde markers in white --------------------
    pc      = ccrs.PlateCarree()
    ax_goes = fig.add_subplot(gs[0, 0], projection=pc)

    synoptic_overview.test_plot(ds_goes, channel='rgb', lats=lats, lons=lons,
                                ax=ax_goes, fig=fig,
                                show_colorbar=False,
                                show_lat=True, show_lon=True,
                                show_title_banner=False,
                                show_time=True,
                                panel_label='a',
                                label_x=0.05, label_y=0.95,
                                time_x=0.95, time_y=0.95)

    for ref_lon, ref_lat, lbl in [(ref_lon21, ref_lat21, '21'),
                                   (ref_lon22, ref_lat22, '22'),
                                   (ref_lon26, ref_lat26, '26')]:
        if (lons[0] <= ref_lon <= lons[1]) and (lats[0] <= ref_lat <= lats[1]):
            ax_goes.scatter(ref_lon, ref_lat, c='w', marker='o', s=40,
                            transform=pc, zorder=100)
            ax_goes.text(ref_lon + nudge, ref_lat + nudge, lbl,
                         color='w', fontsize=10, fontweight='bold',
                         transform=pc, zorder=1000)

    # --- panel (b): Skew-T dropsonde 26 (upper right) ------------------------
    skew26 = make_skewt(fig, gs[0, 1], path_drops, sonde_26_file,
                        panel_label='b',
                        xlims=skewt_xlims, ylims=skewt_ylims)
    skew26.ax.set_title('Dropsonde 26', fontsize=label_fs)

    # --- panel (c): Skew-T dropsonde 21 (lower left) -------------------------
    skew21 = make_skewt(fig, gs[1, 0], path_drops, sonde_21_file,
                        panel_label='c',
                        xlims=skewt_xlims, ylims=skewt_ylims,
                        add_leg=False)
    skew21.ax.set_title('Dropsonde 21', fontsize=label_fs)

    # --- panel (d): Skew-T dropsonde 22 (lower right) ------------------------
    skew22 = make_skewt(fig, gs[1, 1], path_drops, sonde_22_file,
                        panel_label='d',
                        xlims=skewt_xlims, ylims=skewt_ylims,
                        add_leg=False)
    skew22.ax.set_title('Dropsonde 22', fontsize=label_fs)

    # --- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'outflow_profiles.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f'Saved {outfile}')


def _plot_fl_timeseries(ax, fl_time, var_left, label_left, color_left,
                        legend_label_left=None,
                        var_right=None, label_right=None, color_right=None,
                        extra_vars=None,
                        t_start=19.125, t_end=19.45,
                        panel_label=None, add_leg=True):
    """
    Plot one or two flight-level variables vs decimal UTC time on a shared
    x-axis.  A second y-axis is added on the right when var_right is given.
    extra_vars is an optional list of (array, label, color) tuples added to
    the LEFT axis.  Data are plotted unsmoothed at full 1-Hz resolution.

    Parameters
    ----------
    ax                : matplotlib Axes
    fl_time           : 1-D decimal UTC hour array (from build_time_array)
    var_left          : 1-D array for left y-axis
    label_left        : y-axis label string for left axis
    color_left        : line color for var_left
    legend_label_left : legend entry for var_left; defaults to label_left
    var_right         : 1-D array for right y-axis (optional)
    label_right       : y-axis label for right axis (optional)
    color_right       : line color for var_right (optional)
    extra_vars        : list of (array, legend_label, color) for left axis
    t_start/end       : UTC decimal hour limits for x-axis
    panel_label       : single character drawn in upper-left corner
    """
    label_fs = plt.rcParams['axes.labelsize']
    tick_fs  = plt.rcParams['xtick.labelsize']

    mask   = (fl_time >= t_start) & (fl_time <= t_end)
    t_plot = fl_time[mask]

    leg_lbl_left = legend_label_left if legend_label_left is not None else label_left
    ax.plot(t_plot, var_left[mask], color=color_left, lw=0.9, label=leg_lbl_left)

    if extra_vars is not None:
        for ev_arr, ev_lbl, ev_col in extra_vars:
            ax.plot(t_plot, ev_arr[mask], color=ev_col, lw=0.9, label=ev_lbl)

    ax.set_xlim(t_start, t_end)
    ax.set_ylabel(label_left, fontsize=label_fs, color=color_left)
    ax.tick_params(axis='y', labelcolor=color_left, labelsize=tick_fs)
    ax.tick_params(axis='x', labelsize=tick_fs)

    # x-ticks: every ~4 min (≈0.0667 hr)
    xt = np.arange(np.ceil(t_start * 60) / 60,
                   t_end + 1e-9,
                   4 / 60)
    ax.set_xticks(xt)
    def _fmt(h):
        hh = int(h)
        mm = int(round((h - hh) * 60))
        return f'{hh:02d}:{mm:02d}'
    ax.set_xticklabels([_fmt(x) for x in xt], rotation=30, ha='right',
                       fontsize=tick_fs)

    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax_r = None
    if var_right is not None:
        ax_r = ax.twinx()
        ax_r.plot(t_plot, var_right[mask], color=color_right, lw=0.9,
                  label=label_right)
        ax_r.set_ylabel(label_right, fontsize=label_fs, color=color_right)
        ax_r.tick_params(axis='y', labelcolor=color_right, labelsize=tick_fs)

    # combined legend
    lines, labels = ax.get_legend_handles_labels()
    if ax_r is not None:
        lr, ll = ax_r.get_legend_handles_labels()
        lines += lr
        labels += ll
    if extra_vars is not None or var_right is not None:
        if add_leg:
            ax.legend(lines, labels, fancybox=False, shadow=False,
                        facecolor='w', edgecolor='k', framealpha=1,
                        loc='upper right', fontsize=plt.rcParams['legend.fontsize'],
                        ncol=1)

    # panel label — matching bbox style used in make_skewt()
    if panel_label is not None:
        ax.text(0.01, 0.92, f'({panel_label})', transform=ax.transAxes,
                fontsize=label_fs,
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=1.,
                          edgecolor='k', linewidth=1.2, pad=4))


    # load dropsonde data
    filepath="/Users/ethanmurray/files-research-postdoc/data/aew/dropsonde/"
    drops = xr.open_dataset(filepath + 'all_sondes_20240715I1.nc')


    # optional: add vertical lines showing the time of dropsonde releases!
    for li, launch_time in enumerate(drops.launch_time.values):
        # pull starttimes for all dropsondes
        timestamp = pd.Timestamp(launch_time)    
        # Calculate decimal hour
        decimal_hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600

        # Only plot if the dropsonde is within the current x-axis limits
        if t_start <= decimal_hour <= t_end:
            # add dashed line showing location of the drop!
            ax.axvline(x=decimal_hour, c='k', zorder=1, ls='--', lw=.7)
            
            # Calculate normalized x-coordinate for the text
            x_normalized = (decimal_hour - t_start) / (t_end - t_start)            
            ax.text(x_normalized, 1.01, str(li), transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=10, clip_on=False)


    return ax_r


# ---------------------------------------------------------------------------
# Figure 9: 3-row — GOES+Skew-Ts (top) | FL w/u/v (middle) | FL WVMR/T (bottom)
# ---------------------------------------------------------------------------

def convective_mixing():
    """
    Figure 9: 3-row layout.
      Row 0 (top, 3 equal columns):
        (a) GOES-16 RGB (goes-level2-tdr index 1) with dropsonde 16 & 17 markers.
        (b) Skew-T of dropsonde index 16.
        (c) Skew-T of dropsonde index 17.
      Row 1 (full width):
        (d) Flight-level vertical velocity (UWZ.d) and horizontal wind
            components u (UWX.d) and v (UWY.d) vs time, 19:07:30–19:27 UTC.
      Row 2 (full width):
        (e) Flight-level water-vapour mixing ratio (MR.d) on left axis and
            ambient temperature (TA.d) on right axis vs time.

    Saves to figures/convective_mixing.png
    """
    lons = (-55.5, -52.5)
    lats = (9,     12.5)

    path_goes  = str(path_project) + "/data/goes-level2-tdr/"
    path_drops = str(path_project) + "/data/dropsondes/"
    path_fl    = str(path_project) + "/data/flight-level/"
    fl_file    = "20240715I1_A_small.nc"

    t_start_fl = 19.125
    t_end_fl   = 19.45

    # --- GOES data (index 1) -------------------------------------------------
    goes_files = synoptic_overview.pull_files_helper(path_goes)
    ds_goes    = synoptic_overview.data_subset(path_goes, goes_files[1], lats, lons)

    # --- dropsonde files at indices 16 and 17 --------------------------------
    drop_files    = synoptic_overview.pull_files_helper(path_drops)
    sonde_16_file = drop_files[16]
    sonde_17_file = drop_files[17]
    print(f'Sonde 16: {sonde_16_file}')
    print(f'Sonde 17: {sonde_17_file}')

    def _ref_coords(fname):
        ds = xr.open_dataset(path_drops + fname)
        return float(ds.reference_lon.values), float(ds.reference_lat.values)

    ref_lon16, ref_lat16 = _ref_coords(sonde_16_file)
    ref_lon17, ref_lat17 = _ref_coords(sonde_17_file)

    # --- flight-level data ---------------------------------------------------
    fl_data = xr.open_dataset(path_fl + fl_file, decode_times=False)
    fl_time = build_time_array(fl_data, axis_type='decimal')

    w   = fl_data['UWZ.d'].values   # vertical velocity  (m/s)
    u   = fl_data['UWX.d'].values   # u wind component   (m/s)
    v   = fl_data['UWY.d'].values   # v wind component   (m/s)
    mr  = fl_data['MR.d'].values    # mixing ratio       (g/kg)
    ta  = fl_data['TA.d'].values    # ambient temperature (°C)

    # --- figure sizing -------------------------------------------------------
    # Smaller figure → fonts appear larger relative to panels.
    panel_height = 3.5          # top-row panel height (inches)
    skewt_width  = panel_height
    ts_height    = 2.5          # height of each time-series row

    fig_width  = skewt_width * 3
    fig_height = panel_height + 2 * ts_height
    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[panel_height, ts_height, ts_height],
                  wspace=0.40, hspace=0.30)

    label_fs = plt.rcParams['axes.labelsize']
    nudge    = 0.1
    skewt_xlims = [10, 30]
    skewt_ylims = [1020, 630]

    # --- panel (a): GOES RGB + dropsonde markers -----------------------------
    pc      = ccrs.PlateCarree()
    ax_goes = fig.add_subplot(gs[0, 0], projection=pc)

    synoptic_overview.test_plot(ds_goes, channel='rgb', lats=lats, lons=lons,
                                ax=ax_goes, fig=fig,
                                show_colorbar=False,
                                show_lat=True, show_lon=True,
                                show_title_banner=False,
                                show_time=True,
                                panel_label='a',
                                label_x=0.05, label_y=0.95,
                                time_x=0.95, time_y=0.95)
    
    # plot P-3 flight track atop GOES image (clipped to t_start_fl → t_end_fl)
    starti = int(np.searchsorted(fl_time, t_start_fl))
    endi   = int(np.searchsorted(fl_time, t_end_fl))
    ax_goes.plot(fl_data['LONref'].values[starti:endi],
                 fl_data['LATref'].values[starti:endi],
                 color='w', lw=1.5, transform=pc, zorder=50)

    for ref_lon, ref_lat, lbl in [(ref_lon16, ref_lat16, '16'),
                                   (ref_lon17, ref_lat17, '17')]:
        if (lons[0] <= ref_lon <= lons[1]) and (lats[0] <= ref_lat <= lats[1]):
            ax_goes.scatter(ref_lon, ref_lat, c='w', marker='o', s=40,
                            transform=pc, zorder=100)
            ax_goes.text(ref_lon + nudge, ref_lat + nudge, lbl,
                         color='w', fontsize=10, fontweight='bold',
                         transform=pc, zorder=1000)

    # --- panel (b): Skew-T dropsonde 16 -------------------------------------
    skew16 = make_skewt(fig, gs[0, 1], path_drops, sonde_16_file,
                        panel_label='b',
                        xlims=skewt_xlims, ylims=skewt_ylims,
                        add_leg=True)
    skew16.ax.set_title('Dropsonde 16', fontsize=label_fs)

    # --- panel (c): Skew-T dropsonde 17 -------------------------------------
    skew17 = make_skewt(fig, gs[0, 2], path_drops, sonde_17_file,
                        panel_label='c',
                        xlims=skewt_xlims, ylims=skewt_ylims,
                        add_leg=False)
    skew17.ax.set_title('Dropsonde 17', fontsize=label_fs)

    # --- panel (d): vertical velocity + u/v winds ----------------------------
    ax_d = fig.add_subplot(gs[1, :])   # span all 3 columns
    _plot_fl_timeseries(
        ax_d, fl_time,
        var_left=u,
        label_left='Wind components (m s$^{-1}$)',
        legend_label_left='U',
        color_left='steelblue',
        extra_vars=[(v, 'V', 'tomato'),
                    (w, 'W', 'k')],
        t_start=t_start_fl, t_end=t_end_fl,
        panel_label='d',
    )
    ax_d.set_ylim([-6,12])

    # --- panel (e): WVMR + temperature ---------------------------------------
    ax_e = fig.add_subplot(gs[2, :])   # span all 3 columns
    _plot_fl_timeseries(
        ax_e, fl_time,
        var_left=mr,
        label_left='WVMR (g kg$^{-1}$)',
        color_left='royalblue',
        var_right=ta,
        label_right='Temperature (°C)',
        color_right='firebrick',
        t_start=t_start_fl, t_end=t_end_fl,
        panel_label='e',
        add_leg=False
    )
    ax_e.set_xlabel('Time (Hours, UTC)', fontsize=label_fs)
    ax_e.set_ylim([8,16])

    # --- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'convective_mixing.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f'Saved {outfile}')