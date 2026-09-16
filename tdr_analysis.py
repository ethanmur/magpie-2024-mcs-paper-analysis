#%%
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
import pandas as pd
from matplotlib.colors import ListedColormap

path_project = Path(__file__).parent.parent
path_figures = str(path_project) + "/figures/"
path_data    = str(path_project) + "/data/"
path_code    = str(path_project) + "/code/"
sys.path.append(path_code)

import synoptic_overview

sys.path.append("/Users/ethanmurray/files-research-postdoc/code/aew-analysis/plotting/")
import magpie_data_plotting


# ---------------------------------------------------------------------------
# Helper: build rain classification colormap (from old_code.py conventions)
# ---------------------------------------------------------------------------

def _rain_cmap():
    viridis = plt.cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0:51,   :] = [0.00, 0.00, 0.00, 1.0]   # Black        - Weak Echo
    newcolors[52:103, :] = [0.30, 0.55, 1.00, 1.0]   # Lighter blue - Stratiform
    newcolors[104:155,:] = [0.78, 0.78, 0.78, 1.0]   # Lighter grey  - Shallow Conv.
    newcolors[156:207,:] = [1.00, 0.55, 0.00, 1.0]   # Vibrant orange- Moderate Conv.
    newcolors[208:256,:] = [1.00, 1.00, 0.00, 1.0]   # Bright yellow- Deep Conv.
    return ListedColormap(newcolors)


def _set_vort_ticks(cbar):
    """Label a relative-vorticity colorbar with rounded decimals (no leading 0)."""
    ticks = [-4e-3, -2e-3, 0., 2e-3, 4e-3]
    def _fmt(t):
        if t == 0:
            return '0'
        return f'{t:.3f}'.replace('0.', '.')   # -0.004 -> -.004, 0.002 -> .002
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([_fmt(t) for t in ticks])


# ---------------------------------------------------------------------------
# Helper: plot GOES RGB background on an existing cartopy axes
# ---------------------------------------------------------------------------

def _plot_goes_background(ax, goes_ds):
    """Plot GOES RGB composite as a pcolormesh background."""
    pc = ccrs.PlateCarree()
    RGB = goes_ds['rgb'].values
    RGB = np.nan_to_num(RGB, nan=0.0)
    RGB = np.clip(RGB, 0.0, 1.0)

    x = goes_ds.lon.values
    y = goes_ds.lat.values
    x = np.where(np.isfinite(x), x, 0)
    y = np.where(np.isfinite(y), y, 0)

    coord_mask = np.isfinite(goes_ds.lon.values) & np.isfinite(goes_ds.lat.values)
    RGB = np.where(
        np.stack([coord_mask, coord_mask, coord_mask], axis=-1),
        RGB, np.nan)

    ax.pcolormesh(x, y, RGB, transform=pc)


# ---------------------------------------------------------------------------
# Helper: plot one reflectivity + wind barb panel
# ---------------------------------------------------------------------------

def _plot_tdr_panel(ax, tdr_ds, lev_i, lons, lats, goes_ds=None,
                    downsample=12, barb_color='k', barb_length=4.5,
                    panel_label=None, title=None, show_barbs=True):
    """
    Optionally plot GOES RGB background, then fill-contour reflectivity
    and overlay wind barbs at a given level index.

    Parameters
    ----------
    ax        : cartopy GeoAxes (PlateCarree projection)
    tdr_ds    : xarray Dataset from one plan-view NC file
    lev_i     : integer index into the `level` dimension
    lons, lats: (min, max) tuples for domain
    goes_ds   : optional data_subset result for GOES RGB background
    downsample: spatial downsampling step for wind barbs
    show_barbs: set False to skip wind barbs entirely (e.g. Pass 1, whose
        wind data was corrupted by poor instrument performance)
    """
    pc = ccrs.PlateCarree()

    # --- optional GOES background ---
    if goes_ds is not None:
        _plot_goes_background(ax, goes_ds)

    lat2d = tdr_ds['LATITUDE'].isel(time=0).values    # (x, y)
    lon2d = tdr_ds['LONGITUDE'].isel(time=0).values

    dbz = tdr_ds['REFLECTIVITY'].isel(time=0, level=lev_i).values
    u   = tdr_ds['U'].isel(time=0, level=lev_i).values
    v   = tdr_ds['V'].isel(time=0, level=lev_i).values

    # mask missing values
    fill = -999.
    dbz = np.where(dbz < fill + 1, np.nan, dbz)
    u   = np.where(u   < fill + 1, np.nan, u)
    v   = np.where(v   < fill + 1, np.nan, v)

    # domain mask
    in_dom = ((lon2d >= lons[0]) & (lon2d <= lons[1]) &
              (lat2d >= lats[0]) & (lat2d <= lats[1]))
    dbz = np.where(in_dom, dbz, np.nan)

    # reflectivity filled contours
    levels_dbz = np.arange(0, 46, 2.5)
    cf = ax.contourf(lon2d, lat2d, dbz,
                     levels=levels_dbz, cmap='RdYlBu_r',
                     extend='both', transform=pc)

    # wind barbs (m/s → knots), downsampled and domain-filtered
    if show_barbs:
        u_kts = u * 1.944
        v_kts = v * 1.944
        lat_d = lat2d[::downsample, ::downsample]
        lon_d = lon2d[::downsample, ::downsample]
        u_d   = u_kts[::downsample, ::downsample]
        v_d   = v_kts[::downsample, ::downsample]
        in_d  = ((lon_d >= lons[0]) & (lon_d <= lons[1]) &
                 (lat_d >= lats[0]) & (lat_d <= lats[1]) &
                 np.isfinite(u_d) & np.isfinite(v_d))
        ax.barbs(lon_d[in_d], lat_d[in_d], u_d[in_d], v_d[in_d],
                 length=barb_length, color=barb_color, pivot='middle',
                 transform=pc)

    _decorate_ax(ax, lons, lats, panel_label=panel_label, title=title)

    return cf


# ---------------------------------------------------------------------------
# Helper: plot one rainfall classification panel
# ---------------------------------------------------------------------------

def _plot_class_panel(ax, class_ds, tdr_ds, case_i, lons, lats,
                      rain_cmap, goes_ds=None,
                      panel_label=None, title=None):
    """
    Optionally plot GOES RGB background, then pcolormesh rainfall classification.
    Uses lat/lon from the paired TDR plan-view file.
    """
    pc = ccrs.PlateCarree()

    # --- optional GOES background ---
    if goes_ds is not None:
        _plot_goes_background(ax, goes_ds)

    lat2d = tdr_ds['LATITUDE'].isel(time=0).values
    lon2d = tdr_ds['LONGITUDE'].isel(time=0).values

    # classification: (num_cases, northward_distance, eastward_distance)
    # no transpose needed — northward_distance maps to x, eastward_distance to y
    profile = class_ds['rainfall_classification'].isel(num_cases=case_i).values

    in_dom = ((lon2d >= lons[0]) & (lon2d <= lons[1]) &
              (lat2d >= lats[0]) & (lat2d <= lats[1]))
    profile = np.where(in_dom, profile, np.nan)

    c = ax.pcolormesh(lon2d, lat2d, profile,
                      cmap=rain_cmap, vmin=-0.5, vmax=4.5,
                      transform=pc, shading='nearest')

    _decorate_ax(ax, lons, lats, panel_label=panel_label, title=title)

    return c


# ---------------------------------------------------------------------------
# Helper: common axes decoration
# ---------------------------------------------------------------------------

def _decorate_ax(ax, lons, lats, panel_label=None, title=None):
    pc       = ccrs.PlateCarree()
    tick_fs  = plt.rcParams['xtick.labelsize']
    label_fs = plt.rcParams['axes.labelsize']

    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=pc)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=':')

    lon_ticks = np.linspace(lons[0], lons[1], 4)
    lat_ticks = np.linspace(lats[0], lats[1], 4)

    # gridlines locked to the same locations as the ticks so they line up
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.4,
                      color='0.6', linestyle='--')
    gl.xlocator = plt.matplotlib.ticker.FixedLocator(lon_ticks)
    gl.ylocator = plt.matplotlib.ticker.FixedLocator(lat_ticks)

    ax.set_xticks(lon_ticks, crs=pc)
    ax.set_yticks(lat_ticks, crs=pc)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.1f', degree_symbol='°'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.1f',  degree_symbol='°'))
    ax.tick_params(axis='both', which='both', labelsize=tick_fs,
                   top=False, right=False, direction='out')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    if title is not None:
        ax.set_title(title, fontsize=label_fs)

    if panel_label is not None:
        ax.text(0.05, 0.95, f'({panel_label})',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=label_fs,
                bbox=dict(facecolor='white', alpha=1.,
                          edgecolor='k', linewidth=1.2, pad=4))


# ---------------------------------------------------------------------------
# Figure 4: 3 × 3 TDR radar figure
# ---------------------------------------------------------------------------


# oldlimits: (11, -54.0, 10.75, -53.5)
def tdr_radar_figure(cross_endpoints_1=(10.55, -54.3, 10.15, -53.7),
                     cross_endpoints_2=(10.55, -54.3, 10.15, -53.7)):
    """
    Figure 4: 3 x 3 layout with GOES RGB background in every panel.
      Columns: TDR passes at 18:04, 19:19, 20:30 UTC
      Row 0:   Reflectivity + wind barbs at 2 km
      Row 1:   Reflectivity + wind barbs at 6 km
      Row 2:   Rainfall classification
    Colorbars in a narrow column to the right (dBZ repeated for rows 0–1;
    rain classification for row 2).

    The full P-3 flight track is drawn (white) on the top-row panels (a–c),
    and the across-axis cross-section is drawn on panel (b).

    cross_endpoints : tuple (lat1, lon1, lat2, lon2) or None
        Endpoints of the across-axis cross-section (matches build_cross_curtain
        in tdr_analysis_new_cross.py). When given, the section line is drawn on
        panel (b) and its great-circle length is appended to the filename as a
        distance suffix. Pass None to disable both.

    Saves to figures/tdr_radar{_crossNNkm}.png
    """
    # lons = (-56.5, -51.5)
    # lats = (8,     13)
    lons = (-55., -53.)
    lats = (9., 12.)

    path_pv   = path_data + "tdr/plan-view/"
    path_cls  = path_data + "tdr/precip-classifications/"
    path_goes = path_data + "goes-level2-tdr/"

    # plan-view files, column titles, GOES files (matched by time)
    pv_files   = ['240715I1_1804_xy.nc',
                  '240715I1_1919_xy.nc',
                  '240715I1_2030_xy.nc']
    col_titles = ['Pass 1: 18:04 UTC', 'Pass 2: 19:19 UTC', 'Pass 3: 20:30 UTC']
    class_indices = [0, 2, 4]

    # GOES files sorted alphabetically → same order as TDR passes
    goes_files = synoptic_overview.pull_files_helper(path_goes)
    print(f'GOES files: {goes_files}')

    # load TDR and classification datasets
    tdr_datasets  = [xr.open_dataset(path_pv + f) for f in pv_files]
    class_ds      = xr.open_dataset(path_cls + 'tdr_aew_rainfall_classifications.nc')
    goes_datasets = [synoptic_overview.data_subset(path_goes, f, lats, lons)
                     for f in goes_files]

    # find level indices for 2 km and 6 km
    levels  = tdr_datasets[0]['level'].values   # km
    lev_2km = int(np.argmin(np.abs(levels - 2.)))
    lev_6km = int(np.argmin(np.abs(levels - 6.)))
    print(f'Level index 2 km: {lev_2km} ({levels[lev_2km]:.1f} km)')
    print(f'Level index 6 km: {lev_6km} ({levels[lev_6km]:.1f} km)')

    rain_cmap = _rain_cmap()

    # --- P-3 flight track (colored per radar pass, top row) ------------------
    fl_path = '/Users/ethanmurray/files-research-postdoc/data/aew/flight-level/'
    fl = xr.open_dataset(fl_path + '20240715I1_A_small.nc', decode_times=False)
    fl_lat  = fl['LATref'].values
    fl_lon  = fl['LONref'].values
    fl_time = magpie_data_plotting.build_time_array(fl, axis_type='decimal')  # hours UTC

    # only draw the track within each pass window (reduces clutter)
    radar_passes = [([17.8,  18.3],  'red'),
                    ([18.95, 19.45], 'blue'),
                    ([20.2,  20.65], 'w')]

    # --- across-axis cross-section + distance suffix -------------------------
    dist_suffix = ''
    if cross_endpoints_1 is not None:
        lat1, lon1, lat2, lon2 = cross_endpoints_1
        lat3, lon3, lat4, lon4 = cross_endpoints_2
        
        r_earth = 6371.0  # km
        p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
        dphi   = np.deg2rad(lat2 - lat1)
        dlmb   = np.deg2rad(lon2 - lon1)
        a      = np.sin(dphi / 2.)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.)**2
        cross_dist_km = 2. * r_earth * np.arcsin(np.sqrt(a))
        dist_suffix = f'_cross{cross_dist_km:.0f}km'
        print(f'Cross-section length: {cross_dist_km:.1f} km')

    # --- figure sizing -------------------------------------------------------
    # 3 data columns + 1 narrow colorbar column
    asp          = synoptic_overview._panel_aspect(lons, lats)
    panel_height = 3.5
    panel_width  = panel_height * asp
    cbar_width   = 0.25             # inches — narrow colorbar column
    fig_width    = panel_width * 3 + cbar_width
    fig_height   = panel_height * 3

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(3, 4, figure=fig,
                   width_ratios=[panel_width, panel_width, panel_width, cbar_width],
                   wspace=0.22, hspace=0.08)

    panel_labels = [['a', 'b', 'c'],
                    ['d', 'e', 'f'],
                    ['g', 'h', 'i']]
    row_labels   = ['', '', '']

    label_fs = plt.rcParams['axes.labelsize']

    cf_dbz = None
    c_cls  = None

    for col, (tdr_ds, goes_ds, ctitle, ci) in enumerate(
            zip(tdr_datasets, goes_datasets, col_titles, class_indices)):

        # --- row 0: 2 km reflectivity ----------------------------------------
        ax0 = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())
        cf  = _plot_tdr_panel(ax0, tdr_ds, lev_2km, lons, lats,
                              goes_ds=goes_ds,
                              panel_label=panel_labels[0][col],
                              title=ctitle,
                              show_barbs=(col != 0))   # Pass 1 wind data corrupted
        if cf_dbz is None:
            cf_dbz = cf

        # P-3 flight track: only this column's own pass segment (col -> pass)
        (t0, t1), pass_color = radar_passes[col]
        seg = (fl_time >= t0) & (fl_time <= t1)
        ax0.plot(fl_lon[seg], fl_lat[seg], color=pass_color, lw=1.5, zorder=6,
                 transform=ccrs.PlateCarree())

        # across-axis cross-section on panel (b) only
        if cross_endpoints_1 is not None and col == 1:
            ax0.plot([lon1, lon2], [lat1, lat2], color='firebrick', lw=2.0,
                     zorder=7, transform=ccrs.PlateCarree())
            ax0.scatter([lon1, lon2], [lat1, lat2], color='firebrick', s=18,
                        zorder=8, transform=ccrs.PlateCarree())

        if cross_endpoints_1 is not None and col == 2:
            ax0.plot([lon3, lon4], [lat3, lat4], color='firebrick', lw=2.0,
                     zorder=7, transform=ccrs.PlateCarree())
            ax0.scatter([lon3, lon4], [lat3, lat4], color='firebrick', s=18,
                        zorder=8, transform=ccrs.PlateCarree())

        if col == 0:
            ax0.set_ylabel(f'{row_labels[0]}\nLatitude (°)', fontsize=label_fs)
        else:
            ax0.set_ylabel('')
            ax0.set_yticklabels([])
        ax0.set_xlabel('')
        ax0.set_xticklabels([])

        # --- row 1: 6 km reflectivity ----------------------------------------
        ax1 = fig.add_subplot(gs[1, col], projection=ccrs.PlateCarree())
        cf1 = _plot_tdr_panel(ax1, tdr_ds, lev_6km, lons, lats,
                              goes_ds=goes_ds,
                              panel_label=panel_labels[1][col],
                              title=None,
                              show_barbs=(col != 0))   # Pass 1 wind data corrupted

        if col == 0:
            ax1.set_ylabel(f'{row_labels[1]}\nLatitude (°)', fontsize=label_fs)
        else:
            ax1.set_ylabel('')
            ax1.set_yticklabels([])
        ax1.set_xlabel('')
        ax1.set_xticklabels([])

        # --- row 2: rainfall classification ----------------------------------
        ax2 = fig.add_subplot(gs[2, col], projection=ccrs.PlateCarree())
        c   = _plot_class_panel(ax2, class_ds, tdr_ds, ci, lons, lats,
                                rain_cmap,
                                goes_ds=goes_ds,
                                panel_label=panel_labels[2][col],
                                title=None)
        if c_cls is None:
            c_cls = c

        if col == 0:
            ax2.set_ylabel(f'{row_labels[2]}\nLatitude (°)', fontsize=label_fs)
        else:
            ax2.set_ylabel('')
            ax2.set_yticklabels([])
        ax2.set_xlabel('Longitude (°)', fontsize=label_fs)

    # --- colorbars in the narrow right column --------------------------------
    # Row 0: dBZ colorbar
    cax0 = fig.add_subplot(gs[0, 3])
    cb0  = fig.colorbar(cf_dbz, cax=cax0, orientation='vertical', extend='both')
    cb0.set_label('2 km TDR Reflectivity (dBZ)', fontsize=label_fs)

    # Row 1: dBZ colorbar (repeated)
    cax1 = fig.add_subplot(gs[1, 3])
    cb1  = fig.colorbar(cf1, cax=cax1, orientation='vertical', extend='both')
    cb1.set_label('6 km TDR Reflectivity (dBZ)', fontsize=label_fs)

    # Row 2: rain classification colorbar
    cax2 = fig.add_subplot(gs[2, 3])
    cb2  = fig.colorbar(c_cls, cax=cax2, orientation='vertical',
                        ticks=[0, 1, 2, 3, 4])
    cb2.ax.set_yticklabels(
        ['Weak Echo', 'Stratiform', 'Shallow Conv.', 'Mod. Conv.', 'Strong Conv.'],
    )#fontsize=plt.rcParams['xtick.labelsize'] - 1)
    cb2.set_label('TDR Rainfall Classification', fontsize=label_fs)

    # --- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + f'tdr_radar{dist_suffix}.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')



def plot_tdr_curtains(convective_axis_deg=None, add_barbs=False, add_vort=False,
                      x_axis='distance', p3_speed=110., cross_endpoints=(10.55, -54.3, 10.15, -53.7),
                      passind=None, inset_extend_km=18.):
    """
    3- to 6-panel TDR curtain figure.
      a: W-band refl
      b: TDR refl (+ optional U'/W barbs)
      c: wind curtain — U' (along-axis) when add_barbs=False, else V'
         (across-axis, since U' is already shown via barbs on panel b)
      d, e: V' across-axis wind and W vertical velocity (curtains, only
            when add_barbs=False); U' is always plotted atop V'
      last: relative vorticity (optional, when add_vort=True; always last)

    convective_axis_deg : float or None
        Rotates U/V into along/across-axis winds (degrees CW from N).
    add_barbs : bool
        Overlays U'/W wind barbs on panel b; appends '_wind_barbs' to
        filename. When False, U' and V' are instead plotted as their own
        curtain panels (U' above V'), followed by a W curtain.
    add_vort : bool
        Adds a panel with the TDR VORT field, placed after the wind panels;
        appends '_vort' to filename.
    x_axis : {'distance', 'time'}
        'distance' -> along-track distance from the start of each flight leg
        (km), computed from elapsed time at ``p3_speed``; 'time' -> decimal
        hours UTC.
    p3_speed : float
        Nominal P-3 ground speed (m/s) used to convert elapsed time to
        along-track distance. Default 110 m/s.
    passind : int or None
        Which pass (index into time_ranges: 0 -> ~1919 UTC, 1 -> ~2030 UTC)
        to process and mark with ``cross_endpoints``. When None (default),
        BOTH passes are processed using the same cross_endpoints — only
        appropriate if both passes share one cross-section line. When the
        two passes use different cross-section endpoints (as in run.py's
        per-pass loop), pass the matching passind on each call so it only
        overwrites that one pass's output file, instead of also
        re-marking the other pass with the wrong cross_endpoints.
    inset_extend_km : float
        Only used when x_axis='distance'. Extends every curtain panel's
        x-axis by this many km beyond the flight leg's actual length, to
        leave blank room for the spatial location-map inset drawn in the
        top-right corner of panel (c) (TDR reflectivity).
    """
    # --- load data ------------------------------------------------------------
    curtain_path = path_data + '/tdr/'
    wband_path   = path_data + '/w-band-radar/'
    tdr_curtain  = xr.open_dataset(curtain_path + 'tdr_curtain_2160m.nc')
    wband        = xr.open_dataset(wband_path   + 'P3_Wband-radar_20240715T16-20240715T22_v1.0.nc')

    filepath = "/Users/ethanmurray/files-research-postdoc/data/aew/dropsonde/"
    drops    = xr.open_dataset(filepath + 'all_sondes_20240715I1.nc')

    rain_cmap = _rain_cmap()

    # across-axis cross-section endpoints (must match build_cross_curtain
    # defaults in tdr_analysis_new_cross.py) — used to mark where that
    # perpendicular cut crosses the flight-path curtain on pass 2
    cross_lat1, cross_lon1 = cross_endpoints[0], cross_endpoints[1]
    cross_lat2, cross_lon2 = cross_endpoints[2], cross_endpoints[3]

    def _dist_to_cross_line(lat, lon):
        """Perpendicular distance (deg) from (lat, lon) to the cross-section segment."""
        dx, dy = cross_lon2 - cross_lon1, cross_lat2 - cross_lat1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            return np.hypot(lon - cross_lon1, lat - cross_lat1)
        t = ((lon - cross_lon1) * dx + (lat - cross_lat1) * dy) / seg_len2
        t = np.clip(t, 0., 1.)
        proj_lon = cross_lon1 + t * dx
        proj_lat = cross_lat1 + t * dy
        return np.hypot(lon - proj_lon, lat - proj_lat)

    time_ranges = [[18.9, 19.45], [20.1, 20.75]]

    for flighti, flighth in enumerate(time_ranges):
        if passind is not None and flighti != passind:
            continue

        starth, endh = flighth

        print(f'Pass {flighti+1}: {flighth} UTC')

        # --- trim TDR data ----------------------------------------------------
        tdr_trimmed = tdr_curtain.sel(time=slice(starth, endh))
        t_tdr = tdr_trimmed.time.values
        h_tdr = tdr_trimmed.level.values   # km

        # --- x-axis: elapsed-time -> along-track distance ---------------------
        # distance (km) = elapsed hours * 3600 s * p3_speed (m/s) / 1000
        def _to_x(t):
            if x_axis == 'distance':
                return (t - starth) * 3600. * p3_speed / 1000.
            elif x_axis == 'time':
                return t
            raise ValueError("x_axis must be 'distance' or 'time'")

        if x_axis == 'distance':
            xlabel        = 'Distance from Flight Leg Start (km)'
            xlim_lo, xlim_hi = 0., (endh - starth) * 3600. * p3_speed / 1000.
            # extra blank room on the right for the panel (c) location-map inset
            xlim_hi += inset_extend_km
        else:
            xlabel        = 'Time (Hours, UTC)'
            xlim_lo, xlim_hi = starth, endh

        x_tdr = _to_x(t_tdr)

        # --- across-axis cross-section crossing point, using cross_endpoints
        # (only meaningful for the pass selected by passind, if given) -----
        cross_x = None
        fl_lat_trim = tdr_trimmed['LATref'].values
        fl_lon_trim = tdr_trimmed['LONref'].values
        dists = np.array([_dist_to_cross_line(la, lo)
                          for la, lo in zip(fl_lat_trim, fl_lon_trim)])
        if np.any(np.isfinite(dists)):
            cross_i = np.nanargmin(dists)
            cross_x = x_tdr[cross_i]
            print(f'  [along-track curtain, pass {flighti}] intersection at '
                  f'({fl_lat_trim[cross_i]:.4f}, {fl_lon_trim[cross_i]:.4f}), '
                  f'dist to line = {dists[cross_i]:.5f} deg')

        # --- optional axis rotation -------------------------------------------
        U_raw = tdr_trimmed['U'].values * 1.944   # m/s → knots  # (time, level)
        V_raw = tdr_trimmed['V'].values * 1.944   # m/s → knots
        W_raw = tdr_trimmed['W'].values * 1.944   # m/s → knots

        if convective_axis_deg is not None:
            theta = np.deg2rad(convective_axis_deg)
            Uprime = U_raw * np.sin(theta) + V_raw * np.cos(theta)  # along-axis
            Vprime = U_raw * np.cos(theta) - V_raw * np.sin(theta)  # across-axis
            Uprime_label = "TDR v' (Along Long Axis, kt)"    # along-axis component
            Vprime_label = "TDR u' (Across Long Axis, kt)"   # across-axis component
        else:
            Uprime = U_raw
            Vprime = V_raw
            Uprime_label = 'TDR U (kt)'
            Vprime_label = 'TDR V (kt)'

        # --- figure layout: panel a (W-band) + b (TDR refl) + wind curtain(s),
        # plus vorticity (always last) when add_vort=True. When add_barbs=False,
        # both wind components are shown as curtains with U' always plotted
        # above V', followed by W. When add_barbs=True, only V' (the
        # out-of-plane, across-axis component not already shown via barbs) is
        # plotted as a curtain. -------------------------------------------
        n_rows        = 3 + (2 if not add_barbs else 0) + (1 if add_vort else 0)
        height_ratios = [1] + [1.6] * (n_rows - 1)
        fig_height    = 8 + 2 * (n_rows - 3)
        fig = plt.figure(figsize=(12, fig_height))
        gs  = GridSpec(n_rows, 2, width_ratios=[1, 0.04],
                       height_ratios=height_ratios,
                       hspace=0.40, wspace=0.06)
        axs  = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
        caxs = [fig.add_subplot(gs[i, 1]) for i in range(n_rows)]
        ax_a, cax_a = axs[0], caxs[0]   # W-band reflectivity
        ax_b, cax_b = axs[1], caxs[1]   # TDR reflectivity (+ optional barbs)
        row = 2
        if not add_barbs:
            ax_c, cax_c = axs[row], caxs[row]; row += 1   # U' / along-axis wind (atop V')
            ax_v, cax_v = axs[row], caxs[row]; row += 1   # V' / across-axis wind
            ax_w, cax_w = axs[row], caxs[row]; row += 1   # W (vertical velocity)
        else:
            ax_c, cax_c = axs[row], caxs[row]; row += 1   # V' / across-axis wind
        if add_vort:
            ax_d, cax_d = axs[row], caxs[row]; row += 1   # relative vorticity

        # ---- panel (a): W-band reflectivity ----------------------------------
        times_wb = pd.DatetimeIndex(wband.time.values)
        t_wb = times_wb.hour + times_wb.minute / 60. + times_wb.second / 3600.
        h_wb = wband.height.values / 1000.
        wb_data = wband['corrected_reflectivity'].values.transpose()

        p_a = ax_a.pcolormesh(_to_x(t_wb), h_wb, wb_data,
                              cmap='RdYlBu_r', vmin=-10, vmax=20,
                              shading='auto')
        fig.colorbar(p_a, cax=cax_a, label='W-Band Refl. (dBZ)', extend='both')
        # nudge the top colorbar up slightly so it clears the rain strip
        cax_a_pos = cax_a.get_position()
        cax_a.set_position([cax_a_pos.x0, cax_a_pos.y0 + 0.02,
                            cax_a_pos.width, cax_a_pos.height])

        # rain classification bar above panel a (raised so it clears the
        # dropsonde number labels along the top of the axis)
        rain_class = tdr_trimmed['rainfall_classification'].values
        ax_a.pcolormesh(x_tdr, [3.95, 4.2],
                        np.vstack((rain_class, rain_class)),
                        cmap=rain_cmap, vmin=-0.5, vmax=4.5,
                        shading='nearest', clip_on=False)

        ax_a.set_xlim([xlim_lo, xlim_hi])
        ax_a.set_ylim([0, 3.25])
        ax_a.set_ylabel('Height (km)')
        ax_a.axhline(y=3., c='k', ls='--', lw=1.2)

        # ---- panel (b): TDR reflectivity (+ optional U/W barbs) --------------
        p_b = ax_b.pcolormesh(x_tdr, h_tdr, tdr_trimmed['REFLECTIVITY'].values.transpose(),
                              cmap='RdYlBu_r', vmin=-10, vmax=40, shading='auto')
        fig.colorbar(p_b, cax=cax_b, label='TDR Refl. (dBZ)', extend='both')

        if add_barbs:
            # downsample time and height for legible barb density
            t_skip = max(1, len(t_tdr) // 30) # 20)
            h_skip = max(1, len(h_tdr) // 16) # 12)
            x_b  = x_tdr[::t_skip]
            h_b  = h_tdr[::h_skip]
            # along-axis wind U' (in-plane for the flight-path curtain) → barbs
            u_b  = Uprime[::t_skip, ::h_skip]
            w_b  = W_raw[::t_skip, ::h_skip]
            x_2d, h_2d = np.meshgrid(x_b, h_b, indexing='ij')
            valid = np.isfinite(u_b) & np.isfinite(w_b)
            ax_b.barbs(x_2d[valid], h_2d[valid], u_b[valid], w_b[valid],
                       length=5, linewidth=0.7, color='k', pivot='middle')

        ax_b.set_xlim([xlim_lo, xlim_hi])
        ax_b.set_ylim([0, 18])
        ax_b.set_ylabel('Height (km)')
        ax_b.axhline(y=3., c='k', ls='--', lw=1.2)

        # ---- location-map inset in panel (c)'s top-right corner: spatial
        # reflectivity (colored, same as the curtain panels) + the P-3
        # flight path for this pass (this is the along-track curtain, so
        # only the flight path is relevant here — no cross-section line),
        # so the reader can see where this curtain's data were taken -------
        if x_axis == 'distance':
            pv_files = ['240715I1_1804_xy.nc', '240715I1_1919_xy.nc', '240715I1_2030_xy.nc']
            pv_file  = pv_files[flighti + 1]   # flighti 0 -> 1919, 1 -> 2030
            pv_ds    = xr.open_dataset(path_data + 'tdr/plan-view/' + pv_file)

            pv_lev_2km = int(np.argmin(np.abs(pv_ds['level'].values - 2.)))
            pv_lat  = pv_ds['LATITUDE'].isel(time=0).values
            pv_lon  = pv_ds['LONGITUDE'].isel(time=0).values
            pv_dbz  = pv_ds['REFLECTIVITY'].isel(time=0, level=pv_lev_2km).values
            pv_dbz  = np.where(pv_dbz < -998., np.nan, pv_dbz)
            pv_ds.close()

            pc = ccrs.PlateCarree()
            # anchored to ax_b's own axes-fraction coordinates (not a
            # floating figure-space box), so it stays locked to the panel's
            # upper-right corner through any later layout/bbox adjustments
            ax_inset = ax_b.inset_axes([0.58, 0.35, 0.42, 0.65],
                                       transform=ax_b.transAxes, projection=pc)

            pad = 0.25
            lon_lo = np.nanmin(fl_lon_trim) - pad
            lon_hi = np.nanmax(fl_lon_trim) + pad
            lat_lo = np.nanmin(fl_lat_trim) - pad
            lat_hi = np.nanmax(fl_lat_trim) + pad
            ax_inset.set_extent([lon_lo, lon_hi, lat_lo, lat_hi], crs=pc)
            # keep the correct geographic aspect ratio (don't stretch/distort
            # the radar data) — instead anchor the drawn map to the
            # top-right corner of its box, so any letterboxing needed to
            # preserve aspect eats into the left/bottom margin instead of
            # leaving a gap on the right edge (which must stay flush)
            ax_inset.set_anchor('NE')

            # colored reflectivity via filled contours, 5 dBZ intervals
            # (matches Figure 4's plan-view reflectivity color scheme)
            ax_inset.contourf(pv_lon, pv_lat, pv_dbz, levels=np.arange(0, 46, 5),
                              cmap='RdYlBu_r', extend='both', transform=pc)

            # P-3 flight path for this pass
            ax_inset.plot(fl_lon_trim, fl_lat_trim, color='k', lw=1.2,
                         transform=pc, zorder=5)
            ax_inset.scatter(fl_lon_trim[0], fl_lat_trim[0], marker='x', color='k',
                            s=35, linewidths=1.8, transform=pc, zorder=6)
            ax_inset.scatter(fl_lon_trim[-1], fl_lat_trim[-1], marker='*', color='k',
                            s=55, transform=pc, zorder=6)

            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            for spine in ax_inset.spines.values():
                spine.set_edgecolor('k')
                spine.set_linewidth(0.8)

        # ---- panel (c): wind curtain --------------------------------------
        # U' when both components are shown (always plotted atop V'); V' when
        # only one wind curtain is shown (add_barbs=True already puts U' on
        # panel b as barbs).
        c_data  = Uprime if not add_barbs else Vprime
        c_label = Uprime_label if not add_barbs else Vprime_label
        p_c = ax_c.pcolormesh(x_tdr, h_tdr, c_data.transpose(),
                              cmap='seismic', vmin=-30, vmax=30, shading='auto')
        fig.colorbar(p_c, cax=cax_c, label=c_label, extend='both')

        ax_c.set_xlim([xlim_lo, xlim_hi])
        ax_c.set_ylim([0, 18])
        ax_c.set_ylabel('Height (km)')
        if add_barbs and not add_vort:
            ax_c.set_xlabel(xlabel)

        # ---- panels: V' and W curtains (shown when barbs are off) ------------
        if not add_barbs:
            p_v = ax_v.pcolormesh(x_tdr, h_tdr, Vprime.transpose(),
                                  cmap='seismic', vmin=-30, vmax=30, shading='auto')
            fig.colorbar(p_v, cax=cax_v, label=Vprime_label, extend='both')
            ax_v.set_xlim([xlim_lo, xlim_hi])
            ax_v.set_ylim([0, 18])
            ax_v.set_ylabel('Height (km)')

            p_w = ax_w.pcolormesh(x_tdr, h_tdr, W_raw.transpose(),
                                  cmap='seismic', vmin=-10, vmax=10, shading='auto')
            fig.colorbar(p_w, cax=cax_w, label='TDR W (kt)', extend='both')
            ax_w.set_xlim([xlim_lo, xlim_hi])
            ax_w.set_ylim([0, 18])
            ax_w.set_ylabel('Height (km)')
            if not add_vort:
                ax_w.set_xlabel(xlabel)

        # ---- panel: relative vorticity (optional, always last) ---------------
        if add_vort:
            # / 1000. to convert to s-1 units!
            vort_data = tdr_trimmed['VORT'].values.transpose() / 1000.
            vmin=-4e-3
            vmax=4e-3
            # odd number of filled bands (7), with an even number of edges so
            # the central band straddles zero and renders pure white
            n_bands = 15
            levels = np.linspace(vmin, vmax, n_bands + 1)
            p_d = ax_d.contourf(x_tdr, h_tdr, vort_data,
                                  cmap='RdBu_r', levels=levels, extend='both')
                                #   shading='auto')
            cb_d = fig.colorbar(p_d, cax=cax_d, label='Rel. Vorticity (s⁻¹)')
            _set_vort_ticks(cb_d)
            ax_d.set_xlim([xlim_lo, xlim_hi])
            ax_d.set_ylim([0, 18])
            ax_d.set_ylabel('Height (km)')
            ax_d.set_xlabel(xlabel)

        # ---- panel (a) label: same figure-space height as (b)-(d), placed
        # next to the top colorbar rather than inside the (shorter) axis ------
        b_label_disp = ax_b.transAxes.transform((0.015, 0.93))
        a_label_axes = ax_a.transAxes.inverted().transform(b_label_disp)
        ax_a.text(a_label_axes[0], a_label_axes[1] + 1.92, '(a)',
                  transform=ax_a.transAxes, ha='left', va='center', fontsize=10,
                  zorder=20, clip_on=False,
                  bbox=dict(facecolor='white', alpha=1.,
                            edgecolor='k', linewidth=1.2, pad=4))

        # ---- shared: panel labels + dropsonde lines --------------------------
        panel_axs     = [ax_a, ax_b, ax_c]
        if not add_barbs:
            panel_axs += [ax_v, ax_w]
        if add_vort:
            panel_axs += [ax_d]
        # letters start at 'b' since ax_a already carries the floating '(a)'
        # label above it; ax_a itself now also gets an in-axis '(b)' label,
        # like the top colorbar/precip-strip panel in tdr_analysis_new_cross.py
        panel_letters = list('bcdefghi')[:len(panel_axs)]
        for ax, letter in zip(panel_axs, panel_letters):
            label_y = 0.80 if ax is ax_a else 0.93   # nudge (b) down slightly
            ax.text(0.015, label_y, f'({letter})',
                    transform=ax.transAxes, ha='left', va='top', fontsize=10,
                    zorder=20,
                    bbox=dict(facecolor='white', alpha=1.,
                              edgecolor='k', linewidth=1.2, pad=4))

            # mark where the across-axis cross-section cuts the flight track
            if cross_x is not None:
                ax.axvline(x=cross_x, color='firebrick', ls='--', lw=1.4, zorder=4)
                if ax is ax_a:
                    xn_cross = (cross_x - xlim_lo) / (xlim_hi - xlim_lo)
                    ax.text(xn_cross, 1.01, 'x', color='red', fontsize=12,
                            fontweight='bold', transform=ax.transAxes,
                            ha='center', va='bottom', zorder=21, clip_on=False)

            for li, launch_time in enumerate(drops.launch_time.values):
                ts  = pd.Timestamp(launch_time)
                dh  = ts.hour + ts.minute / 60. + ts.second / 3600.
                if starth <= dh <= endh:
                    ax.axvline(x=_to_x(dh), c='k', zorder=1, ls='--', lw=0.7)
                    if ax is ax_a:
                        xn = (_to_x(dh) - xlim_lo) / (xlim_hi - xlim_lo)
                        ax.text(xn, 1.01, str(li),
                                transform=ax.transAxes,
                                ha='center', va='bottom', fontsize=10, clip_on=False)

        # ---- filename --------------------------------------------------------
        os.makedirs(path_figures, exist_ok=True)
        suffix = ''
        if add_barbs:
            suffix += '_wind_barbs'
        if convective_axis_deg is not None:
            suffix += f'_along_across_{convective_axis_deg:.0f}deg'
        if add_vort:
            suffix += '_vort'
        if x_axis=='distance':
            suffix += '_dist'

        fname = f'curtain_pass_{flighti}{suffix}.png'
        plt.savefig(path_figures + fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')


# ---------------------------------------------------------------------------
# Figure: grouped bar chart of rainfall classification statistics
# ---------------------------------------------------------------------------

def plot_tdr_statistics():
    """
    Two-row figure:
      Row 1 (a–e): grouped bar charts of TDR rainfall classification percentages
      Row 2 (f–g): total-echo and convective area-vs-height profiles from calc_tdr_size

    Saves to figures/tdr_statistics.png
    """
    # --- font sizes -----------------------------------------------------------
    label_fs = 13.
    tick_fs  = 11.

    # =========================================================================
    # Row 1: classification bar charts
    # =========================================================================
    categories = ['Weak\nEcho', 'Stratiform', 'Shallow\nConvection',
                  'Moderate\nConvection', 'Deep\nConvection']

    data = np.array([
        [18.7, 13.3, 14.7],
        [56.0, 70.6, 74.1],
        [12.8,  8.6,  7.5],
        [12.0,  6.4,  3.7],
        [ 0.5,  1.0,  0.0],
    ])

    cat_colors = ['black',    # Weak Echo
                  '#4D8CFF',  # Lighter blue  - Stratiform
                  '#C7C7C7',  # Lighter grey  - Shallow Convection
                  '#FF8C00',  # Vibrant orange- Moderate Convection
                  '#FFFF00']  # Bright yellow - Deep Convection
    pass_labels = ['Pass 1', 'Pass 2', 'Pass 3']
    n_passes    = len(pass_labels)
    x           = np.arange(n_passes)
    bar_width   = 0.45   # narrower → gaps between bars without touching

    # =========================================================================
    # Row 2: area-vs-height data from TDR plan-view files
    # =========================================================================
    passes = {
        'Pass 1': ('240715I1_1804_xy.nc', 0),
        'Pass 2': ('240715I1_1919_xy.nc', 2),
        'Pass 3': ('240715I1_2030_xy.nc', 4),
    }
    pass_colors  = {'Pass 1': 'steelblue', 'Pass 2': 'darkorange', 'Pass 3': 'firebrick'}
    CONV_VALS    = {2, 3, 4}
    STRAT_VALS   = {0, 1}
    PIXEL_AREA   = 4.0    # km² (2 km resolution)
    MISSING      = -999.9

    path_tdr_pv  = path_data + 'tdr/plan-view/'
    path_cls     = path_data + 'tdr/precip-classifications/'
    class_ds     = xr.open_dataset(path_cls + 'tdr_aew_rainfall_classifications.nc')

    area_results = {}
    for pass_name, (fname, ci) in passes.items():
        ds      = xr.open_dataset(path_tdr_pv + fname)
        refl    = ds['REFLECTIVITY'].values   # (x, y, level, time)
        levels  = ds['level'].values
        ds.close()

        cls2d      = class_ds['rainfall_classification'].isel(num_cases=ci).values
        conv_mask  = np.isin(cls2d, list(CONV_VALS))
        strat_mask = np.isin(cls2d, list(STRAT_VALS))

        areas_total = np.full(len(levels), np.nan)
        areas_conv  = np.full(len(levels), np.nan)
        areas_strat = np.full(len(levels), np.nan)
        for li in range(len(levels)):
            slab  = refl[:, :, li, 0]
            valid = slab > MISSING
            areas_total[li] = np.sum(valid)               * PIXEL_AREA
            areas_conv[li]  = np.sum(valid & conv_mask)   * PIXEL_AREA
            areas_strat[li] = np.sum(valid & strat_mask)  * PIXEL_AREA

        area_results[pass_name] = {'levels': levels,
                                   'total':  areas_total,
                                   'conv':   areas_conv,
                                   'strat':  areas_strat}
    class_ds.close()

    # =========================================================================
    # Build figure: 2 rows, top row has 5 equal columns, bottom row has 2
    # Use GridSpec with nested specs for clean sizing
    # =========================================================================
    fig = plt.figure(figsize=(13, 8))
    gs_top = GridSpec(1, 5, figure=fig,
                      left=0.06, right=0.98, top=0.95, bottom=0.59,
                      wspace=0.48)
    gs_bot = GridSpec(1, 3, figure=fig,
                      left=0.06, right=0.98, top=0.41, bottom=0.08,
                      wspace=0.35)

    axs_top = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
    ax_f    = fig.add_subplot(gs_bot[0, 0])
    ax_g    = fig.add_subplot(gs_bot[0, 1])
    ax_h    = fig.add_subplot(gs_bot[0, 2])

    # ---- bar panels (a–e) ---------------------------------------------------
    for col, (ax, cat, vals, color, letter) in enumerate(
            zip(axs_top, categories, data, cat_colors, 'abcde')):

        ax.bar(x, vals, width=bar_width, color=color, edgecolor='k', linewidth=0.8)

        ymax = max(vals) * 1.25 if max(vals) > 0 else 1.0
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(pass_labels, fontsize=tick_fs, rotation=30, ha='right')
        ax.tick_params(axis='y', labelsize=tick_fs)

        if col == 0:
            ax.set_ylabel('Coverage (%)', fontsize=label_fs)
        else:
            ax.set_ylabel('')

        # category title: not bold, nudged down to align with panel label
        ax.set_xlabel(cat, fontsize=label_fs, fontweight='normal')

        # value labels above bars
        for xi, val in zip(x, vals):
            ax.text(xi + .08, val + ymax * 0.02, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=tick_fs - 1)

        # panel label: nudged down and right
        ax.text(0.1, 0.99, f'({letter})',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=label_fs,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                          linewidth=1.2, pad=4))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ---- panel (f): total echo area vs height --------------------------------
    for pass_name, res in area_results.items():
        valid = res['total'] > 0
        ax_f.plot(res['total'][valid], res['levels'][valid],
                  color=pass_colors[pass_name], lw=2, label=pass_name)
    ax_f.set_xlabel('Total Cell Area (km²)', fontsize=label_fs)
    ax_f.set_ylabel('Height (km)',     fontsize=label_fs)
    ax_f.tick_params(labelsize=tick_fs)
    ax_f.grid(True, alpha=0.3)
    ax_f.set_xlim(left=0)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.text(0.10, 1.01, '(f)',
              transform=ax_f.transAxes, ha='left', va='top',
              fontsize=label_fs,
              bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                        linewidth=1.2, pad=4))

    # ---- panel (g): convective area vs height --------------------------------
    for pass_name, res in area_results.items():
        valid = res['conv'] > 0
        ax_g.plot(res['conv'][valid], res['levels'][valid],
                  color=pass_colors[pass_name], lw=2, label=pass_name)
    ax_g.set_xlabel('Convective Cell Area (km²)', fontsize=label_fs)
    ax_g.set_ylabel('',               fontsize=label_fs)
    ax_g.tick_params(labelsize=tick_fs)
    ax_g.legend(fontsize=tick_fs)
    ax_g.grid(True, alpha=0.3)
    ax_g.set_xlim(left=0)
    ax_g.spines['top'].set_visible(False)
    ax_g.spines['right'].set_visible(False)
    ax_g.text(0.10, 1.01, '(g)',
              transform=ax_g.transAxes, ha='left', va='top',
              fontsize=label_fs,
              bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                        linewidth=1.2, pad=4))

    # ---- panel (h): stratiform + weak echo area vs height --------------------
    for pass_name, res in area_results.items():
        valid = res['strat'] > 0
        ax_h.plot(res['strat'][valid], res['levels'][valid],
                  color=pass_colors[pass_name], lw=2, label=pass_name)
    ax_h.set_xlabel('Stratiform Cell Area (km²)', fontsize=label_fs)
    ax_h.set_ylabel('',               fontsize=label_fs)
    ax_h.tick_params(labelsize=tick_fs)
    ax_h.grid(True, alpha=0.3)
    ax_h.set_xlim(left=0)
    ax_h.spines['top'].set_visible(False)
    ax_h.spines['right'].set_visible(False)
    ax_h.text(0.10, 1.01, '(h)',
              transform=ax_h.transAxes, ha='left', va='top',
              fontsize=label_fs,
              bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                        linewidth=1.2, pad=4))
    ax_h.set_xticks([5000, 10000, 15000])

    # ---- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'tdr_statistics.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')
