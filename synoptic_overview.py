#%%

import os
import sys
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate as nd_rotate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime


path_project = Path(__file__).parent.parent
path_figures = str(path_project) + "/figures/"
path_code    = str(path_project) + "/code/"
path_sal_tpw = str(path_project) + "/data/sal-tpw/"

# ---------------------------------------------------------------------------
# SAL colormap (1–251) — duplicated from download_and_visualize_sal.py
# ---------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap, Normalize as _Normalize

def _build_sal_cmap():
    _VMIN, _VMAX = 1, 251
    _r = _VMAX - _VMIN
    def _n(v): return (v - _VMIN) / _r
    colors = [
        (_n(  1), (1.000, 1.000, 0.000)),
        (_n( 12), (1.000, 0.647, 0.000)),
        (_n( 25), (1.000, 0.000, 0.000)),
        (_n( 35), (1.000, 0.412, 0.706)),
        (_n( 61), (1.000, 1.000, 1.000)),
        (_n( 62), ( 40/255,  40/255,  40/255)),
        (_n(127), (250/255, 250/255, 250/255)),
        (_n(143), ( 35/255, 190/255,   0/255)),
        (_n(202), (130/255,   0/255,  90/255)),
        (_n(203), (  0/255,  80/255,  50/255)),
        (_n(213), (160/255,  20/255, 200/255)),
        (_n(214), ( 79/255,   5/255, 158/255)),
        (_n(242), ( 15/255, 197/255,  40/255)),
        (_n(251), (0.000, 0.000, 0.000)),
    ]
    return LinearSegmentedColormap.from_list('sal', colors), _Normalize(vmin=_VMIN, vmax=_VMAX)

def _build_tpw_cmap():
    colors = [
        ( 0/75, ( 80/255,  15/255,  15/255)),
        ( 8/75, ( 95/255,  18/255,  55/255)),
        (18/75, ( 70/255,  25/255, 130/255)),
        (28/75, ( 30/255,  50/255, 170/255)),
        (38/75, ( 20/255, 105/255, 200/255)),
        (48/75, ( 20/255, 190/255, 190/255)),
        (54/75, ( 50/255, 230/255,  50/255)),
        (58/75, (255/255, 255/255,   0/255)),
        (63/75, (255/255, 160/255,   0/255)),
        (68/75, (255/255,  40/255,  40/255)),
        (72/75, (220/255,  20/255, 150/255)),
        (75/75, (255/255, 130/255, 220/255)),
    ]
    return LinearSegmentedColormap.from_list('mimic_tpw', colors), _Normalize(vmin=0, vmax=75)

sal_cmap, sal_norm = _build_sal_cmap()
tpw_cmap, tpw_norm = _build_tpw_cmap()


def calc_latlon(ds):
    x = ds.x
    y = ds.y
    goes_imager_projection = ds.goes_imager_projection

    x, y = np.meshgrid(x, y)

    r_eq = goes_imager_projection.attrs["semi_major_axis"]
    r_pol = goes_imager_projection.attrs["semi_minor_axis"]
    l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi / 180)
    h_sat = goes_imager_projection.attrs["perspective_point_height"]
    H = r_eq + h_sat

    a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2

    r_s = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)

    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x)**2 + s_y**2))) * (180 / np.pi)
    lon = (l_0 - np.arctan(s_y / (H - s_x))) * (180 / np.pi)

    ds = ds.assign_coords({
        "lat": (["y", "x"], lat),
        "lon": (["y", "x"], lon)
    })
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    return ds


def get_xy_from_latlon(ds, lats, lons):
    lat1, lat2 = lats
    lon1, lon2 = lons

    lat = ds.lat.data
    lon = ds.lon.data

    x = ds.x.data
    y = ds.y.data

    x, y = np.meshgrid(x, y)

    x = x[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]
    y = y[(lat >= lat1) & (lat <= lat2) & (lon >= lon1) & (lon <= lon2)]

    return ((min(x), max(x)), (min(y), max(y)))


def data_subset(data_path, file, lats, lons):
    print('data subset time')
    print(data_path)
    print(file)

    C = xr.open_dataset(data_path + file)

    ds = calc_latlon(C)
    ((x1, x2), (y1, y2)) = get_xy_from_latlon(ds, lats, lons)
    subset = ds.sel(x=slice(x1, x2), y=slice(y2, y1))

    r = subset['CMI_C02'].data; r = np.clip(r, 0, 1)
    g = subset['CMI_C03'].data; g = np.clip(g, 0, 1)
    b = subset['CMI_C01'].data; b = np.clip(b, 0, 1)
    gamma = 2.5; r = np.power(r, 1/gamma); g = np.power(g, 1/gamma); b = np.power(b, 1/gamma)
    g_true = 0.45 * r + 0.1 * g + 0.45 * b
    g_true = np.clip(g_true, 0, 1)
    rgb = np.dstack((r, g_true, b))
    rgb_veggie = np.dstack((r, g, b))

    subset = subset.assign_coords(rgb_channel=['r', 'g', 'b'])
    subset['rgb'] = (('y', 'x', 'rgb_channel'), rgb)
    subset['rgb_veggie'] = (('y', 'x', 'rgb_channel'), rgb_veggie)

    return subset


def pull_files_helper(data_path=''):
    file_names = []
    for (dirpath, dirnames, file) in os.walk(data_path):
        file_names.extend(file)
        break
    return sorted([f for f in file_names if f.endswith('.nc')])


def dt_to_string(t):
    ts = pd.to_datetime(str(t))
    d = ts.strftime('%Y-%m-%d %H:%M:%S.%fZ')
    d = d[:-8] + ' Z'
    return d


def dt_to_time_label(t):
    """Format a scan end-time as 'M/D HH:MM UTC', ceiling to the nearest minute."""
    ts = pd.Timestamp(str(t)).ceil('min')
    return ts.strftime('%-m/%-d %H:%M UTC')


def contrast_correction(color, contrast):
    F = (259 * (contrast + 255)) / (255. * 259 - contrast)
    COLOR = F * (color - .5) + .5
    COLOR = np.clip(COLOR, 0, 1)
    return COLOR


def build_custom_cmap():
    spectral_colors = [
        (158/255, 1/255, 66/255),
        (213/255, 62/255, 79/255),
        (244/255, 109/255, 67/255),
        (253/255, 174/255, 97/255),
        (254/255, 224/255, 139/255),
        (255/255, 255/255, 191/255),
        (230/255, 245/255, 152/255),
        (171/255, 221/255, 164/255),
        (102/255, 194/255, 165/255),
        (50/255, 136/255, 189/255),
        (94/255, 79/255, 162/255),
        (120/255, 140/255, 200/255),
        (180/255, 200/255, 230/255),
        (255/255, 255/255, 255/255),
    ]
    grey_colors = [
        (255/255, 255/255, 255/255),
        (240/255, 240/255, 240/255),
        (217/255, 217/255, 217/255),
        (189/255, 189/255, 189/255),
        (150/255, 150/255, 150/255),
        (115/255, 115/255, 115/255),
        (82/255, 82/255, 82/255),
        (37/255, 37/255, 37/255),
        (0/255, 0/255, 0/255),
    ]

    total_range = 330 - 163
    transition_point = (245 - 163) / total_range

    spectral_positions = np.linspace(0, transition_point, len(spectral_colors))
    grey_positions = np.linspace(transition_point, 1.0, len(grey_colors))

    all_colors = spectral_colors + grey_colors
    all_positions = np.concatenate([spectral_positions, grey_positions])

    colors_and_positions = list(zip(all_positions, all_colors))
    combined_cmap = LinearSegmentedColormap.from_list('spectral_grey', colors_and_positions)
    return combined_cmap


def id_mt_itcz_aews(input_time, use_raw_aew_tracks=False):
    path = "/Users/ethanmurray/files-research-postdoc/papers/2026-magpie-convection/2026-magpie-figures/data/itcz-mt-centers-will-downs/"
    centers = xr.open_dataset(path + 'itcz_mt_centers.nc')
    tracks = xr.open_dataset(path + 'tew_tracks_include_Africa_2024.nc')

    time_series = pd.to_datetime(centers.time)
    time_diffs = np.abs(time_series - pd.Timestamp(input_time))
    nearest_idx = np.argmin(time_diffs)
    nearest_time = time_series[nearest_idx]

    mask = np.zeros(len(time_series), dtype=bool)
    mask[nearest_idx] = True
    centers_trimmed = centers.isel(time=mask)

    print(f'Nearest MT / ITCZ Time: \nBase: {input_time},\nNearest: {nearest_time}')

    itczs = [centers_trimmed.longitude.values, centers_trimmed.itcz_latitude.values[0]]
    mts = [centers_trimmed.longitude.values, centers_trimmed.mt_latitude.values[0]]

    time_series = pd.to_datetime(tracks.time)
    time_diffs = np.abs(time_series - pd.Timestamp(input_time))
    nearest_idx = np.argmin(time_diffs)
    nearest_time = time_series[nearest_idx]

    mask = np.zeros(len(time_series), dtype=bool)
    mask[nearest_idx] = True
    tracks_trimmed = tracks.isel(time=mask)

    print(f'Nearest AEW Time: \nBase: {input_time},\nNearest: {nearest_time}')

    track_bts = tracks_trimmed.ir_mean_400km.values
    if use_raw_aew_tracks:
        track_lats, track_lons = tracks_trimmed.smooth_center_lat.values, tracks_trimmed.smooth_center_lon.values
    else:
        track_lats, track_lons = tracks_trimmed.raw_center_lat.values, tracks_trimmed.raw_center_lon.values
    aews = [track_lons, track_lats, track_bts]

    return mts, itczs, aews


def _panel_aspect(lons, lats):
    """Return the width/height aspect ratio for a lat/lon domain."""
    lon_range = lons[1] - lons[0]
    lat_range = lats[1] - lats[0]
    lat_center = (lats[0] + lats[1]) / 2
    return (lon_range * np.cos(np.radians(lat_center))) / lat_range


def test_plot(ds, channel='rgb', lats=[], lons=[], fill_with_ir=False,
              big_font=False, add_itcz_mt_aews=False,
              # --- axes injection (for combined figure) ---
              ax=None, fig=None,
              # --- publication options ---
              show_colorbar=True, show_lat=True, show_lon=True,
              show_title_banner=True, show_time=True,
              panel_label=None, cax=None,
              label_x=0.02, label_y=0.97,
              time_x=0.98, time_y=0.97,
              legend_loc='upper left'):
    """
    Plot a GOES satellite panel.

    Standalone usage (ax=None): creates and returns its own fig, ax.
    Combined usage: pass existing ax and fig; fig/ax are returned unchanged
    but decorated with the satellite image and annotations.

    Publication options (all default to existing standalone behaviour):
      show_colorbar    – draw the IR colorbar (IR channel only)
      show_latlon      – show lat/lon tick labels and axis labels
      show_title_banner – show the GOES / channel title text above the panel
      show_time        – overlay scan-time string inside the upper portion of the panel
      panel_label      – single character placed in upper-left corner, e.g. 'a'
    """

    title_fs = plt.rcParams['axes.labelsize']
    label_fs = plt.rcParams['axes.labelsize']
    tick_fs  = plt.rcParams['xtick.labelsize']

    # ---- create standalone fig/ax if none provided -------------------------
    if ax is None:
        aspect_ratio = _panel_aspect(lons, lats)
        base_height = 3.5 if big_font else 9
        figsize = (base_height * aspect_ratio, base_height)
        fig = plt.figure(figsize=figsize)
        ax  = plt.axes(projection=ccrs.PlateCarree())

    # title layout flag – kept for standalone two-line title
    two_tits = big_font

    # ---- satellite imagery -------------------------------------------------
    if channel == 'ir':
        cleanIR = ds['CMI_C13'].data
        combined_cmap = build_custom_cmap()

        x, y = ds.lon.values, ds.lat.values
        x = np.where(np.isfinite(x), x, 0)
        y = np.where(np.isfinite(y), y, 0)
        coord_mask = np.isfinite(x) & np.isfinite(y)
        cleanIR = np.where(coord_mask, cleanIR, np.nan)

        cleanIR = cleanIR - 273.15   # K → °C
        p = ax.pcolormesh(x, y, cleanIR, transform=ccrs.PlateCarree(),
                          vmin=190. - 273.15, vmax=305. - 273.15, cmap=combined_cmap)

        if show_colorbar:
            tick_values = np.arange(-110, 36, 20)   # °C ticks matching original K spacing
            if cax is not None:
                cbar = fig.colorbar(p, cax=cax, ticks=tick_values, extend='both')
                cbar.set_label('IR $T_b$ (°C)')
            else:
                cbar = fig.colorbar(p, ax=ax, ticks=tick_values, extend='both',
                                    label='IR $T_b$ (°C)', shrink=0.9)
            cbar.ax.invert_yaxis()

    elif fill_with_ir:
        cleanIR = ds['CMI_C13'].data
        cleanIR = (cleanIR - 90) / (313 - 90)
        cleanIR = np.clip(cleanIR, 0, 1)
        cleanIR = 1 - cleanIR
        cleanIR[cleanIR > .15] += .1
        cleanIR[cleanIR > 1] = 1.

        RGB_contrast = contrast_correction(ds[channel], 0)
        RGB_contrast_IR = np.dstack([np.maximum(RGB_contrast[:, :, 0], cleanIR),
                                     np.maximum(RGB_contrast[:, :, 1], cleanIR),
                                     np.maximum(RGB_contrast[:, :, 2], cleanIR)])
        RGB_contrast_IR = np.nan_to_num(RGB_contrast_IR, nan=0.0)
        RGB_contrast_IR = np.clip(RGB_contrast_IR, 0.0, 1.0)

        x, y = ds.lon.values, ds.lat.values
        x = np.where(np.isfinite(x), x, 0)
        y = np.where(np.isfinite(y), y, 0)
        coord_mask = np.isfinite(x) & np.isfinite(y)
        RGB_contrast_IR = np.where(
            np.stack([coord_mask, coord_mask, coord_mask], axis=-1),
            RGB_contrast_IR, np.nan)
        ax.pcolormesh(x, y, RGB_contrast_IR, transform=ccrs.PlateCarree())

    else:
        RGB_contrast = ds[channel]
        RGB_contrast = np.nan_to_num(RGB_contrast, nan=0.0)
        RGB_contrast = np.clip(RGB_contrast, 0.0, 1.0)

        x, y = ds.lon.values, ds.lat.values
        x = np.where(np.isfinite(x), x, 0)
        y = np.where(np.isfinite(y), y, 0)
        coord_mask = np.isfinite(x) & np.isfinite(y)
        RGB_contrast = np.where(
            np.stack([coord_mask, coord_mask, coord_mask], axis=-1),
            RGB_contrast, np.nan)
        ax.pcolormesh(x, y, RGB_contrast, transform=ccrs.PlateCarree())

    # ---- ITCZ / MT / AEW overlays -----------------------------------------
    if add_itcz_mt_aews:
        mts, itczs, aews = id_mt_itcz_aews(ds.time_bounds.values[0])
        combined_cmap = build_custom_cmap()

        # plot aews as a vertical line, not a star...
        # build points for line here- vertical on one lon

        # print(aews)
        # print(aews[0])
        # print(aews[1])

        # cycle through aew objects for plotting
        for aewi in range(len(aews[0])):
            point1, point2 = (aews[0][aewi], aews[1][aewi] + 4), (aews[0][aewi], aews[1][aewi] - 4)
            if aewi==0:
                ax.plot(point1, point2, c='w', # transform=ccrs.PlateCarree(),
                        lw=5., ls='--', label='AEWs')
            else:
                ax.plot(point1, point2, c='w', # transform=ccrs.PlateCarree(),
                        lw=5., ls='--')

            print((point1, point2))

        # keep aew star as center point, but make smaller
        ax.scatter(aews[0], aews[1], c=aews[2], transform=ccrs.PlateCarree(),
                   cmap=combined_cmap, edgecolors='w',
                   s=50, marker='*', vmin=190., vmax=305.) #, label='AEW Centers')
        
        lw = 2.
        ax.plot(mts[0],   mts[1],   lw=lw, transform=ccrs.PlateCarree(),
                c='darkorange',    label='MT')
        ax.plot(itczs[0], itczs[1], lw=lw, transform=ccrs.PlateCarree(),
                c='cornflowerblue', label='ITCZ')

        # convection domain box
        lonstemp = (-56.5, -51.5)
        latstemp = (8, 13)
        bx = [lonstemp[0], lonstemp[0], lonstemp[1], lonstemp[1], lonstemp[0]]
        by = [latstemp[0], latstemp[1], latstemp[1], latstemp[0], latstemp[0]]
        ax.plot(bx, by, color='limegreen', ls='--', linewidth=lw,
                transform=ccrs.PlateCarree(), label='Domain 1')

        lonstemp = (-60., -50)
        latstemp = (6, 16)
        bx = [lonstemp[0], lonstemp[0], lonstemp[1], lonstemp[1], lonstemp[0]]
        by = [latstemp[0], latstemp[1], latstemp[1], latstemp[0], latstemp[0]]
        ax.plot(bx, by, color='w', ls='--', linewidth=lw, # crimson limegreen
                transform=ccrs.PlateCarree(), label='Domain 2')

        ax.legend(loc=legend_loc, fancybox=False, shadow=False,
                  facecolor='lightgrey', edgecolor='k', framealpha=1, ncols=3,
                  )

    # ---- map features ------------------------------------------------------
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, edgecolor='w')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')

    pc = ccrs.PlateCarree()
    # if not big_font:
    #     ax.gridlines(crs=pc, draw_labels=False, linewidth=0.6, color='0.6', linestyle='--')

    tick_count = 5
    lon_ticks = np.linspace(lons[0], lons[1], tick_count)
    lat_ticks = np.linspace(lats[0], lats[1], tick_count)

    ax.set_xticks(lon_ticks, crs=pc)
    ax.set_yticks(lat_ticks, crs=pc)

    if show_lat:
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.1f', degree_symbol='°'))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.1f', degree_symbol='°'))
        ax.tick_params(axis='both', which='both', labelsize=tick_fs,
                       top=False, right=False, direction='out')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        y_label_pad = 5 if big_font else 0
        ax.set_ylabel("Latitude ($\degree$)",  labelpad=y_label_pad, fontsize=label_fs)
    elif show_lon:
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.1f', degree_symbol='°'))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.1f', degree_symbol='°'))
        ax.tick_params(axis='both', which='both', labelsize=tick_fs,
                       top=False, right=False, direction='out')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        x_label_pad = 5 if big_font else 0
        ax.set_xlabel("Longitude ($\degree$)", labelpad=x_label_pad, fontsize=label_fs)
    else:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.tick_params(axis='both', which='both', labelsize=0,
                       top=False, right=False, direction='out')
        ax.set_xlabel('')
        ax.set_ylabel('')

    # ---- time strings (used by title banner and in-panel annotation) -------
    date           = dt_to_string(ds.time_bounds.values[0])[:11]
    time_start_tit = dt_to_string(ds.time_bounds.values[0])[11:].strip()
    time_end_tit   = dt_to_string(ds.time_bounds.values[1])[11:].strip()

    # ---- optional title banner above the panel (standalone mode) ----------
    if show_title_banner:
        if channel == 'ir':
            titlename = 'Clean IR'
        elif channel == 'rgb':
            titlename = 'RGB'
        elif channel == 'rgb_veggie':
            titlename = 'RGB Veggie'
        if fill_with_ir:
            titlename += ' IR Blend'

        if two_tits:
            ax.text(x=0., y=1.205, s=f'GOES-16 {titlename}',
                    fontsize=title_fs, transform=ax.transAxes)
            ax.text(x=0., y=1.12,  s=f'{date}',
                    fontsize=title_fs, transform=ax.transAxes)
            ax.text(x=0., y=1.035, s=f'{time_start_tit} – {time_end_tit}',
                    fontsize=title_fs, transform=ax.transAxes)
        else:
            time_start = dt_to_string(ds.time_bounds.values[0])
            time_end   = dt_to_string(ds.time_bounds.values[1])[11:]
            ax.set_title(f'GOES-16 {titlename}, {time_start} – {time_end}',
                         fontsize=title_fs)

    # ---- in-panel UTC time annotation (upper-right) -----------------------
    if show_time:
        time_str = dt_to_time_label(ds.time_bounds.values[1])
        ax.text(time_x, time_y, time_str,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=tick_fs,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k', linewidth=1.2, pad=4))

    # ---- panel label (upper area) -----------------------------------------
    if panel_label is not None:
        ax.text(label_x, label_y, f'({panel_label})',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=label_fs,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k', linewidth=1.2, pad=4))

    return fig, ax


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------

def goes_plots_simple(big_font=False, fill_with_ir=True,
                      data_path='',
                      savepath=path_figures + "goes/",
                      savename='',
                      lons=(-56.5, -51.5), lats=(8, 13), ch='rgb',
                      add_itcz_mt_aews=False):

    file_names = pull_files_helper(data_path)

    print(data_path)
    print(file_names)

    for filei, file in enumerate(file_names):
        ds = data_subset(data_path, file, lats, lons)
        print(f'Data processed for file {filei}, {file}')

        fig, ax = test_plot(ds, channel=ch, lats=lats, lons=lons,
                            fill_with_ir=fill_with_ir, big_font=big_font,
                            add_itcz_mt_aews=add_itcz_mt_aews)

        # build save filename
        if savename:
            base, ext = os.path.splitext(savename)
            cur_savename = f"{base}_{filei}{ext}" if len(file_names) > 1 else savename
        else:
            cur_savename = file[:-3] + '.png'

        os.makedirs(savepath, exist_ok=True)
        plt.savefig(savepath + cur_savename, dpi=300., bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {cur_savename}')


# ---------------------------------------------------------------------------
# Top-level figure functions
# ---------------------------------------------------------------------------

def synoptic():
    path_data = str(path_project) + "/data/goes-level2-synoptic/"
    goes_plots_simple(big_font=True, fill_with_ir=False,
                      data_path=path_data,
                      savepath=path_figures,
                      savename='synoptic.png',
                      lons=(-70, -10), lats=(0, 25), ch='ir', # lons=(-70, -5),
                      add_itcz_mt_aews=True)


def convection():
    path_data = str(path_project) + "/data/goes-level2-convection/"
    goes_plots_simple(big_font=True, fill_with_ir=False,
                      data_path=path_data,
                      savepath=path_figures,
                      savename='convection.png',
                      lons=(-56.5, -51.5), lats=(8, 13), ch='ir',
                      add_itcz_mt_aews=False)


def plot_sal_rgb(ax, fig, ds_syn, sal_ds, lats, lons,
                panel_label=None, cax=None,
                label_x=0.02, label_y=0.92,
                time_x=0.97, time_y=0.92,
                draw_domain_boxes=True):
    """Panel b: GOES RGB background with SAL dust features overlaid."""
    pc = ccrs.PlateCarree()

    # RGB base layer — reuse test_plot for the GOES imagery + map features
    test_plot(ds_syn, channel='rgb', lats=lats, lons=lons,
              ax=ax, fig=fig,
              show_colorbar=False, show_lat=True, show_lon=False,
              show_title_banner=False, show_time=False,
              panel_label=panel_label,
              label_x=label_x, label_y=label_y,
              time_x=time_x,   time_y=time_y)

    # SAL overlay — only dust features (values < 61); NaN cells are transparent
    # so the RGB base shows through everywhere else
    sal_data = sal_ds.data.isel(time=0).where(sal_ds.data.isel(time=0) < 61.)
    p = ax.pcolormesh(sal_ds.lon, sal_ds.lat, sal_data,
                      cmap=sal_cmap, norm=sal_norm,
                      transform=pc, zorder=3)

    # if cax is not None:
    #     cbar = fig.colorbar(p, cax=cax)
    #     cbar.set_label('SAL index')


    label_fs = plt.rcParams['axes.labelsize']
    if panel_label is not None:
        ax.text(label_x, label_y, f'({panel_label})',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=label_fs,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k', linewidth=1.2, pad=4))


    if draw_domain_boxes:
        lw = 2.
        lonstemp = (-56.5, -51.5)
        latstemp = (8, 13)
        bx = [lonstemp[0], lonstemp[0], lonstemp[1], lonstemp[1], lonstemp[0]]
        by = [latstemp[0], latstemp[1], latstemp[1], latstemp[0], latstemp[0]]
        ax.plot(bx, by, color='limegreen', ls='--', linewidth=lw,
                transform=ccrs.PlateCarree(), label='Domain 1', zorder=1000000.)

        lonstemp = (-60., -50)
        latstemp = (6, 16)
        bx = [lonstemp[0], lonstemp[0], lonstemp[1], lonstemp[1], lonstemp[0]]
        by = [latstemp[0], latstemp[1], latstemp[1], latstemp[0], latstemp[0]]
        ax.plot(bx, by, color='w', ls='--', linewidth=lw,
                transform=ccrs.PlateCarree(), label='Domain 2', zorder=1000000.)
        

def plot_tpw(ax, fig, tpw_ds, lats, lons,
             panel_label=None, cax=None,
             label_x=0.02, label_y=0.92,
             time_x=0.97, time_y=0.92):
    """Panel c: MIMIC-TPW."""
    pc = ccrs.PlateCarree()
    tick_fs = plt.rcParams['xtick.labelsize']
    label_fs = plt.rcParams['axes.labelsize']

    p = ax.pcolormesh(tpw_ds.lonArr, tpw_ds.latArr, tpw_ds.tpwGrid,
                      cmap=tpw_cmap, norm=tpw_norm,
                      transform=pc)

    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=pc)
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.set_xticks(np.linspace(lons[0], lons[1], 5), crs=pc)
    ax.set_yticks(np.linspace(lats[0], lats[1], 5), crs=pc)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.1f', degree_symbol='°'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.1f', degree_symbol='°'))
    ax.tick_params(axis='both', which='both', labelsize=tick_fs,
                   top=False, right=False, direction='out')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_ylabel("Latitude ($\degree$)",  fontsize=label_fs)
    ax.set_xlabel("Longitude ($\degree$)", fontsize=label_fs)

    if cax is not None:
        cbar = fig.colorbar(p, cax=cax)
        cbar.set_label('TPW (mm)')

    if panel_label is not None:
        ax.text(label_x, label_y, f'({panel_label})',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=label_fs,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k', linewidth=1.2, pad=4))

    # 2 domain boxes here
    lw = 2.
    lonstemp = (-56.5, -51.5)
    latstemp = (8, 13)
    bx = [lonstemp[0], lonstemp[0], lonstemp[1], lonstemp[1], lonstemp[0]]
    by = [latstemp[0], latstemp[1], latstemp[1], latstemp[0], latstemp[0]]
    ax.plot(bx, by, color='limegreen', ls='--', linewidth=lw,
            transform=ccrs.PlateCarree(), label='Domain 1')

    lonstemp = (-60., -50)
    latstemp = (6, 16)
    bx = [lonstemp[0], lonstemp[0], lonstemp[1], lonstemp[1], lonstemp[0]]
    by = [latstemp[0], latstemp[1], latstemp[1], latstemp[0], latstemp[0]]
    ax.plot(bx, by, color='w', ls='--', linewidth=lw, # crimson limegreen
            transform=ccrs.PlateCarree(), label='Domain 2')


def combined_figure():
    """
    Builds a single publication-ready figure:
      3 rows × 2 columns.
      Left column  (synoptic domain, lons_syn / lats_syn):
        (a) GOES IR
        (b) GOES RGB + SAL dust overlay
        (c) MIMIC-TPW
      Right column (convective zoom domain, lons_conv / lats_conv):
        (d) GOES IR — zoom
        (e) GOES RGB + SAL — zoom
        (f) MIMIC-TPW — zoom
      All panels use the 18 UTC snapshot.
    """
    lons_syn  = (-70, -10);        lats_syn  = (0, 19)
    lons_conv = (-56.5, -51.5);   lats_conv = (8, 13)

    path_data_syn  = str(path_project) + "/data/goes-level2-synoptic/"
    path_data_conv = str(path_project) + "/data/goes-level2-convection/"

    # load datasets — single 18 UTC snapshot for both columns
    syn_files  = pull_files_helper(path_data_syn)
    conv_files = pull_files_helper(path_data_conv)

    # conv_files[2] is the ~17:50 UTC (18 UTC) snapshot
    print(f'Loading synoptic:   {syn_files[0]}')
    print(f'Loading convection: {conv_files[2]}')

    ds_syn  = data_subset(path_data_syn,  syn_files[0],  lats_syn,  lons_syn)
    ds_conv = data_subset(path_data_conv, conv_files[2], lats_conv, lons_conv)

    sal_ds = xr.open_dataset(path_sal_tpw + '20240715.18.SAL.Split-Window.nc')
    tpw_ds = xr.open_dataset(path_sal_tpw + 'MIMIC-TPW_20240715.180000.nc')

    # ---- figure sizing -------------------------------------------------------
    asp_syn  = _panel_aspect(lons_syn,  lats_syn)
    asp_conv = _panel_aspect(lons_conv, lats_conv)

    # row height chosen so the synoptic column fills a comfortable width
    row_h      = 3.5                            # inches per row
    col_w_syn  = row_h * asp_syn                # left column width
    col_w_conv = row_h * asp_conv               # right column width
    fig_width  = col_w_syn + col_w_conv
    fig_height = row_h * 3

    print(f'Figure size: {fig_width:.2f} × {fig_height:.2f} in')

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(3, 2,
                           width_ratios=[col_w_syn, col_w_conv],
                           hspace=0.30, wspace=0.05)

    ax_syn  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_sal  = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_tpw  = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    ax_ir_z = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_sal_z= fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax_tpw_z= fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())

    # ---- panel (a): synoptic IR — no colorbar --------------------------------
    test_plot(ds_syn, channel='ir', lats=lats_syn, lons=lons_syn,
              big_font=True, add_itcz_mt_aews=True,
              ax=ax_syn, fig=fig,
              show_colorbar=False,
              show_lat=True, show_lon=False,
              show_title_banner=False, show_time=True,
              panel_label='a',
              label_x=0.02, label_y=0.92,
              time_x=0.97, time_y=0.92,
              legend_loc='lower right')

    # ---- panel (b): zoom IR — no time label, no lat ylabel -------------------
    cax_ir_z = ax_ir_z.inset_axes([1.03, 0.05, 0.055, 0.90])
    test_plot(ds_conv, channel='ir', lats=lats_conv, lons=lons_conv,
              big_font=True,
              ax=ax_ir_z, fig=fig,
              show_colorbar=True, cax=cax_ir_z,
              show_lat=True, show_lon=False,
              show_title_banner=False, show_time=False,
              panel_label='b',
              label_x=0.05, label_y=0.95,
              time_x=0.95, time_y=0.95)
    ax_ir_z.set_ylabel('')

    # ---- panel (c): synoptic RGB + SAL overlay (with domain boxes) -----------
    plot_sal_rgb(ax_sal, fig, ds_syn, sal_ds,
                 lats=lats_syn, lons=lons_syn,
                 panel_label='c',
                 cax=None,
                 label_x=0.02, label_y=0.92,
                 time_x=0.97, time_y=0.92,
                 draw_domain_boxes=True)

    # ---- panel (d): zoom RGB + SAL — no boxes, no lat ylabel -----------------
    plot_sal_rgb(ax_sal_z, fig, ds_conv, sal_ds,
                 lats=lats_conv, lons=lons_conv,
                 panel_label='d',
                 cax=None,
                 label_x=0.05, label_y=0.95,
                 time_x=0.95, time_y=0.95,
                 draw_domain_boxes=False)
    ax_sal_z.set_ylabel('')

    # ---- panel (e): synoptic TPW — no colorbar -------------------------------
    plot_tpw(ax_tpw, fig, tpw_ds,
             lats=lats_syn, lons=lons_syn,
             panel_label='e',
             cax=None,
             label_x=0.02, label_y=0.92,
             time_x=0.97, time_y=0.92)

    # ---- panel (f): zoom TPW — no lat ylabel ---------------------------------
    cax_tpw_z = ax_tpw_z.inset_axes([1.03, 0.05, 0.055, 0.90])
    plot_tpw(ax_tpw_z, fig, tpw_ds,
             lats=lats_conv, lons=lons_conv,
             panel_label='f',
             cax=cax_tpw_z,
             label_x=0.05, label_y=0.95,
             time_x=0.95, time_y=0.95)
    ax_tpw_z.set_ylabel('')

    # ---- save ----------------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + 'combined_overview.png'
    plt.savefig(outfile, dpi=170., bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')




def overlay_flight_data(ax, goes_time, lons, lats,
                        sonde_indices=None, sonde_labels=None,
                        fl_interval_min=5, linec='k', drop_fs=10, label_color=None):
    """
    Overlay P-3 flight data on an existing cartopy GeoAxes:
      - Aircraft symbol at GOES scan time
      - Flight-level wind barbs every fl_interval_min minutes up to goes_time
        (only data before the plane's position at that time)
      - Dropsonde circles + number labels at specified sonde indices

    Parameters
    ----------
    ax              : cartopy GeoAxes to draw on
    goes_time       : numpy datetime64 of the GOES scan end time (used as cutoff)
    lons, lats      : (lo, hi) domain bounds for in-bounds checks
    sonde_indices   : list of int, 0-based indices into sorted dropsonde file list
    sonde_labels    : list of labels to annotate at sonde locations (defaults to indices)
    fl_interval_min : minutes between plotted wind barb positions (default 5)
    linec           : colour for all overlaid elements ('k' or 'w')
    drop_fs         : font size for sonde number labels
    label_color     : colour for the sonde number labels only; defaults to
        ``linec`` if not given
    """
    if label_color is None:
        label_color = linec
    pc    = ccrs.PlateCarree()
    nudge = 0.1   # degree offset for sonde number label

    # --- parse GOES cutoff time (decimal hours UTC) ---------------------------
    goes_ts  = pd.Timestamp(str(goes_time))
    goes_dec = goes_ts.hour + goes_ts.minute / 60. + goes_ts.second / 3600.

    # --- load flight-level data -----------------------------------------------
    fl_path = str(path_project) + "/data/flight-level/20240715I1_A.nc"
    fl_data = xr.open_dataset(fl_path, decode_times=False)


    # # RAF convention: Time in seconds since midnight UTC → convert to decimal hours
    # fl_time_dec = fl_data['Time'].values / 3600.

    # build correct time axis using helper function
    x = build_time_array(fl_data, axis_type='decimal')
    x = xr.DataArray(x, coords={'Time': x})
    # add x as Time array in data -> assign as a coordinate
    fl_data = fl_data.assign_coords(Time=x)
    fl_time_dec = fl_data.Time.values

    # Index of FL data closest to GOES scan end time
    fl_cutoff_i = int(np.nanargmin(np.abs(fl_time_dec - goes_dec)))

    # --- aircraft symbol at GOES time -----------------------------------------
    lat_p3  = float(fl_data['LATref'].values[fl_cutoff_i])
    lon_p3  = float(fl_data['LONref'].values[fl_cutoff_i])
    heading = float(fl_data['THDGref'].values[fl_cutoff_i])

    if (not np.isnan(lat_p3) and not np.isnan(lon_p3) and not np.isnan(heading)
            and lons[0] <= lon_p3 <= lons[1] and lats[0] <= lat_p3 <= lats[1]):
        plane_img_path = str(path_project) + "/data/plane/plane_symbol.png"
        img = mpimg.imread(plane_img_path)
        if linec == 'w':
            non_transparent = img[:, :, 3] > 0
            img = img.copy()
            img[non_transparent, 0:3] = 1.0
        rotated = nd_rotate(img, -heading, reshape=True)
        rotated = np.clip(np.nan_to_num(rotated, nan=0.0), 0.0, 1.0)
        imagebox = OffsetImage(rotated, zoom=0.05) # .025
        ab = AnnotationBbox(imagebox, (lon_p3, lat_p3),
                            frameon=False, box_alignment=(0.5, 0.5),
                            transform=pc)
        ax.add_artist(ab)

    # --- flight-level wind barbs (every fl_interval_min min, up to GOES time) -
    # Assumed 1 Hz data → fl_interval_min * 60 points per interval
    downsample_rate = fl_interval_min * 60

    lats_fl = fl_data['LATref'].values[:fl_cutoff_i]
    lons_fl = fl_data['LONref'].values[:fl_cutoff_i]
    u_fl    = fl_data['UWX.d'].values[:fl_cutoff_i]
    v_fl    = fl_data['UWY.d'].values[:fl_cutoff_i]

    lats_fl = lats_fl[::downsample_rate]
    lons_fl = lons_fl[::downsample_rate]
    u_fl    = u_fl[::downsample_rate] * 1.944   # m/s → knots
    v_fl    = v_fl[::downsample_rate] * 1.944

    # keep only points within the plot domain
    in_domain = ((lons_fl >= lons[0]) & (lons_fl <= lons[1]) &
                 (lats_fl >= lats[0]) & (lats_fl <= lats[1]) &
                 np.isfinite(u_fl) & np.isfinite(v_fl))

    if np.any(in_domain):
        ax.barbs(lons_fl[in_domain], lats_fl[in_domain],
                 u_fl[in_domain], v_fl[in_domain],
                 length=5.75, color=linec, pivot='middle', transform=pc)

    # --- dropsonde circles and number labels ----------------------------------
    if sonde_indices is not None:
        if sonde_labels is None:
            sonde_labels = sonde_indices
        drop_path  = str(path_project) + "/data/dropsondes/"
        drop_files = pull_files_helper(drop_path)

        for si, lbl in zip(sonde_indices, sonde_labels):
            drop    = xr.open_dataset(drop_path + drop_files[si])
            ref_lon = float(drop.reference_lon.values)
            ref_lat = float(drop.reference_lat.values)

            if (lons[0] <= ref_lon <= lons[1] and lats[0] <= ref_lat <= lats[1]):
                ax.scatter(ref_lon, ref_lat, c=linec, marker='o', s=40,
                           transform=pc, zorder=100)
                ax.text(ref_lon + nudge, ref_lat + nudge, str(lbl),
                        color=label_color, fontsize=drop_fs, fontweight='bold',
                        transform=pc, zorder=1000)


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



def combined_cold_pools():
    """
    Figure 2: 5-row × 3-column grid showing cold pool evolution.

      5 rows  = 5 time steps (6Z, 12Z, 16Z, 18Z, 0Z next day)
      3 cols  = channel/domain type:
        col 0: cold pool zoom  (lons_cp  / lats_cp)   — IR  (wider domain, left)
                + green rectangle outlining the tight convective zoom domain
        col 1: tight conv zoom (lons_conv / lats_conv) — IR  (center)
        col 2: tight conv zoom (lons_conv / lats_conv) — RGB (right; blank rows 0 & 4)

    Publication options:
      - Panel labels in upper-left of every active panel
      - UTC scan times in upper-right of every active panel
      - Lat/lon tick labels on col-0 only (bottom row for lon)
      - Cols 1 & 2: no tick labels
      - IR colorbar hanging off right edge of col-1, bottom row (panel m)
      - Green box on col-0 panels showing lons_conv/lats_conv extent
    """
    # domain extents
    lons_cp   = (-60,   -50);    lats_cp   = (6, 16)   # cold pool domain (wider, left col)
    lons_conv = (-56.5, -51.5);  lats_conv = (8, 13)   # tight convective zoom (center/right cols)

    path_data = str(path_project) + "/data/goes-level2-cold pool new/"

    cp_files = pull_files_helper(path_data)
    n_times  = len(cp_files)   # expect 5: 6Z, 12Z, 16Z, 18Z, 0Z
    n_cols   = 3
    n_rows   = n_times         # one row per time step

    print(f'Loading cold pool files: {cp_files}')
    ds_cp_list   = [data_subset(path_data, f, lats_cp,   lons_cp)   for f in cp_files]
    ds_conv_list = [data_subset(path_data, f, lats_conv, lons_conv) for f in cp_files]

    # ---- figure sizing: width_ratios to match geographic aspect ratios ------
    asp_cp   = _panel_aspect(lons_cp,   lats_cp)
    asp_conv = _panel_aspect(lons_conv, lats_conv)

    panel_h   = 2.5                       # inches per row
    w_cp      = panel_h * asp_cp          # width of col 0
    w_conv    = panel_h * asp_conv        # width of cols 1 & 2
    fig_width  = w_cp + 2 * w_conv
    fig_height = panel_h * n_rows

    print(f'CP asp: {asp_cp:.3f}  |  Conv asp: {asp_conv:.3f}')
    print(f'Figure: {fig_width:.2f} × {fig_height:.2f} in')

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(n_rows, n_cols, figure=fig,
                   width_ratios=[w_cp, w_conv, w_conv],
                   hspace=0.05, wspace=0.08)

    # RGB only for middle 3 time steps (rows 1–3); skip rows 0 and 4
    rgb_rows = {1, 2, 3}

    # assign panel labels sequentially over active cells (row-major order)
    active_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)
                    if not (c == 2 and r not in rgb_rows)]
    label_iter = iter('abcdefghijklmnopqrstuvwxyz')
    cell_label = {cell: next(label_iter) for cell in active_cells}

    col_configs = [
        ('ir',  lons_cp,   lats_cp,   ds_cp_list),
        ('ir',  lons_conv, lats_conv, ds_conv_list),
        ('rgb', lons_conv, lats_conv, ds_conv_list),
    ]

    pc = ccrs.PlateCarree()

    # shared grid positions — even degree values covering both domains
    # ticks and gridlines use identical locations so they always coincide
    grid_lats = np.arange(0,  20, 2).astype(float)   # 0, 2, 4, … 18
    grid_lons = np.arange(-62, -49, 2).astype(float)  # -62, -60, -58, … -50

    label_fs = plt.rcParams['axes.labelsize']
    tick_fs  = plt.rcParams['xtick.labelsize']

    col_titles = ['Domain 2, IR', 'Domain 1, IR', 'Domain 1, RGB']

    for col, (channel, lons, lats, ds_list) in enumerate(col_configs):
        for row in range(n_rows):
            # skip RGB panels at 6Z (row 0) and 0Z (row 4)
            if col == 2 and row not in rgb_rows:
                continue

            ax  = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            ds  = ds_list[row]
            lbl = cell_label[(row, col)]

            is_left   = (col == 0)
            is_bottom = (row == n_rows - 1)
            is_panel_m = (col == 1 and is_bottom)

            # IR colorbar on col-1 bottom row (panel m), padded from panel edge
            show_cbar = (col == 1 and is_bottom and channel == 'ir')
            cax = None
            if show_cbar:
                cax = ax.inset_axes([1.06, 0.05, 0.06, 0.90])

            # let test_plot draw the image; suppress its tick logic entirely
            test_plot(ds, channel=channel, lats=lats, lons=lons,
                      ax=ax, fig=fig,
                      show_colorbar=show_cbar, cax=cax,
                      show_lat=False, show_lon=False,
                      show_title_banner=False,
                      show_time=is_left,
                      panel_label=lbl,
                      label_x=0.06, label_y=0.94,
                      time_x=0.94, time_y=0.94)

            # ---- ticks: use grid positions so ticks always sit on gridlines ----
            # strict-interior filter avoids crowded boundary ticks between panels
            panel_lon_ticks = grid_lons[(grid_lons > lons[0]) & (grid_lons < lons[1])]
            panel_lat_ticks = grid_lats[(grid_lats > lats[0]) & (grid_lats < lats[1])]

            ax.set_xticks(panel_lon_ticks, crs=pc)
            ax.set_yticks(panel_lat_ticks, crs=pc)
            ax.tick_params(axis='both', which='both',
                           top=False, right=False, direction='out', labelsize=tick_fs)

            if is_left:
                # lat numbers + ylabel on all col-0 panels
                ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f', degree_symbol='°'))
                ax.set_ylabel("Latitude ($\degree$)", fontsize=label_fs)
                if is_bottom:
                    # lon numbers + xlabel on panel l only
                    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f', degree_symbol='°'))
                    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                    ax.set_xlabel("Longitude ($\degree$)", fontsize=label_fs)
                else:
                    ax.xaxis.set_major_formatter(plt.NullFormatter())
                    ax.set_xlabel('')
            elif is_panel_m:
                # lon numbers + xlabel on panel m; no lat numbers
                ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f', degree_symbol='°'))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                ax.set_xlabel("Longitude ($\degree$)", fontsize=label_fs)
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.set_ylabel('')
            else:
                # tick marks only, no numbers or axis labels
                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())
                ax.set_xlabel('')
                ax.set_ylabel('')

            # add matching gridlines (same locs on all panels)
            ax.gridlines(xlocs=grid_lons, ylocs=grid_lats,
                         draw_labels=False,
                         linewidth=0.5, color='white', alpha=0.5,
                         linestyle='-', zorder=5)

            # col 0: draw thick green box showing the tight zoom domain extent
            if is_left:
                ax.plot(
                    [lons_conv[0], lons_conv[1], lons_conv[1], lons_conv[0], lons_conv[0]],
                    [lats_conv[0], lats_conv[0], lats_conv[1], lats_conv[1], lats_conv[0]],
                    color='limegreen', ls='--', lw=2.0, transform=pc, zorder=10
                )

            # column titles on the top panel of each column
            # col 2 has no row-0 panel so its title goes on row 1
            is_col_title_row = (row == 0 and col < 2) or (row == 1 and col == 2)
            if is_col_title_row:
                ax.set_title(col_titles[col], fontsize=9, fontweight='bold', pad=4)

    # ---- save ----------------------------------------------------------------
    savepath = path_figures
    os.makedirs(savepath, exist_ok=True)
    outfile = savepath + 'combined_cold_pools.png'
    plt.savefig(outfile, dpi=300., bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'Saved {outfile}')

# %%
