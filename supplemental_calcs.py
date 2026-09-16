# additional self contained scripts to calculate relevant quantities about our case study

# import...
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr
import shapely

import sys
sys.path.insert(0, str(Path(__file__).parent))
from synoptic_overview import calc_latlon, get_xy_from_latlon

path_project = Path(__file__).parent.parent
path_tdr = str(path_project) + "/data/tdr/plan-view/"

# TDR horizontal resolution (km) — confirmed from global attribute
TDR_RES_KM = 2.0
PIXEL_AREA_KM2 = TDR_RES_KM ** 2   # 4 km² per pixel

MISSING = -999.9


# calculate the size of the convective region using TDR data! do so at each height level
def calc_tdr_size():
    passes = {
        '1804': ('240715I1_1804_xy.nc', 0),   # (plan-view file, num_cases index)
        '1919': ('240715I1_1919_xy.nc', 2),
        '2030': ('240715I1_2030_xy.nc', 4),
    }
    colors = {'1804': 'steelblue', '1919': 'darkorange', '2030': 'firebrick'}

    # levels to show in the plan-view footprint plots (km)
    plot_levels_km = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    # classification masks:
    #   convective       : values 2 (weak), 3 (moderate), 4 (strong)
    #   stratiform/echo  : values 0 (weak echo), 1 (stratiform)
    CONV_VALS  = {2, 3, 4}
    STRAT_VALS = {0, 1}

    path_cls  = str(path_project) + "/data/tdr/precip-classifications/"
    cls_file  = "tdr_aew_rainfall_classifications.nc"
    class_ds  = xr.open_dataset(path_cls + cls_file)

    # -------------------------------------------------------------------------
    # 1. Calculate and print area at each level for each pass
    # -------------------------------------------------------------------------
    results = {}

    print(f"\n{'Pass':>6}  {'Height (km)':>11}  {'N pix (total)':>14}  "
          f"{'Area total (km²)':>17}  {'Area conv (km²)':>16}  {'Area strat (km²)':>17}")
    print("-" * 90)

    for pass_name, (fname, ci) in passes.items():
        ds    = xr.open_dataset(path_tdr + fname)
        refl  = ds['REFLECTIVITY'].values  # (x, y, level, time)
        levels = ds['level'].values
        x_km  = ds['x'].values
        y_km  = ds['y'].values
        ds.close()

        # classification map for this pass: (northward=250, eastward=250)
        # northward maps to the x-index, eastward to y (matching TDR array layout)
        cls2d = class_ds['rainfall_classification'].isel(num_cases=ci).values  # (x, y)

        conv_mask  = np.isin(cls2d, list(CONV_VALS))   # (x, y) bool
        strat_mask = np.isin(cls2d, list(STRAT_VALS))  # (x, y) bool

        n_levels = len(levels)
        areas_total = np.full(n_levels, np.nan)
        areas_conv  = np.full(n_levels, np.nan)
        areas_strat = np.full(n_levels, np.nan)

        for li in range(n_levels):
            slab  = refl[:, :, li, 0]          # (x, y)
            valid = slab > MISSING

            n_tot   = int(np.sum(valid))
            n_conv  = int(np.sum(valid & conv_mask))
            n_strat = int(np.sum(valid & strat_mask))

            areas_total[li] = n_tot   * PIXEL_AREA_KM2
            areas_conv[li]  = n_conv  * PIXEL_AREA_KM2
            areas_strat[li] = n_strat * PIXEL_AREA_KM2

            print(f"{pass_name:>6}  {levels[li]:>11.1f}  {n_tot:>14d}  "
                  f"{areas_total[li]:>17.1f}  {areas_conv[li]:>16.1f}  {areas_strat[li]:>17.1f}")

        results[pass_name] = {
            'levels':      levels,
            'area_total':  areas_total,
            'area_conv':   areas_conv,
            'area_strat':  areas_strat,
            'refl':        refl,
            'x_km':        x_km,
            'y_km':        y_km,
        }

    class_ds.close()

    # -------------------------------------------------------------------------
    # 2. Area-vs-height plots: (a) total  (b) convective  (c) strat+weak echo
    # -------------------------------------------------------------------------
    def _area_ax(ax, area_key, title):
        for pass_name, res in results.items():
            arr   = res[area_key]
            valid = arr > 0
            ax.plot(arr[valid], res['levels'][valid],
                    color=colors[pass_name], lw=2, label=f'Pass {pass_name}')
        ax.set_xlabel('Cell area (km²)')
        ax.set_ylabel('Height (km)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    fig_a, axes_a = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    _area_ax(axes_a[0], 'area_total', '(a) Total echo area')
    _area_ax(axes_a[1], 'area_conv',  '(b) Convective area\n(weak/mod/strong conv.)')
    _area_ax(axes_a[2], 'area_strat', '(c) Stratiform + weak echo area')
    axes_a[0].set_ylabel('Height (km)')
    for ax in axes_a[1:]:
        ax.set_ylabel('')
    fig_a.tight_layout()

    # -------------------------------------------------------------------------
    # 3. Test plot B — plan-view footprints at selected height levels
    # -------------------------------------------------------------------------
    n_passes = len(passes)
    n_plots  = len(plot_levels_km)
    fig_b, axes = plt.subplots(n_passes, n_plots,
                               figsize=(n_plots * 2.5, n_passes * 2.5),
                               sharex=True, sharey=True)

    for row, (pass_name, res) in enumerate(results.items()):
        levels = res['levels']
        for col, target_lev in enumerate(plot_levels_km):
            ax   = axes[row, col]
            li   = int(np.argmin(np.abs(levels - target_lev)))
            slab = res['refl'][:, :, li, 0]
            mask = slab > MISSING            # True where data exist

            ax.pcolormesh(res['x_km'], res['y_km'], mask.T,
                          cmap='Blues', vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.set_title(f'{target_lev:.0f} km', fontsize=8)

            if col == 0:
                ax.set_ylabel(f'Pass {pass_name}\nN–S (km)', fontsize=7)
            if row == n_passes - 1:
                ax.set_xlabel('E–W (km)', fontsize=7)
            ax.tick_params(labelsize=6)

    fig_b.suptitle('TDR radar footprint (blue = data present)', y=1.01)
    fig_b.tight_layout()

    plt.show()


def calc_tdr_mean_flow():
    """
    For each of the 3 TDR passes, build CFADs (Contoured Frequency by Altitude
    Diagrams) for:
      (a) wind direction  (0–360°, derived from U and V)
      (b) horizontal wind speed  (sqrt(U²+V²))
      (c) vertical velocity  (W)

    Produces two figures:
      1. All passes pooled into a single CFAD, per-pass means overlaid.
      2. One 1×3 CFAD per pass (3 separate figures).
    """
    passes = {
        '1804': '240715I1_1804_xy.nc',
        '1919': '240715I1_1919_xy.nc',
        '2030': '240715I1_2030_xy.nc',
    }
    colors = {'1804': 'steelblue', '1919': 'darkorange', '2030': 'firebrick'}

    # ── tuneable axis limits ──────────────────────────────────────────────────
    xlim_wd = (0,   360)   # wind direction (°)
    xlim_ws = (0,    30)   # horizontal wind speed (m/s)
    xlim_w  = (-6,   16)   # vertical velocity (m/s)

    # ── tuneable CFAD display ─────────────────────────────────────────────────
    MAX_HEIGHT_KM = 13.0   # discard data above this level
    FREQ_VMAX     = 15.0   # colorbar ceiling (%) — lower = more contrast at mid-levels

    # ── CFAD bin edges ────────────────────────────────────────────────────────
    wd_bins = np.linspace(xlim_wd[0], xlim_wd[1], 73)   # 5° bins
    ws_bins = np.linspace(xlim_ws[0], xlim_ws[1], 61)   # 0.5 m/s bins
    w_bins  = np.linspace(xlim_w[0],  xlim_w[1],  89)   # 0.25 m/s bins

    # ── load and compute ──────────────────────────────────────────────────────
    ds0    = xr.open_dataset(path_tdr + list(passes.values())[0])
    levels = ds0['level'].values
    ds0.close()

    lev_mask = levels <= MAX_HEIGHT_KM
    levels   = levels[lev_mask]
    n_lev    = len(levels)

    # per-pass storage
    pass_data = {}   # pass_name -> {'means': {...}, 'cfads': {...}}

    # pooled accumulators
    pool_wd = np.zeros((n_lev, len(wd_bins) - 1))
    pool_ws = np.zeros((n_lev, len(ws_bins) - 1))
    pool_w  = np.zeros((n_lev, len(w_bins)  - 1))

    for pass_name, fname in passes.items():
        ds   = xr.open_dataset(path_tdr + fname)
        U    = ds['U'].values[:, :, lev_mask, :]
        V    = ds['V'].values[:, :, lev_mask, :]
        W    = ds['W'].values[:, :, lev_mask, :]
        refl = ds['REFLECTIVITY'].values[:, :, lev_mask, :]
        ds.close()

        wd_mean = np.full(n_lev, np.nan)
        ws_mean = np.full(n_lev, np.nan)
        w_mean  = np.full(n_lev, np.nan)
        p_wd = np.zeros((n_lev, len(wd_bins) - 1))
        p_ws = np.zeros((n_lev, len(ws_bins) - 1))
        p_w  = np.zeros((n_lev, len(w_bins)  - 1))

        for li in range(n_lev):
            valid = refl[:, :, li, 0] > MISSING
            u_v   = U[:, :, li, 0][valid]
            v_v   = V[:, :, li, 0][valid]
            w_v   = W[:, :, li, 0][valid]
            if len(u_v) == 0:
                continue

            wd_v = np.degrees(np.arctan2(-u_v, -v_v)) % 360
            ws_v = np.sqrt(u_v**2 + v_v**2)

            # unit-vector mean for direction (handles 0/360 wrap)
            wd_mean[li] = np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(wd_v))),
                np.mean(np.cos(np.radians(wd_v))))) % 360
            ws_mean[li] = np.mean(ws_v)
            w_mean[li]  = np.mean(w_v)

            p_wd[li] = np.histogram(wd_v, bins=wd_bins)[0]
            p_ws[li] = np.histogram(ws_v, bins=ws_bins)[0]
            p_w[li]  = np.histogram(w_v,  bins=w_bins) [0]

        pool_wd += p_wd
        pool_ws += p_ws
        pool_w  += p_w

        pass_data[pass_name] = {
            'means': {'wd': wd_mean, 'ws': ws_mean, 'w': w_mean},
            'cfads': {'wd': p_wd,    'ws': p_ws,    'w': p_w},
        }

    # ── row-normalise: zero-count rows → NaN so they render white ────────────
    def _norm_rows(cfad):
        out      = cfad.astype(float).copy()
        row_sums = cfad.sum(axis=1)
        for li in range(out.shape[0]):
            if row_sums[li] == 0:
                out[li] = np.nan
            else:
                out[li] = out[li] / row_sums[li] * 100
        # set true-zero bins to NaN so they render white
        out[out == 0] = np.nan
        return out

    pool_freq = {
        'wd': _norm_rows(pool_wd),
        'ws': _norm_rows(pool_ws),
        'w':  _norm_rows(pool_w),
    }
    for pn in pass_data:
        c = pass_data[pn]['cfads']
        pass_data[pn]['freq'] = {k: _norm_rows(c[k]) for k in c}

    # ── shared plotting helper ────────────────────────────────────────────────
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('white')    # NaN cells render white

    lev_edges = np.append(levels - 0.25, levels[-1] + 0.25)

    def _draw_cfad(fig, ax, freq, bin_edges, xlabel, xlim, mean_profiles):
        pc = ax.pcolormesh(bin_edges, lev_edges, freq,
                           cmap=cmap, shading='flat',
                           vmin=0, vmax=FREQ_VMAX)
        fig.colorbar(pc, ax=ax, label='Frequency (%)', shrink=0.8)
        for pass_name, mean_val, c in mean_profiles:
            ax.plot(mean_val, levels, color=c, lw=2, label=f'Pass {pass_name}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Height (km)')
        ax.set_xlim(xlim)
        ax.set_ylim(0, MAX_HEIGHT_KM)
        ax.grid(True, alpha=0.25, color='k')
        if xlabel.startswith('Vertical'):
            ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.6)

    def _make_fig(title, wd_freq, ws_freq, w_freq, mean_profiles):
        fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
        _draw_cfad(fig, axes[0], wd_freq, wd_bins, 'Wind direction (°)',
                   xlim_wd, mean_profiles)
        axes[0].set_xticks(np.arange(0, 361, 90))

        _draw_cfad(fig, axes[1], ws_freq, ws_bins,
                   'Horizontal wind speed (m s⁻¹)', xlim_ws, mean_profiles)

        _draw_cfad(fig, axes[2], w_freq,  w_bins,
                   'Vertical velocity (m s⁻¹)', xlim_w, mean_profiles)

        for ax, letter in zip(axes, ['a', 'b', 'c']):
            ax.text(0.03, 0.97, f'({letter})', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top',
                    bbox=dict(facecolor='white', edgecolor='k', pad=3))

        axes[2].legend(fontsize=9, loc='lower right')
        fig.suptitle(title, y=1.01)
        fig.tight_layout()

    # ── Figure 1: all passes pooled ───────────────────────────────────────────
    all_means = [(pn, pass_data[pn]['means']['wd'],
                      pass_data[pn]['means']['ws'],
                      pass_data[pn]['means']['w'],
                      colors[pn])
                 for pn in passes]

    _make_fig(
        'TDR wind CFADs — all passes pooled',
        pool_freq['wd'], pool_freq['ws'], pool_freq['w'],
        [(pn, pass_data[pn]['means']['wd'], colors[pn]) for pn in passes],
    )
    # (ws and w means need to be passed per variable)
    # rebuild cleanly — _make_fig takes one mean_profiles list shared for all panels,
    # so re-call with the correct variable:
    plt.close('all')

    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    for var_key, ax, bins, xlabel, xlim in [
        ('wd', axes[0], wd_bins, 'Wind direction (°)',            xlim_wd),
        ('ws', axes[1], ws_bins, 'Horizontal wind speed (m s⁻¹)', xlim_ws),
        ('w',  axes[2], w_bins,  'Vertical velocity (m s⁻¹)',     xlim_w),
    ]:
        means_for_var = [(pn, pass_data[pn]['means'][var_key], colors[pn])
                         for pn in passes]
        _draw_cfad(fig, ax, pool_freq[var_key], bins, xlabel, xlim, means_for_var)

    axes[0].set_xticks(np.arange(0, 361, 90))
    for ax, letter in zip(axes, ['a', 'b', 'c']):
        ax.text(0.03, 0.97, f'({letter})', transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top',
                bbox=dict(facecolor='white', edgecolor='k', pad=3))
    axes[2].legend(fontsize=9, loc='lower right')
    fig.suptitle('TDR wind CFADs — all passes pooled, per-pass means overlaid', y=1.01)
    fig.tight_layout()

    # ── Figures 2-4: one per pass ─────────────────────────────────────────────
    for pass_name in passes:
        c    = colors[pass_name]
        freq = pass_data[pass_name]['freq']
        mean = pass_data[pass_name]['means']

        fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
        for var_key, ax, bins, xlabel, xlim in [
            ('wd', axes[0], wd_bins, 'Wind direction (°)',            xlim_wd),
            ('ws', axes[1], ws_bins, 'Horizontal wind speed (m s⁻¹)', xlim_ws),
            ('w',  axes[2], w_bins,  'Vertical velocity (m s⁻¹)',     xlim_w),
        ]:
            _draw_cfad(fig, ax, freq[var_key], bins, xlabel, xlim,
                       [(pass_name, mean[var_key], c)])

        axes[0].set_xticks(np.arange(0, 361, 90))
        for ax, letter in zip(axes, ['a', 'b', 'c']):
            ax.text(0.03, 0.97, f'({letter})', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top',
                    bbox=dict(facecolor='white', edgecolor='k', pad=3))
        axes[2].legend(fontsize=9, loc='lower right')
        fig.suptitle(f'TDR wind CFADs — Pass {pass_name}', y=1.01)
        fig.tight_layout()

    plt.show()

def find_ir_bt_edges(dataset_path, lats, lons, bt_cutoff=-60, pixel_size_km=2.0):
    """
    Load a GOES L2 dataset, spatially subset it, and identify cold cloud-top pixels.

    Parameters
    ----------
    dataset_path : str
        Full path to the GOES L2 NetCDF file.
    lats : tuple of float
        (lat_min, lat_max) bounding box for spatial trim.
    lons : tuple of float
        (lon_min, lon_max) bounding box for spatial trim.
    bt_cutoff : float
        Brightness temperature threshold (°C). Pixels below this are flagged.
    pixel_size_km : float
        Assumed pixel side length in km (default 2 km for GOES-16 Band 13).

    Returns
    -------
    lat2d : np.ndarray
        2-D latitude array for the trimmed region (y, x).
    lon2d : np.ndarray
        2-D longitude array for the trimmed region (y, x).
    cold_mask : np.ndarray of int
        1/0 mask — 1 where BT < bt_cutoff, 0 elsewhere.
    coverage_km2 : float
        Total area of cold pixels in km².
    """
    ds = xr.open_dataset(dataset_path)
    ds = calc_latlon(ds)

    ((x1, x2), (y1, y2)) = get_xy_from_latlon(ds, lats, lons)
    subset = ds.sel(x=slice(x1, x2), y=slice(y2, y1))

    bt_c = subset['CMI_C13'].data - 273.15   # K → °C

    lat2d = subset['lat'].data
    lon2d = subset['lon'].data

    cold_mask = (bt_c < bt_cutoff).astype(int)

    pixel_area_km2 = pixel_size_km ** 2
    coverage_km2 = float(cold_mask.sum()) * pixel_area_km2

    ds.close()
    return lat2d, lon2d, cold_mask, coverage_km2


def plot_ir_bt_edges():
    import cartopy.crs as ccrs
    from synoptic_overview import calc_latlon, get_xy_from_latlon, build_custom_cmap, pull_files_helper

    path_data = str(path_project) + "/data/goes-level2-cold pool new/" # "/data/goes-level2-tdr/"
    lats = (8, 13)
    lons = (-56.5, -51.5)
    bt_cutoff = -18.15 # -55.0

    files = sorted(pull_files_helper(path_data))

    for fname in files:
        lat2d, lon2d, cold_mask, coverage_km2 = find_ir_bt_edges(
            path_data + fname, lats, lons, bt_cutoff=bt_cutoff
        )
        print(f"{fname}")
        print(f"  Cold pixels (BT < {bt_cutoff}°C): {cold_mask.sum()}  "
              f"Coverage: {coverage_km2:.0f} km²\n")

        # --- load dataset again for IR background ---
        import xarray as xr
        ds = xr.open_dataset(path_data + fname)
        ds = calc_latlon(ds)
        ((x1, x2), (y1, y2)) = get_xy_from_latlon(ds, lats, lons)
        subset = ds.sel(x=slice(x1, x2), y=slice(y2, y1))

        bt_c = subset['CMI_C13'].data - 273.15

        combined_cmap = build_custom_cmap()
        fig = plt.figure(figsize=(6, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())

        x = np.where(np.isfinite(lon2d), lon2d, 0)
        y = np.where(np.isfinite(lat2d), lat2d, 0)
        ax.pcolormesh(x, y, bt_c, transform=ccrs.PlateCarree(),
                      vmin=190. - 273.15, vmax=305. - 273.15,
                      cmap=combined_cmap, zorder=1)

        # overlay cold-cloud outline
        ax.contour(lon2d, lat2d, cold_mask, levels=[0.5],
                   colors='orange', linewidths=1.2,
                   transform=ccrs.PlateCarree(), zorder=2)

        ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', linewidth=0.8)

        time_str = fname.split('_s')[1][:13]   # e.g. 20241971750210 → yydddHHMMss
        ax.set_title(f"GOES IR  |  scan {time_str}  |  BT < {bt_cutoff}°C: {coverage_km2:.0f} km²",
                     fontsize=9)
        fig.tight_layout()
        ds.close()

    plt.show()



def find_tams_info(lats=(8, 13), lons=(-56.5, -51.5),
                   ctt_threshold_c=-54.0, ctt_core_threshold_c=-65.0):
    """
    Apply the TAMS algorithm to GOES IR BT data from the TDR pass times and
    produce a static figure showing how the identified CE boundary evolves.

    Parameters
    ----------
    lats, lons : tuple
        Spatial bounding box for the subset.
    ctt_threshold_c : float
        Outer CE boundary threshold in °C (converted to K for TAMS).
    ctt_core_threshold_c : float
        Cold-core threshold in °C (converted to K for TAMS). Must be colder
        than ctt_threshold_c.
    """
    import tams
    import cartopy.crs as ccrs
    import pandas as pd
    from synoptic_overview import calc_latlon, get_xy_from_latlon, build_custom_cmap, pull_files_helper

    K_OFFSET = 273.15
    ctt_threshold_k      = ctt_threshold_c      + K_OFFSET
    ctt_core_threshold_k = ctt_core_threshold_c + K_OFFSET

    path_data = str(path_project) + "/data/goes-level2-tdr/"
    files = sorted(pull_files_helper(path_data))

    # ── 1. Load and subset each file, build a concatenated BT DataArray ──────
    slices = []
    times  = []
    for fname in files:
        ds = xr.open_dataset(path_data + fname)
        ds = calc_latlon(ds)
        ((x1, x2), (y1, y2)) = get_xy_from_latlon(ds, lats, lons)
        subset = ds.sel(x=slice(x1, x2), y=slice(y2, y1))

        bt_k = subset['CMI_C13'].values.astype(float)   # K

        # parse scan-start time from filename  e.g. s20241971750210 → datetime
        time_str = fname.split('_s')[1][:13]   # '20241971750210'[:13] → '2024197175021'
        t = pd.to_datetime(time_str, format='%Y%j%H%M%S')
        times.append(t)

        lat2d = subset['lat'].values
        lon2d = subset['lon'].values
        ds.close()

        da = xr.DataArray(
            bt_k,
            dims=['y', 'x'],
            coords={
                'lat': (['y', 'x'], lat2d),
                'lon': (['y', 'x'], lon2d),
                'time': t,
            },
        )
        slices.append(da)

    tb = xr.concat(slices, dim='time')
    tb.attrs.update(units='K', long_name='Brightness temperature')

    # ── 2. Run TAMS identify (size_filter=False: domain too small for 4000 km² cut) ──
    contour_sets, _ = tams.identify(
        tb,
        ctt_threshold=ctt_threshold_k,
        ctt_core_threshold=ctt_core_threshold_k,
        size_filter=False,
    )

    # ── 3. Static figure: IR background at each time, CE outlines overlaid ───
    colors = ['#e41a1c', '#377eb8', '#4daf4a']   # red / blue / green per time step
    combined_cmap = build_custom_cmap()
    tran = ccrs.PlateCarree()

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             subplot_kw={'projection': tran})
    if n == 1:
        axes = [axes]

    for i, (ax, fname, t, cs) in enumerate(zip(axes, files, times, contour_sets)):
        # IR background
        bt_c = tb.sel(time=t).values - K_OFFSET
        lat2d = tb.sel(time=t).coords['lat'].values
        lon2d = tb.sel(time=t).coords['lon'].values

        x = np.where(np.isfinite(lon2d), lon2d, 0)
        y = np.where(np.isfinite(lat2d), lat2d, 0)
        ax.pcolormesh(x, y, bt_c, transform=tran,
                      vmin=190. - K_OFFSET, vmax=305. - K_OFFSET,
                      cmap=combined_cmap, zorder=1)

        # CE outlines from TAMS
        if not cs.empty:
            cs.plot(ax=ax, facecolor='none', edgecolor=colors[i % len(colors)],
                    linewidth=1.5, transform=tran, zorder=2)
            print(f"  {t:%Y-%m-%d %H:%MZ}  →  {len(cs)} CE(s) identified")
        else:
            print(f"  {t:%Y-%m-%d %H:%MZ}  →  no CEs above threshold")

        ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=tran)
        ax.coastlines(resolution='10m', linewidth=0.7)
        ax.set_title(f"{t:%H:%MZ}  |  {len(cs)} CE(s)", fontsize=10)

    fig.suptitle(
        f"TAMS CE identification  |  BT threshold: {ctt_threshold_c}°C",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    plt.show()


def load_goes_to_nc(
    lats_syn=(0, 25), lons_syn=(-70, -10),
    every_nth=6,
):
    """
    Load GOES files, subset to the synoptic domain, and save the brightness
    temperature DataArray to data/tams.nc.
    """
    import pandas as pd
    from synoptic_overview import calc_latlon, get_xy_from_latlon

    path_goes = "/Users/ethanmurray/files-research-postdoc/data/aew/goes/level2/"
    path_tb   = str(path_project) + "/data/tams.nc"

    all_files = sorted(f for f in os.listdir(path_goes) if f.endswith('.nc'))
    files = all_files[::every_nth]
    print(f"Using {len(files)} files (every {every_nth}th of {len(all_files)})")

    slices = []
    print("Loading GOES files...")
    for i, fname in enumerate(files):
        ds = xr.open_dataset(path_goes + fname)
        ds = calc_latlon(ds)
        ((x1, x2), (y1, y2)) = get_xy_from_latlon(ds, lats_syn, lons_syn)
        sub = ds.sel(x=slice(x1, x2), y=slice(y2, y1))
        t = pd.to_datetime(fname.split('_s')[1][:13], format='%Y%j%H%M%S')
        slices.append(xr.DataArray(
            sub['CMI_C13'].values.astype(float),
            dims=['y', 'x'],
            coords={'lat': (['y', 'x'], sub['lat'].values),
                    'lon': (['y', 'x'], sub['lon'].values),
                    'time': t},
        ))
        ds.close()
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(files)}")

    tb = xr.concat(slices, dim='time')
    tb.attrs.update(units='K', long_name='Brightness temperature')

    print(f"Saving BT data → {path_tb}")
    tb.to_netcdf(path_tb)
    print("Done.")
    return tb


# sensitivity tests completed:
# ctt_threshold_c= -15 -> way too many cells combined
# size_filter = True removes our case and many smaller ones
# ctt_threshold_=-25 -> ok results, but lumps our cloud in with convection to se or sw
# result: keep both thresholds at default values
def run_tams_tracking(
    ctt_threshold_c=-35.0, ctt_core_threshold_c=-54.0,
    u_projection=-5, min_area_km2=1000,
):
    """
    Load data/tams.nc, run TAMS identify + track, and save
    data/tams_tracked.parquet.

    Run load_goes_to_nc() first to create tams.nc.
    """
    import tams
    import shapely
    import geopandas as gpd
    import pandas as pd

    K_OFFSET  = 273.15
    path_tb   = str(path_project) + "/data/tams.nc"
    path_cs   = str(path_project) + "/data/tams_tracked.parquet"

    print(f"Loading BT data from {path_tb}...")
    tb = xr.open_dataarray(path_tb)

    # ── identify: one time step at a time for geometry repair ────────────────
    print("Running TAMS identify...")
    contour_sets = []
    n_skip = 0
    for i, t in enumerate(tb.time):
        tb_t = tb.sel(time=t)
        try:
            cs_t, _ = tams.identify(
                tb_t,
                ctt_threshold=ctt_threshold_c + K_OFFSET,
                ctt_core_threshold=ctt_core_threshold_c + K_OFFSET,
                size_filter=False,
            )
            frame = cs_t[0]
            if not frame.empty:
                frame['geometry'] = frame['geometry'].apply(shapely.make_valid)
                # drop tiny CEs whose complex geometry causes topology errors in track
                frame['area_km2'] = frame.to_crs('EPSG:32663').area / 1e6
                frame = frame[frame['area_km2'] >= min_area_km2].reset_index(drop=True)
        except Exception as e:
            print(f"  Warning: skipped {pd.Timestamp(t.values):%Y-%m-%d %H:%MZ} — {e}")
            frame = gpd.GeoDataFrame(columns=['geometry', 'area_km2'])
            n_skip += 1
        contour_sets.append(frame)
        if (i + 1) % 10 == 0:
            print(f"  Identified {i+1}/{len(tb.time)} time steps ({n_skip} skipped)")

    print(f"Identify complete. {n_skip} time steps skipped due to topology errors.")

    # re-apply make_valid before track does its spatial joins
    for frame in contour_sets:
        if not frame.empty:
            frame['geometry'] = frame['geometry'].apply(shapely.make_valid)

    # ── track: bisect to find bad frames, replace with empty, then retry ─────
    print("Running TAMS track...")

    def _try_track(cs_list, times):
        return tams.track(cs_list, times, u_projection=u_projection).reset_index(drop=True)

    def _bisect_bad(cs_list, times, lo, hi):
        """Return indices of all frames that cause topology errors."""
        if hi - lo < 1:
            return []
        try:
            _try_track(cs_list[lo:hi+1], times[lo:hi+1])
            return []   # no bad frames in this range
        except Exception:
            if hi - lo == 0:
                t_str = pd.Timestamp(times[lo].values).strftime('%Y-%m-%d %H:%MZ')
                print(f"  Bad frame found: index {lo}  ({t_str})")
                return [lo]
            mid = (lo + hi) // 2
            return _bisect_bad(cs_list, times, lo, mid) + \
                   _bisect_bad(cs_list, times, mid + 1, hi)

    bad = _bisect_bad(contour_sets, tb.time, 0, len(contour_sets) - 1)

    if bad:
        print(f"\n{len(bad)} bad frame(s) identified — replacing with empty GeoDataFrame:")
        for idx in bad:
            t_str = pd.Timestamp(tb.time[idx].values).strftime('%Y-%m-%d %H:%MZ')
            print(f"  index {idx:3d}  {t_str}  ({len(contour_sets[idx])} CE(s) dropped)")
            contour_sets[idx] = gpd.GeoDataFrame(columns=['geometry', 'area_km2'])

    cs = _try_track(contour_sets, tb.time)

    cs = cs.drop(columns=[c for c in ['cs219', 'inds219'] if c in cs.columns])

    print(f"Saving tracked data → {path_cs}")
    cs.to_parquet(path_cs)
    print(f"Done. {cs['mcs_id'].nunique()} unique MCS IDs across {len(tb.time)} time steps.")
    return cs


def find_tams_synoptic(
    lats_syn=(0, 25), lons_syn=(-70, -10),
    lats_zoom=(8, 13),   lons_zoom=(-56.5, -51.5),
    lats_mid=(6, 16),    lons_mid=(-60., -50.),
    ctt_threshold_c=-35.0,
    plot_type='all',   # 'synoptic' | 'mid' | 'zoom' | 'all'
):
    """
    Load saved TAMS tracking output and save figures per time step.

    plot_type controls which figures are produced:
      'synoptic' — full domain only
      'mid'      — intermediate domain with mcs_id labels
      'zoom'     — tight convective domain, no labels
      'all'      — all three

    Saves to figures/tams-tests/{synoptic,mid,zoom}/
    """
    import geopandas as gpd
    import cartopy.crs as ccrs
    from synoptic_overview import build_custom_cmap

    K_OFFSET  = 273.15
    path_tb   = str(path_project) + "/data/tams.nc"
    path_cs   = str(path_project) + "/data/tams_tracked.parquet"
    path_figs = str(path_project) + "/figures/tams-tests/"

    do_syn  = plot_type in ('synoptic', 'all')
    do_mid  = plot_type in ('mid', 'all')
    do_zoom = plot_type in ('zoom', 'all')

    if do_syn:  os.makedirs(path_figs + "synoptic/", exist_ok=True)
    if do_mid:  os.makedirs(path_figs + "mid/",      exist_ok=True)
    if do_zoom: os.makedirs(path_figs + "zoom/",     exist_ok=True)

    print("Loading saved TAMS data...")
    tb     = xr.open_dataarray(path_tb)
    cs_all = gpd.read_parquet(path_cs)

    combined_cmap = build_custom_cmap()
    tran  = ccrs.PlateCarree()
    times = tb.time.to_index()

    def _plot_ces(ax, cs_t, lons, lats, label_ids=False):
        """Draw CE outlines and optionally label mcs_id for centroids inside the domain."""
        if cs_t.empty:
            return
        # clip to CEs that intersect this domain before plotting
        from shapely.geometry import box
        domain_box = box(lons[0], lats[0], lons[1], lats[1])
        cs_visible = cs_t[cs_t.intersects(domain_box)].reset_index(drop=True)
        if cs_visible.empty:
            return
        cs_visible.plot(ax=ax, facecolor='none', edgecolor='white',
                        linewidth=1.4, transform=tran, zorder=2)
        if label_ids:
            for _, row in cs_visible.iterrows():
                cx = row.geometry.centroid.x
                cy = row.geometry.centroid.y
                if lons[0] <= cx <= lons[1] and lats[0] <= cy <= lats[1]:
                    ax.text(cx, cy, str(int(row['mcs_id'])),
                            transform=tran, fontsize=8, color='white',
                            ha='center', va='center', fontweight='bold', zorder=3)

    print(f"Saving {plot_type} figures for {len(times)} time steps...")
    for i, t in enumerate(times):
        cs_t  = cs_all[cs_all['time'] == t].reset_index(drop=True)
        bt_c  = tb.sel(time=t).values - K_OFFSET
        lat2d = tb.sel(time=t).coords['lat'].values
        lon2d = tb.sel(time=t).coords['lon'].values
        x     = np.where(np.isfinite(lon2d), lon2d, 0)
        y     = np.where(np.isfinite(lat2d), lat2d, 0)
        label = f"{t:%Y-%m-%d %H:%MZ}  |  {len(cs_t)} CE(s)  |  {ctt_threshold_c}°C"

        def _base_ax(figsize, lons, lats, coast_res):
            fig = plt.figure(figsize=figsize)
            ax  = plt.axes(projection=tran)
            ax.pcolormesh(x, y, bt_c, transform=tran,
                          vmin=190.-K_OFFSET, vmax=305.-K_OFFSET,
                          cmap=combined_cmap, zorder=1)
            ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=tran)
            ax.coastlines(resolution=coast_res, linewidth=0.7)
            ax.set_title(label, fontsize=9)
            return fig, ax

        if do_syn:
            fig, ax = _base_ax((12, 5), lons_syn, lats_syn, '50m')
            _plot_ces(ax, cs_t, lons_syn, lats_syn, label_ids=False)
            # draw mid and zoom domain boxes
            for bxl, byl in [(lons_mid, lats_mid), (lons_zoom, lats_zoom)]:
                ax.plot([bxl[0], bxl[0], bxl[1], bxl[1], bxl[0]],
                        [byl[0], byl[1], byl[1], byl[0], byl[0]],
                        'w-', lw=1.0, transform=tran, zorder=3)
            fig.tight_layout()
            fig.savefig(path_figs + f"synoptic/syn_{t:%Y%m%d_%H%M}.png",
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

        if do_mid:
            fig, ax = _base_ax((8, 6), lons_mid, lats_mid, '10m')
            _plot_ces(ax, cs_t, lons_mid, lats_mid, label_ids=True)
            fig.tight_layout()
            fig.savefig(path_figs + f"mid/mid_{t:%Y%m%d_%H%M}.png",
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

        if do_zoom:
            fig, ax = _base_ax((6, 5), lons_zoom, lats_zoom, '10m')
            _plot_ces(ax, cs_t, lons_zoom, lats_zoom, label_ids=True)
            fig.tight_layout()
            fig.savefig(path_figs + f"zoom/zoom_{t:%Y%m%d_%H%M}.png",
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

        if (i + 1) % 10 == 0 or i == len(times) - 1:
            print(f"  Saved {i+1}/{len(times)}: {t:%Y-%m-%d %H:%MZ}  ({len(cs_t)} CEs)")

    active = [n for n, f in [('synoptic', do_syn), ('mid', do_mid), ('zoom', do_zoom)] if f]
    print(f"\nDone. Figures saved: {', '.join(active)} → {path_figs}")




def plot_single_mcs_track(
    mcs_id=782,
    companion_ids=None,
    include_companions=True,
):
    """
    Plot CE polygons and centroid track for one primary MCS, with optional
    companion MCS IDs shown alongside it.

    Also computes and prints per-timestep statistics:
      speed (km/h), movement direction (° from north), orientation (°),
      min BT (°C), mean BT (°C), area (km²).

    Parameters
    ----------
    mcs_id : int
        Primary MCS to highlight (default 782).
    companion_ids : list of int, optional
        Additional MCS IDs that belong to the same convective line.
        Default: [490, 757, 683].
    include_companions : bool
        If True, overlay companion polygons and tracks. Default True.
    """
    import geopandas as gpd
    import pandas as pd
    import regionmask

    if companion_ids is None:
        companion_ids = [490, 757, 683]

    K_OFFSET = 273.15
    path_tb = str(path_project) + "/data/tams.nc"
    path_cs = str(path_project) + "/data/tams_tracked.parquet"

    print("Loading saved TAMS data...")
    tb = xr.open_dataarray(path_tb)
    cs = gpd.read_parquet(path_cs)

    ids_to_plot = [mcs_id] + (companion_ids if include_companions else [])
    colors = {mcs_id: 'firebrick'}
    companion_colors = ['steelblue', 'seagreen', 'darkorange']
    for cid, col in zip(companion_ids, companion_colors):
        colors[cid] = col

    def _compute_stats(mid):
        group = cs[cs['mcs_id'] == mid].sort_values('time').reset_index(drop=True)
        rows = []
        for _, row in group.iterrows():
            t    = row['time']
            geom = row['geometry']
            gs   = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs('EPSG:32663')
            geom_m = gs.values[0]
            area_km2 = gs.area.values[0] / 1e6
            cen_geo = gs.centroid.to_crs('EPSG:4326')
            clon = cen_geo.x.values[0]
            clat = cen_geo.y.values[0]
            rect   = shapely.minimum_rotated_rectangle(geom_m)
            coords = np.array(rect.exterior.coords[:-1])
            sides  = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords)-1)]
            li     = int(np.argmax(sides))
            dx     = coords[li+1][0] - coords[li][0]
            dy     = coords[li+1][1] - coords[li][1]
            orient = float(np.degrees(np.arctan2(dy, dx)) % 180)
            try:
                tb_t   = tb.sel(time=t)
                gdf_ce = gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:4326')
                mask   = regionmask.from_geopandas(gdf_ce).mask(tb_t)
                bt_vals = (tb_t.where(mask == 0) - K_OFFSET).values
                bt_vals = bt_vals[np.isfinite(bt_vals)]
                min_bt  = float(np.min(bt_vals))  if len(bt_vals) > 0 else np.nan
                mean_bt = float(np.mean(bt_vals)) if len(bt_vals) > 0 else np.nan
            except Exception:
                min_bt = mean_bt = np.nan
            rows.append({'mcs_id': mid, 'time': t,
                         'centroid_lat': clat, 'centroid_lon': clon,
                         'area_km2': area_km2, 'min_bt_c': min_bt,
                         'mean_bt_c': mean_bt, 'orientation_deg': orient})

        df = pd.DataFrame(rows)

        # speed and movement direction
        df['speed_kmh']     = np.nan
        df['direction_deg'] = np.nan
        if len(df) >= 2:
            pts = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(df['centroid_lon'], df['centroid_lat']),
                crs='EPSG:4326',
            ).to_crs('EPSG:32663')
            xs = np.array([p.x for p in pts.geometry]) / 1000
            ys = np.array([p.y for p in pts.geometry]) / 1000
            ts = pd.DatetimeIndex(df['time'].values)
            for j in range(1, len(df)):
                dt_h = (ts[j] - ts[j-1]).total_seconds() / 3600
                ddx  = xs[j] - xs[j-1]
                ddy  = ys[j] - ys[j-1]
                dist = np.sqrt(ddx**2 + ddy**2)
                df.loc[j, 'speed_kmh']     = dist / dt_h if dt_h > 0 else np.nan
                # direction: degrees from north, clockwise
                df.loc[j, 'direction_deg'] = float(np.degrees(np.arctan2(ddx, ddy)) % 360)

        return df

    # ── compute stats ──────────────────────────────────────────────────────────
    all_stats = {}
    for mid in ids_to_plot:
        if mid not in cs['mcs_id'].values:
            print(f"  Warning: mcs_id {mid} not found in parquet, skipping.")
            continue
        print(f"  Computing stats for mcs_id {mid}...")
        all_stats[mid] = _compute_stats(mid)

    # ── print summary table ────────────────────────────────────────────────────
    for mid, df in all_stats.items():
        print(f"\n── mcs_id {mid} ──")
        print(df[['time', 'speed_kmh', 'direction_deg', 'orientation_deg',
                  'min_bt_c', 'mean_bt_c', 'area_km2']].to_string(index=False))

    # ── map: polygons + centroid tracks ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    for mid, df in all_stats.items():
        color = colors.get(mid, '0.5')
        label = f'mcs_id {mid}' + (' (primary)' if mid == mcs_id else '')
        geoms = cs[cs['mcs_id'] == mid].sort_values('time')
        geoms.plot(ax=ax, facecolor='none', edgecolor=color, linewidth=1.2, alpha=0.6, label=label)
        # centroid track
        centroids = geoms.to_crs('EPSG:32663').centroid.to_crs('EPSG:4326')
        ax.plot(centroids.x, centroids.y, '-o', color=color, linewidth=1.8, markersize=5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    title = f'mcs_id {mcs_id}'
    if include_companions and companion_ids:
        title += f'  +  companions {companion_ids}'
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()

    # ── stats time series ─────────────────────────────────────────────────────
    stat_cols = ['area_km2', 'min_bt_c', 'mean_bt_c', 'speed_kmh', 'direction_deg', 'orientation_deg']
    ylabels   = ['Area (km²)', 'Min BT (°C)', 'Mean BT (°C)', 'Speed (km/h)', 'Direction (°N)', 'Orientation (°)']

    fig2, axes = plt.subplots(len(stat_cols), 1, figsize=(11, 3 * len(stat_cols)), sharex=True)
    for mid, df in all_stats.items():
        color = colors.get(mid, '0.5')
        lw    = 2.0 if mid == mcs_id else 1.2
        alpha = 1.0 if mid == mcs_id else 0.7
        sub   = df.sort_values('time')
        for ax, col in zip(axes, stat_cols):
            ax.plot(sub['time'], sub[col], color=color, lw=lw, alpha=alpha,
                    marker='o', ms=4, label=f'mcs_id {mid}')

    for ax, ylabel in zip(axes, ylabels):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title(f'MCS statistics — primary: {mcs_id}' +
                      (f', companions: {companion_ids}' if include_companions else ''), fontsize=11)
    axes[-1].set_xlabel('Time (UTC)', fontsize=9)
    fig2.autofmt_xdate(rotation=30)
    fig2.tight_layout()

    plt.show()
    return {mid: df for mid, df in all_stats.items()}


def plot_tams_stats_nice(
    mcs_id=782,
    companion_ids=None,
    include_companions=True,
    start_time=None,
    end_time=None,
):
    """
    Publication-style MCS-evolution figure: a large spatial-track panel (a,
    left column, mirroring plot_single_mcs_track's map) alongside stacked
    time-series panels (b-f, right column) for area, min BT, movement
    direction, speed, and orientation.

    Parameters
    ----------
    mcs_id : int
        Primary MCS to highlight (default 782).
    companion_ids : list of int, optional
        Additional MCS IDs that belong to the same convective line.
        Default: [490, 757, 683].
    include_companions : bool
        If True, overlay companion polygons/tracks/time series. Default True.
    start_time, end_time : str or pandas.Timestamp, optional
        If given, restrict every MCS's timesteps to this window (inclusive)
        before plotting.

    Saves to figures/tams_stats_nice_{mcs_id}.png
    """
    import geopandas as gpd
    import pandas as pd
    import regionmask

    if companion_ids is None:
        companion_ids = [490, 757, 683]

    K_OFFSET = 273.15
    path_tb = str(path_project) + "/data/tams.nc"
    path_cs = str(path_project) + "/data/tams_tracked.parquet"

    print("Loading saved TAMS data...")
    tb = xr.open_dataarray(path_tb)
    cs = gpd.read_parquet(path_cs)

    if start_time is not None:
        cs = cs[cs['time'] >= pd.Timestamp(start_time)]
    if end_time is not None:
        cs = cs[cs['time'] <= pd.Timestamp(end_time)]

    ids_to_plot = [mcs_id] + (companion_ids if include_companions else [])
    colors = {mcs_id: 'firebrick'}
    companion_colors = ['steelblue', 'seagreen', 'darkorange']
    for cid, col in zip(companion_ids, companion_colors):
        colors[cid] = col

    def _compute_stats(mid):
        group = cs[cs['mcs_id'] == mid].sort_values('time').reset_index(drop=True)
        rows = []
        for _, row in group.iterrows():
            t    = row['time']
            geom = row['geometry']
            gs   = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs('EPSG:32663')
            geom_m = gs.values[0]
            area_km2 = gs.area.values[0] / 1e6
            cen_geo = gs.centroid.to_crs('EPSG:4326')
            clon = cen_geo.x.values[0]
            clat = cen_geo.y.values[0]
            # orientation: long side of the minimum-area bounding rectangle
            rect   = shapely.minimum_rotated_rectangle(geom_m)
            coords = np.array(rect.exterior.coords[:-1])
            sides  = [np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords)-1)]
            li     = int(np.argmax(sides))
            odx    = coords[li+1][0] - coords[li][0]
            ody    = coords[li+1][1] - coords[li][1]
            # measured from due north (0 = north-south, 90 = east-west),
            # so a steep positive slope (near-vertical, tilted slightly
            # east of north) reads as a small angle rather than near-90
            orient = float((90. - np.degrees(np.arctan2(ody, odx))) % 180)
            try:
                tb_t   = tb.sel(time=t)
                gdf_ce = gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:4326')
                mask   = regionmask.from_geopandas(gdf_ce).mask(tb_t)
                bt_vals = (tb_t.where(mask == 0) - K_OFFSET).values
                bt_vals = bt_vals[np.isfinite(bt_vals)]
                min_bt  = float(np.min(bt_vals)) if len(bt_vals) > 0 else np.nan
            except Exception:
                min_bt = np.nan
            rows.append({'mcs_id': mid, 'time': t,
                         'centroid_lat': clat, 'centroid_lon': clon,
                         'area_km2': area_km2, 'min_bt_c': min_bt,
                         'orientation_deg': orient})

        df = pd.DataFrame(rows)

        # movement speed and direction
        df['speed_kmh']     = np.nan
        df['direction_deg'] = np.nan
        if len(df) >= 2:
            pts = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(df['centroid_lon'], df['centroid_lat']),
                crs='EPSG:4326',
            ).to_crs('EPSG:32663')
            xs = np.array([p.x for p in pts.geometry]) / 1000
            ys = np.array([p.y for p in pts.geometry]) / 1000
            ts = pd.DatetimeIndex(df['time'].values)
            for j in range(1, len(df)):
                dt_h = (ts[j] - ts[j-1]).total_seconds() / 3600
                ddx  = xs[j] - xs[j-1]
                ddy  = ys[j] - ys[j-1]
                dist = np.sqrt(ddx**2 + ddy**2)
                df.loc[j, 'speed_kmh']     = dist / dt_h if dt_h > 0 else np.nan
                df.loc[j, 'direction_deg'] = float(np.degrees(np.arctan2(ddx, ddy)) % 360)

        return df

    # ── compute stats ──────────────────────────────────────────────────────
    all_stats = {}
    for mid in ids_to_plot:
        if mid not in cs['mcs_id'].values:
            print(f"  Warning: mcs_id {mid} not found (or empty after time cutoff), skipping.")
            continue
        print(f"  Computing stats for mcs_id {mid}...")
        all_stats[mid] = _compute_stats(mid)

    # ── font sizes ────────────────────────────────────────────────────────
    label_fs = 18.
    tick_fs  = 15.

    def _panel_label(ax, letter, y=0.96):
        ax.text(0.03, y, f'({letter})', transform=ax.transAxes,
                ha='left', va='top', fontsize=label_fs, zorder=20,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                          linewidth=1.2, pad=4))

    # ── figure: panel (a) spatial map (left), panels (b-f) time series (right)
    fig = plt.figure(figsize=(16, 13))
    gs  = GridSpec(5, 2, figure=fig, width_ratios=[1.3, 1],
                   wspace=0.14, hspace=0.15)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_b   = fig.add_subplot(gs[0, 1])
    ax_c   = fig.add_subplot(gs[1, 1], sharex=ax_b)
    ax_d   = fig.add_subplot(gs[2, 1], sharex=ax_b)
    ax_e   = fig.add_subplot(gs[3, 1], sharex=ax_b)
    ax_f   = fig.add_subplot(gs[4, 1], sharex=ax_b)

    # -- panel (a): polygons + centroid tracks --------------------------------
    for mid, df in all_stats.items():
        color = colors.get(mid, '0.5')
        geoms = cs[cs['mcs_id'] == mid].sort_values('time')
        geoms.plot(ax=ax_map, facecolor='none', edgecolor=color, linewidth=1.4,
                  alpha=0.6)
        centroids = geoms.to_crs('EPSG:32663').centroid.to_crs('EPSG:4326')
        ax_map.plot(centroids.x, centroids.y, '-o', color=color,
                   linewidth=2.0, markersize=6)

    ax_map.set_xlabel('Longitude', fontsize=label_fs)
    ax_map.set_ylabel('Latitude', fontsize=label_fs)
    ax_map.tick_params(labelsize=tick_fs)
    ax_map.set_aspect('equal')
    _panel_label(ax_map, 'a', y=0.98)

    # -- panels (b-f): area, min BT, direction, speed, orientation -----------
    stat_cols  = ['area_km2', 'min_bt_c', 'direction_deg', 'speed_kmh', 'orientation_deg']
    ylabels    = ['Area (km²)', 'Min BT (°C)', 'Direction (°N)', 'Speed (km/h)', 'Orientation (°)']
    axes       = [ax_b, ax_c, ax_d, ax_e, ax_f]
    letters    = ['b', 'c', 'd', 'e', 'f']

    for mid, df in all_stats.items():
        color = colors.get(mid, '0.5')
        lw    = 2.2 if mid == mcs_id else 1.4
        alpha = 1.0 if mid == mcs_id else 0.7
        sub   = df.sort_values('time')
        label = f'mcs_id {mid}' + (' (primary)' if mid == mcs_id else '')
        for ax, col in zip(axes, stat_cols):
            ax.plot(sub['time'], sub[col], color=color, lw=lw, alpha=alpha,
                    marker='o', ms=5, label=label)

    for ax, ylabel, letter in zip(axes, ylabels, letters):
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, alpha=0.3)
        _panel_label(ax, letter, y=0.88)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    ax_b.set_ylim(bottom=0)   # panel (b) area axis includes 0

    ax_f.set_xlabel('Time (UTC)', fontsize=label_fs)
    fig.autofmt_xdate(rotation=30)

    # thin panel (f)'s time ticks to every other one (drop the first, e.g. 15 UTC)
    f_ticks = ax_f.get_xticks()
    ax_f.set_xticks(f_ticks[1::2])

    # ── save ─────────────────────────────────────────────────────────────
    path_figures = str(path_project) + '/figures/'
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + f'tams_stats_nice_{mcs_id}.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')

    return {mid: df for mid, df in all_stats.items()}


def plot_dbz_area(dbz_threshold=10):
    """
    For each TDR pass, compute the area (km²) of pixels with reflectivity
    >= dbz_threshold at every height level, then plot area vs height.

    Parameters
    ----------
    dbz_threshold : float
        Minimum reflectivity (dBZ) to count a pixel. Default 10.

    Saves to figures/tdr_dbz_area_Xdbz.png
    """
    passes = {
        'Pass 1 (1804)': ('240715I1_1804_xy.nc',),
        'Pass 2 (1919)': ('240715I1_1919_xy.nc',),
        'Pass 3 (2030)': ('240715I1_2030_xy.nc',),
    }
    colors = {
        'Pass 1 (1804)': 'steelblue',
        'Pass 2 (1919)': 'darkorange',
        'Pass 3 (2030)': 'firebrick',
    }

    path_figures = str(path_project) + '/figures/'

    fig, ax = plt.subplots(figsize=(5, 7))

    print(f"\n{'Pass':<20}  {'Height (km)':>11}  {'N pixels':>9}  {'Area (km²)':>11}")
    print("-" * 58)

    for pass_name, (fname,) in passes.items():
        ds     = xr.open_dataset(path_tdr + fname)
        refl   = ds['REFLECTIVITY'].values   # (x, y, level, time)
        levels = ds['level'].values
        ds.close()

        n_levels = len(levels)
        areas = np.full(n_levels, np.nan)

        for li in range(n_levels):
            slab      = refl[:, :, li, 0]              # (x, y)
            valid     = (slab > MISSING) & (slab >= dbz_threshold)
            n_pix     = int(np.sum(valid))
            areas[li] = n_pix * PIXEL_AREA_KM2
            print(f"{pass_name:<20}  {levels[li]:>11.1f}  {n_pix:>9d}  {areas[li]:>11.1f}")

        print()

        mask = areas > 0
        ax.plot(areas[mask], levels[mask],
                color=colors[pass_name], lw=2, label=pass_name)

    ax.set_xlabel(f'Area with refl. ≥ {dbz_threshold} dBZ (km²)')
    ax.set_ylabel('Height (km)')
    ax.set_title(f'TDR echo area ≥ {dbz_threshold} dBZ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    os.makedirs(path_figures, exist_ok=True)
    outfile = path_figures + f'tdr_dbz_area_{dbz_threshold:.0f}dbz.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outfile}')

