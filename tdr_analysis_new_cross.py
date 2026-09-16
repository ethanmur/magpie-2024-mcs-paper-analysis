#%%
"""
Across-axis TDR curtain.

Unlike plot_tdr_curtains() in tdr_analysis.py (which follows the P-3 flight
path, i.e. roughly along the convective long axis), this module builds a
straight spatial cross-section between two arbitrary lat/lon points so the
curtain can be cut ACROSS the convective long axis (along the short axis).

Two pieces:
  build_cross_curtain(...)      -> samples a single TDR volume along a line
                                    and saves tdr_curtain_2160_across.nc
  plot_tdr_curtains_across(...) -> mirrors plot_tdr_curtains() but with panel
                                    (a) (nadir W-band) dropped, since W-band
                                    only exists along the flight track.
"""
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs

path_project = Path(__file__).parent.parent
path_figures = str(path_project) + "/figures/"
path_data    = str(path_project) + "/data/"

sys.path.append("/Users/ethanmurray/files-research-postdoc/code/aew-analysis/plotting/")
import magpie_data_plotting

# raw TDR level-2 volumes (same source used to build the flight-path curtain)
TDR_L2_PATH = '/Users/ethanmurray/files-research-postdoc/data/aew/tdr/level2-all/20240715I1/'
# order used as the num_cases index in the rainfall-classification file
FILE_NAMES = ['240715I1_1804_xy.nc', '240715I1_1838_xy.nc', '240715I1_1919_xy.nc',
              '240715I1_1957_xy.nc', '240715I1_2030_xy.nc']

EARTH_R_M = 6371000.0


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between scalar points."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.)**2
    return 2. * EARTH_R_M * np.arcsin(np.sqrt(a))


def _rain_cmap():
    """Rainfall-classification colormap (mirrors tdr_analysis._rain_cmap)."""
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
    ticks = [-5e-3, -3e-3, -1e-3, 0., 1e-3, 3e-3, 5e-3]
    def _fmt(t):
        if t == 0:
            return '0'
        return f'{t:.3f}'.replace('0.', '.')   # -0.005 -> -.005, 0.001 -> .001
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([_fmt(t) for t in ticks])


#%%
# ---------------------------------------------------------------------------
# Helper: build a straight across-axis curtain between two lat/lon points
# ---------------------------------------------------------------------------

# old values:
# lat1=11., lon1=-54.0,      # <-- PLACEHOLDER endpoint A
#                         lat2=10.75, lon2=-53.5, 

def build_cross_curtain(lat1=10.55, lon1=-54.3,      # <-- PLACEHOLDER endpoint A
                        lat2=10.15, lon2=-53.7,      # <-- PLACEHOLDER endpoint B
                        passind=0, tdr_file='240715I1_1919_xy.nc',
                        resolution_m=2160.):
    """
    Sample a single TDR volume along the straight line A(lat1,lon1)->B(lat2,lon2)
    at ``resolution_m`` spacing and save the result as tdr_curtain_2160_across.nc.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Cross-section endpoints. PLACEHOLDER defaults — set to the real
        across-axis line before running.
    tdr_file : str
        Single TDR volume to sample (no time matching, unlike the flight-path
        curtain). Must be one of FILE_NAMES for the rainfall classification.
    resolution_m : float
        Along-section sample spacing (manually specified; default 2160 m).
    """
    # --- sample points along the line ------------------------------------
    total_m = _haversine_m(lat1, lon1, lat2, lon2)
    n_pts = max(2, int(np.floor(total_m / resolution_m)) + 1)
    frac = np.linspace(0., 1., n_pts)

    seg_lats = lat1 + frac * (lat2 - lat1)
    seg_lons = lon1 + frac * (lon2 - lon1)
    dist_km = frac * (total_m / 1000.)

    print(f'Cross-section {total_m/1000.:.1f} km, '
          f'{n_pts} points at {resolution_m:.0f} m spacing.')

    # --- load the chosen TDR volume --------------------------------------
    tdr_data = xr.open_dataset(TDR_L2_PATH + tdr_file)
    level = tdr_data.level.values
    tdr_lats = tdr_data.LATITUDE.isel(time=0).values     # (y, x)
    tdr_lons = tdr_data.LONGITUDE.isel(time=0).values    # (y, x)

    # --- build empty curtain dataset -------------------------------------
    tdr_curtain = xr.Dataset(
        coords={'dist': dist_km, 'level': level},
    )
    tdr_curtain['seg_lat'] = (['dist'], seg_lats)
    tdr_curtain['seg_lon'] = (['dist'], seg_lons)
    tdr_curtain['tdr_lat_closest'] = (['dist'], np.full(n_pts, np.nan))
    tdr_curtain['tdr_lon_closest'] = (['dist'], np.full(n_pts, np.nan))
    tdr_curtain['rainfall_classification'] = (['dist'], np.full(n_pts, np.nan))

    # 2-D profile variables (same selection rule as the flight-path builder)
    profile_vars = []
    for var in tdr_data.variables:
        if (np.shape(tdr_data[var].values.flatten())[0] > 10000.
                and var not in ('LATITUDE', 'LONGITUDE')):
            profile_vars.append(var)
            tdr_curtain[var] = xr.DataArray(
                np.full((n_pts, len(level)), np.nan),
                coords={'dist': dist_km, 'level': level},
                dims=['dist', 'level'],
                attrs=tdr_data[var].attrs,
            )
    print('Profile variables:', profile_vars)

    # optional rainfall classification for this pass
    tcradar_classes = None
    if tdr_file in FILE_NAMES:
        class_path = '/Users/ethanmurray/files-research-postdoc/data/aew/tdr/precip-classifications/'
        class_name = 'tdr_aew_rainfall_classifications.nc'
        try:
            tcradar_classes = xr.open_dataset(class_path + class_name)
            case_i = FILE_NAMES.index(tdr_file)
        except FileNotFoundError:
            print('Rainfall-classification file not found; skipping.')

    # --- nearest-cell sampling along the section -------------------------
    for i in range(n_pts):
        dist = np.sqrt((tdr_lats - seg_lats[i])**2 + (tdr_lons - seg_lons[i])**2)
        if np.all(np.isnan(dist)):
            continue
        idx_y, idx_x = np.unravel_index(np.nanargmin(dist), dist.shape)

        tdr_curtain['tdr_lat_closest'][i] = tdr_lats[idx_y, idx_x]
        tdr_curtain['tdr_lon_closest'][i] = tdr_lons[idx_y, idx_x]

        for var in profile_vars:
            tdr_curtain[var].values[i, :] = tdr_data[var].isel(time=0).values[idx_y, idx_x, :]

        if tcradar_classes is not None:
            tdr_curtain['rainfall_classification'][i] = (
                tcradar_classes['rainfall_classification']
                .isel(num_cases=case_i).values[idx_y, idx_x]
            )

    # --- save ------------------------------------------------------------
    out_dir = path_data + '/tdr/'
    os.makedirs(out_dir, exist_ok=True)
    out_name = f'tdr_curtain_2160_across_pass_{passind}.nc'
    tdr_curtain.to_netcdf(out_dir + out_name)
    print(f'Saved {out_dir + out_name}')
    return tdr_curtain


#%%
# ---------------------------------------------------------------------------
# Figure: across-axis TDR curtain (mirrors plot_tdr_curtains, panel a dropped)
# ---------------------------------------------------------------------------

def plot_tdr_curtains_across(x_axis='distance', convective_axis_deg=None,
                             add_barbs=False, add_vort=False, cross_endpoints=(10.55, -54.3, 10.15, -53.7),
                             cross_name='', passind=0, inset_extend_km=18.):
    """
    2- to 5-panel across-axis TDR curtain figure.
      a: TDR precip-classification colorbar (atop panel b, no separate axis)
      b: TDR refl (+ optional V'/W barbs)
      c: U' along-axis wind (curtain)
      d, e: V' across-axis wind and W (curtains, only when add_barbs=False)
      last: relative vorticity (optional, when add_vort=True; always last)

    x_axis : {'distance', 'time'}
        'distance' -> along-section distance (km); 'time' -> curtain time axis
        (only valid for a time-based curtain).
    convective_axis_deg : float or None
        Rotates U/V into along/across-axis winds (degrees CW from N).
    add_barbs : bool
        Overlays V'/W wind barbs on the reflectivity panel. When False,
        V' and W are instead plotted as their own curtain panels.
    add_vort : bool
        Adds a panel with the TDR VORT field, placed after the wind panels.
    inset_extend_km : float
        Only used when x_axis='distance'. Extends every curtain panel's
        x-axis by this many km, to leave blank room for the spatial
        location-map inset drawn in the top-right corner of panel (b)
        (TDR reflectivity).
    """
    curtain_path = path_data + '/tdr/'
    tdr = xr.open_dataset(curtain_path + cross_name)

    rain_cmap = _rain_cmap()

    # --- choose x-axis coordinate ----------------------------------------
    if x_axis == 'distance':
        xvals  = tdr['dist'].values
        xlabel = 'Distance (km)'
        save_suffix='dist'
    elif x_axis == 'time':
        xvals  = tdr['time'].values
        xlabel = 'Time (Hours, UTC)'
        save_suffix=''
    else:
        raise ValueError("x_axis must be 'distance' or 'time'")

    h_tdr = tdr['level'].values

    # --- find where the P-3 flight path crosses this straight cross-section
    # (must match the endpoints used in build_cross_curtain) --------------
    cross_lat1, cross_lon1 = cross_endpoints[0], cross_endpoints[1]
    cross_lat2, cross_lon2 = cross_endpoints[2], cross_endpoints[3]

    def _dist_to_cross_line(lat, lon):
        """Perpendicular distance (deg) and projection fraction (0-1) from
        (lat, lon) to the cross-section segment."""
        dx, dy = cross_lon2 - cross_lon1, cross_lat2 - cross_lat1
        seg_len2 = dx * dx + dy * dy
        t = ((lon - cross_lon1) * dx + (lat - cross_lat1) * dy) / seg_len2
        t_clip = np.clip(t, 0., 1.)
        proj_lon = cross_lon1 + t_clip * dx
        proj_lat = cross_lat1 + t_clip * dy
        return np.hypot(lon - proj_lon, lat - proj_lat), t_clip

    cross_x = None
    if x_axis == 'distance':
        fl_path = '/Users/ethanmurray/files-research-postdoc/data/aew/flight-level/'
        fl = xr.open_dataset(fl_path + '20240715I1_A_small.nc', decode_times=False)
        fl_lat  = fl['LATref'].values
        fl_lon  = fl['LONref'].values
        fl_time = magpie_data_plotting.build_time_array(fl, axis_type='decimal')

        # restrict to this pass's time window (mirrors tdr_analysis.py's
        # time_ranges: 0 -> ~1919 UTC, 1 -> ~2030 UTC) — previously
        # hardcoded to the pass-0 window regardless of passind, so passind=1
        # searched the wrong stretch of flight track
        pass_windows = [(18.9, 19.45), (20.1, 20.75)]
        t0, t1 = pass_windows[passind]
        seg = (fl_time >= t0) & (fl_time <= t1)
        dists_t = [_dist_to_cross_line(la, lo)
                   for la, lo in zip(fl_lat[seg], fl_lon[seg])]
        if dists_t:
            dists  = np.array([d for d, t in dists_t])
            fracs  = np.array([t for d, t in dists_t])
            if np.any(np.isfinite(dists)):
                i_min   = np.nanargmin(dists)
                total_km = float(tdr['dist'].values[-1])
                cross_x = fracs[i_min] * total_km
                seg_lat = fl_lat[seg]
                seg_lon = fl_lon[seg]
                print(f'  [across-axis curtain, pass {passind}] intersection at '
                      f'({seg_lat[i_min]:.4f}, {seg_lon[i_min]:.4f}), '
                      f'dist to line = {dists[i_min]:.5f} deg')

    # --- optional axis rotation ------------------------------------------
    U_raw = tdr['U'].values * 1.944   # m/s -> knots  (x, level)
    V_raw = tdr['V'].values * 1.944
    W_raw = tdr['W'].values * 1.944

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

    # --- figure layout ------------------------------------------------------
    # rows: reflectivity, U' (along-axis) curtain, [V' curtain, W curtain if
    # not add_barbs (winds shown as separate panels instead of barbs)],
    # [vorticity, always last, if add_vort]
    n_rows        = 2 + (2 if not add_barbs else 0) + (1 if add_vort else 0)
    height_ratios = [1.6] * n_rows
    fig_height    = 3. * n_rows
    fig = plt.figure(figsize=(12, fig_height))
    gs  = GridSpec(n_rows, 2, width_ratios=[1, 0.04],
                   height_ratios=height_ratios, hspace=0.30, wspace=0.06)
    axs  = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
    caxs = [fig.add_subplot(gs[i, 1]) for i in range(n_rows)]
    ax_a, cax_a = axs[0], caxs[0]   # TDR reflectivity (+ optional V'/W barbs)
    ax_b, cax_b = axs[1], caxs[1]   # U' / along-axis wind (curtain)
    row = 2
    if not add_barbs:
        ax_v, cax_v = axs[row], caxs[row]; row += 1   # V' / across-axis wind
        ax_w, cax_w = axs[row], caxs[row]; row += 1   # W (vertical velocity)
    if add_vort:
        ax_vort, cax_vort = axs[row], caxs[row]; row += 1

    xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
    if x_axis == 'distance':
        # extra blank room on the right for the panel (b) location-map inset
        xmax += inset_extend_km

    # ---- panel (b): TDR reflectivity (+ optional V'/W barbs) ------------
    p_a = ax_a.pcolormesh(xvals, h_tdr, tdr['REFLECTIVITY'].values.transpose(),
                          cmap='RdYlBu_r', vmin=-10, vmax=40, shading='auto')
    fig.colorbar(p_a, cax=cax_a, label='TDR Refl. (dBZ)', extend='both')

    # rain classification bar above panel b, mirroring figures 6/7
    # nudged further up so it clears the top of panel b's axis
    if 'rainfall_classification' in tdr.variables:
        rain_class = tdr['rainfall_classification'].values
        ax_a.pcolormesh(xvals, [20.5, 21.5],
                        np.vstack((rain_class, rain_class)),
                        cmap=rain_cmap, vmin=-0.5, vmax=4.5,
                        shading='nearest', clip_on=False)

    if add_barbs:
        t_skip = max(1, len(xvals) // 30)
        h_skip = max(1, len(h_tdr) // 16)
        x_b = xvals[::t_skip]
        h_b = h_tdr[::h_skip]
        # across-axis wind V' (in-plane for this perpendicular section) → barbs
        u_b = Vprime[::t_skip, ::h_skip]
        w_b = W_raw[::t_skip, ::h_skip]
        x_2d, h_2d = np.meshgrid(x_b, h_b, indexing='ij')
        valid = np.isfinite(u_b) & np.isfinite(w_b)
        ax_a.barbs(x_2d[valid], h_2d[valid], u_b[valid], w_b[valid],
                   length=5, linewidth=0.7, color='k', pivot='middle')

    ax_a.set_xlim([xmin, xmax])
    ax_a.set_ylim([0, 18])
    ax_a.set_ylabel('Height (km)')
    ax_a.axhline(y=3., c='k', ls='--', lw=1.2)

    # ---- location-map inset in panel (c)'s top-right corner: spatial
    # reflectivity (colored contourf, 5 dBZ intervals) + the straight
    # cross-section line this curtain follows (this is the across-axis
    # curtain, so only the cross-section line is relevant here — no P-3
    # flight-path line), so the reader can see where the data were taken --
    if x_axis == 'distance' and np.any(seg):
        pv_files = ['240715I1_1804_xy.nc', '240715I1_1919_xy.nc', '240715I1_2030_xy.nc']
        pv_file  = pv_files[passind + 1]   # passind 0 -> 1919, 1 -> 2030
        pv_ds    = xr.open_dataset(path_data + 'tdr/plan-view/' + pv_file)

        pv_lev_2km = int(np.argmin(np.abs(pv_ds['level'].values - 2.)))
        pv_lat  = pv_ds['LATITUDE'].isel(time=0).values
        pv_lon  = pv_ds['LONGITUDE'].isel(time=0).values
        pv_dbz  = pv_ds['REFLECTIVITY'].isel(time=0, level=pv_lev_2km).values
        pv_dbz  = np.where(pv_dbz < -998., np.nan, pv_dbz)
        pv_ds.close()

        pc_inset = ccrs.PlateCarree()
        # anchored to ax_a's own axes-fraction coordinates (not a floating
        # figure-space box), so it stays locked to the panel's upper-right
        # corner through any later layout/bbox adjustments
        ax_inset = ax_a.inset_axes([0.62, 0.38, 0.38, 0.62],
                                   transform=ax_a.transAxes, projection=pc_inset)

        pad = 0.25
        lon_lo = min(cross_lon1, cross_lon2) - pad
        lon_hi = max(cross_lon1, cross_lon2) + pad
        lat_lo = min(cross_lat1, cross_lat2) - pad
        lat_hi = max(cross_lat1, cross_lat2) + pad
        ax_inset.set_extent([lon_lo, lon_hi, lat_lo, lat_hi], crs=pc_inset)
        # keep the correct geographic aspect ratio (don't stretch/distort
        # the radar data) — instead anchor the drawn map to the top-right
        # corner of its box, so any letterboxing needed to preserve aspect
        # eats into the left/bottom margin instead of leaving a gap on the
        # right edge (which must stay flush)
        ax_inset.set_anchor('NE')

        # colored reflectivity via filled contours, 5 dBZ intervals
        ax_inset.contourf(pv_lon, pv_lat, pv_dbz, levels=np.arange(0, 46, 5),
                          cmap='RdYlBu_r', extend='both', transform=pc_inset)

        # the straight cross-section line this curtain actually samples,
        # with its start/end marked x/* in the same color as the line
        ax_inset.plot([cross_lon1, cross_lon2], [cross_lat1, cross_lat2],
                     color='firebrick', lw=1.6, transform=pc_inset, zorder=7)
        ax_inset.scatter(cross_lon1, cross_lat1, marker='x', color='firebrick',
                        s=35, linewidths=1.8, transform=pc_inset, zorder=8)
        ax_inset.scatter(cross_lon2, cross_lat2, marker='*', color='firebrick',
                        s=55, transform=pc_inset, zorder=8)

        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('k')
            spine.set_linewidth(0.8)

    # ---- panel (c): U' / along-axis wind (curtain) -----------------------
    p_b = ax_b.pcolormesh(xvals, h_tdr, Uprime.transpose(),
                          cmap='seismic', vmin=-30, vmax=30, shading='auto')
    fig.colorbar(p_b, cax=cax_b, label=Uprime_label, extend='both')

    ax_b.set_xlim([xmin, xmax])
    ax_b.set_ylim([0, 18])
    ax_b.set_ylabel('Height (km)')
    if add_barbs and not add_vort:
        ax_b.set_xlabel(xlabel)

    # ---- panels (d)/(e): V' and W curtains, shown when barbs are off ----
    if not add_barbs:
        p_v = ax_v.pcolormesh(xvals, h_tdr, Vprime.transpose(),
                              cmap='seismic', vmin=-30, vmax=30, shading='auto')
        fig.colorbar(p_v, cax=cax_v, label=Vprime_label, extend='both')
        ax_v.set_xlim([xmin, xmax])
        ax_v.set_ylim([0, 18])
        ax_v.set_ylabel('Height (km)')

        p_w = ax_w.pcolormesh(xvals, h_tdr, W_raw.transpose(),
                              cmap='seismic', vmin=-10, vmax=10, shading='auto')
        fig.colorbar(p_w, cax=cax_w, label='TDR W (kt)', extend='both')
        ax_w.set_xlim([xmin, xmax])
        ax_w.set_ylim([0, 18])
        ax_w.set_ylabel('Height (km)')
        if not add_vort:
            ax_w.set_xlabel(xlabel)

    # ---- last panel: relative vorticity (optional) -------------------------
    if add_vort:
        vort_data = tdr['VORT'].values.transpose() / 1000.   # -> s^-1
        vmin, vmax = -4e-3, 4e-3
        n_bands = 15
        levels = np.linspace(vmin, vmax, n_bands + 1)
        p_vort = ax_vort.contourf(xvals, h_tdr, vort_data,
                                  cmap='RdBu_r', levels=levels, extend='both')
        cb_vort = fig.colorbar(p_vort, cax=cax_vort, label='Rel. Vorticity (s⁻¹)')
        _set_vort_ticks(cb_vort)
        ax_vort.set_xlim([xmin, xmax])
        ax_vort.set_ylim([0, 18])
        ax_vort.set_ylabel('Height (km)')
        ax_vort.set_xlabel(xlabel)

    # ---- panel (a) label: base aligned with the base of the precip
    # classification bar (y=20.5 in data coords), left edge aligned with
    # panel (b)'s left edge (x=0.015 in axes-fraction) --------------------
    a_label_y = 20.5 / 18.
    ax_a.text(0.015, a_label_y, '(a)',
              transform=ax_a.transAxes, ha='left', va='bottom', fontsize=10,
              zorder=20, clip_on=False,
              bbox=dict(facecolor='white', alpha=1.,
                        edgecolor='k', linewidth=1.2, pad=4))

    # ---- panel labels (b, c, ...): one per physical panel after (a) ------
    panel_axs = [ax_a, ax_b]
    if not add_barbs:
        panel_axs += [ax_v, ax_w]
    if add_vort:
        panel_axs += [ax_vort]
    panel_letters = list('bcdefgh')[:len(panel_axs)]
    for ax, letter in zip(panel_axs, panel_letters):
        ax.text(0.015, 0.93, f'({letter})',
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                zorder=20,
                bbox=dict(facecolor='white', alpha=1.,
                          edgecolor='k', linewidth=1.2, pad=4))

        # mark where the P-3 flight path crosses this cross-section
        if cross_x is not None:
            ax.axvline(x=cross_x, color='firebrick', ls='--', lw=1.4, zorder=4)
            if ax is ax_a:
                xn_cross = (cross_x - xmin) / (xmax - xmin)
                ax.text(xn_cross, 1.01, 'x', color='red', fontsize=12,
                        fontweight='bold', transform=ax.transAxes,
                        ha='center', va='bottom', zorder=21, clip_on=False)

    # ---- filename -------------------------------------------------------
    os.makedirs(path_figures, exist_ok=True)
    suffix = ''
    if add_barbs:
        suffix += '_wind_barbs'
    if convective_axis_deg is not None:
        suffix += f'_along_across_{convective_axis_deg:.0f}deg'
    if add_vort:
        suffix += '_vort'
    fname = f'curtain_across_pass_{passind}{suffix}_{save_suffix}.png'
    plt.savefig(path_figures + fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}')
