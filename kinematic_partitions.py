#%%
"""
Kinematic (CFAD) partitions by TDR precipitation classification.

Pulls the raw 2-D TDR level-2 volumes (full swath, not just the along-track
curtain) for the 3 flight passes, groups every grid cell into:
  - "weak/stratiform"  : rainfall_classification in {0 (weak echo), 1 (stratiform)}
  - "convective"       : rainfall_classification in {2 (shallow), 3 (moderate),
                                                      4 (deep)}
and builds contoured-frequency-by-altitude-diagrams (CFADs) of reflectivity
and winds for each group.

complete_partitions(...) is the entry point, called from run.py.
"""
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from datetime import datetime
import pandas as pd

path_project = Path(__file__).parent.parent
path_figures = str(path_project) + "/figures/"
path_data    = str(path_project) + "/data/"



# raw TDR level-2 volumes (same source used in tdr_analysis_new_cross.py)
TDR_L2_PATH = '/Users/ethanmurray/files-research-postdoc/data/aew/tdr/level2-all/20240715I1/'
# full 5-volume order used as the num_cases index in the rainfall-classification file
FILE_NAMES = ['240715I1_1804_xy.nc', '240715I1_1838_xy.nc', '240715I1_1919_xy.nc',
              '240715I1_1957_xy.nc', '240715I1_2030_xy.nc']
CLASS_PATH = '/Users/ethanmurray/files-research-postdoc/data/aew/tdr/precip-classifications/'
CLASS_FILE = 'tdr_aew_rainfall_classifications.nc'

# the 3 flight passes represented by Figure 4's radar_passes windows
# (17.8-18.3, 18.95-19.45, 20.2-20.65 UTC); 1838 and 1957 are transitional
# volumes between passes and are skipped
PASS_FILES  = ['240715I1_1804_xy.nc', '240715I1_1919_xy.nc', '240715I1_2030_xy.nc']
PASS_LABELS = ['Pass 1 (1804 UTC)', 'Pass 2 (1919 UTC)', 'Pass 3 (2030 UTC)']
# time windows (decimal hours UTC) associated with each pass file above —
# same windows used for the flight-track overlay in tdr_radar_figure
PASS_TIME_WINDOWS = [(17.8, 18.3), (18.95, 19.45), (20.2, 20.65)]

FLIGHT_LEVEL_PATH = '/Users/ethanmurray/files-research-postdoc/data/aew/flight-level/'
FLIGHT_LEVEL_FILE = '20240715I1_A_small.nc'

# rainfall_classification integer codes
WEAK_STRAT_CODES  = (0, 1)   # weak echo, stratiform
CONVECTIVE_CODES  = (2, 3, 4)  # shallow, moderate, deep convective




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

def _load_pass(tdr_file, class_ds):
    """
    Load one raw TDR volume plus its aligned 2-D rainfall classification.

    Returns a dict with 'level' (km), and (y, x, level) arrays for
    'REFLECTIVITY', 'U', 'V', 'W' (m/s, NOT yet converted to knots), a
    (y, x) 'class' array of rainfall-classification codes, and (y, x)
    'lat'/'lon' grids (for nearest-neighbor lookups from flight positions).

    NOTE: the classification field's native dims are
    (northward_distance, eastward_distance) = (y, x). Earlier code
    transposed the classification to (x, y) to match the raw volume's
    (x, y)-ordered REFLECTIVITY/LATITUDE/LONGITUDE — this turned out to
    mis-align the two grids. Fixed here by instead transposing the raw
    volume's fields to (y, x) so everything shares the classification's
    native orientation.
    """
    ds = xr.open_dataset(TDR_L2_PATH + tdr_file)
    level = ds['level'].values

    refl = ds['REFLECTIVITY'].isel(time=0).values.transpose(1, 0, 2)   # (y, x, level)
    u    = ds['U'].isel(time=0).values.transpose(1, 0, 2)
    v    = ds['V'].isel(time=0).values.transpose(1, 0, 2)
    w    = ds['W'].isel(time=0).values.transpose(1, 0, 2)
    lat  = ds['LATITUDE'].isel(time=0).values.T        # (y, x)
    lon  = ds['LONGITUDE'].isel(time=0).values.T        # (y, x)

    case_i = FILE_NAMES.index(tdr_file)
    class_2d = class_ds['rainfall_classification'].isel(num_cases=case_i).values  # (y, x)

    return {'level': level, 'REFLECTIVITY': refl, 'U': u, 'V': v, 'W': w,
            'class': class_2d, 'lat': lat, 'lon': lon}


def _grouped_values(pass_data_list, var, group_codes, convective_axis_deg=None,
                    component=None):
    """
    Pool a variable across passes for grid cells whose classification is in
    ``group_codes``. Returns (values (n_points, n_level), level array).

    var : {'REFLECTIVITY', 'U', 'V', 'W'}
    component : {None, 'along', 'across'}
        When convective_axis_deg is set and var is 'U' or 'V', selects the
        rotated along-axis or across-axis wind component instead of the raw
        field (mirrors the U'/V' rotation used elsewhere: along-axis =
        U*sin(theta) + V*cos(theta), across-axis = U*cos(theta) - V*sin(theta)).
    """
    level = pass_data_list[0]['level']
    rows = []
    for pd_ in pass_data_list:
        mask = np.isin(pd_['class'], group_codes)   # (y, x)
        if component is not None:
            theta = np.deg2rad(convective_axis_deg)
            if component == 'along':
                data = pd_['U'] * np.sin(theta) + pd_['V'] * np.cos(theta)
            else:  # 'across'
                data = pd_['U'] * np.cos(theta) - pd_['V'] * np.sin(theta)
        else:
            data = pd_[var]
        # data: (x, y, level); select classified cells -> (n_points, level)
        rows.append(data[mask, :])
    return np.concatenate(rows, axis=0), level


def _compute_cfad(values, level, bins):
    """
    values : (n_points, n_level) array (may contain NaN)
    Returns a (n_level, n_bins) probability array, each row normalized to
    sum to 1 (standard CFAD convention: probability distribution per height).
    Empty bins (0 probability) are left as NaN rather than 0, so they don't
    get colored in like real (near-)zero-probability data — cleaner plots.
    """
    n_level = values.shape[1]
    n_bins  = len(bins) - 1
    cfad = np.full((n_level, n_bins), np.nan)
    for li in range(n_level):
        col = values[:, li]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        hist, _ = np.histogram(col, bins=bins)
        total = hist.sum()
        if total > 0:
            row = hist / total
            row[hist == 0] = np.nan
            cfad[li, :] = row
    return cfad


DATA_HEIGHT_CUTOFF_KM = 13.   # data above this height is dropped (unreliable / not of interest)
PLOT_YLIM_KM          = 15.   # all CFAD y-axes are locked to this, regardless of data cutoff


def _plot_cfad(ax, cfad, bins, level, label, cmap='plasma', vmax=None, log_scale=False):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cfad = cfad.copy()
    cfad[level > DATA_HEIGHT_CUTOFF_KM, :] = np.nan
    if log_scale:
        vmin = np.nanmin(cfad[cfad > 0]) if np.any(cfad > 0) else 1e-4
        norm = LogNorm(vmin=vmin, vmax=vmax)
        p = ax.pcolormesh(bin_centers, level, cfad, cmap=cmap, norm=norm, shading='auto')
    else:
        p = ax.pcolormesh(bin_centers, level, cfad, cmap=cmap, vmin=0, vmax=vmax,
                          shading='auto')
    ax.set_xlabel(label)
    ax.set_ylabel('Height (km)')
    ax.set_ylim([0, PLOT_YLIM_KM])
    return p


# variable definitions: (key, label, bins, component)
def _base_variables():
    return [
        ('REFLECTIVITY', None, 'TDR Reflectivity (dBZ)', np.arange(-10, 61, 2), None),
        ('U', None, 'Zonal Wind U (kt)', np.arange(-60, 61, 2), None),
        ('V', None, 'Meridional Wind V (kt)', np.arange(-60, 61, 2), None),
        ('W', None, 'Vertical Velocity W (kt)', np.arange(-20, 21, 1), None),
    ]


def _rotated_variables(convective_axis_deg):
    return [
        ('U', 'along', "TDR v' (Along Long Axis, kt)", np.arange(-60, 61, 2), 'along'),
        ('U', 'across', "TDR u' (Across Long Axis, kt)", np.arange(-60, 61, 2), 'across'),
    ]


def complete_partitions(convective_axis_deg=None, per_leg=False, wind_units='kt',
                        log_scale=False):
    """
    Build CFAD diagrams of TDR reflectivity and winds, partitioned by
    precipitation classification (weak echo + stratiform vs. shallow +
    moderate + deep convective), pooling all 3 flight passes together.

    convective_axis_deg : float or None
        When set, adds along-axis / across-axis rotated-wind CFADs (rotated
        by this many degrees CW from north) alongside the raw zonal/
        meridional wind CFADs.
    per_leg : bool
        When True, also makes one figure per variable comparing CFADs
        (all classifications combined) across the 3 individual flight
        passes, to see how convective properties evolve leg to leg.
    log_scale : bool
        When True, colors the CFAD probability axis on a log scale instead
        of linear — useful for seeing low-probability tails alongside the
        dominant mode.

    NOTE: Pass 1 (1804 UTC) is excluded from every wind variable (U, V, W,
    and the rotated along/across-axis components) due to sparse/improper
    sampling that pass — it is still included in the REFLECTIVITY CFADs.
    """
    os.makedirs(path_figures, exist_ok=True)

    class_ds = xr.open_dataset(CLASS_PATH + CLASS_FILE)
    pass_data = [_load_pass(f, class_ds) for f in PASS_FILES]

    knots = 1.944

    variables = list(_base_variables())
    if convective_axis_deg is not None:
        variables += _rotated_variables(convective_axis_deg)

    # ---- combined-passes figure: weak/strat vs convective, per variable ----
    for var, component, label, bins, comp_kw in variables:
        # Pass 1 excluded from wind variables (sparse/improper sampling)
        pass_data_use = pass_data if var == 'REFLECTIVITY' else pass_data[1:]

        values_ws, level = _grouped_values(pass_data_use, var, WEAK_STRAT_CODES,
                                           convective_axis_deg, comp_kw)
        values_cv, _     = _grouped_values(pass_data_use, var, CONVECTIVE_CODES,
                                           convective_axis_deg, comp_kw)

        if var != 'REFLECTIVITY' and wind_units == 'kt':
            values_ws = values_ws * knots
            values_cv = values_cv * knots

        cfad_ws = _compute_cfad(values_ws, level, bins)
        cfad_cv = _compute_cfad(values_cv, level, bins)
        vmax = np.nanmax([np.nanmax(cfad_ws), np.nanmax(cfad_cv)])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        p1 = _plot_cfad(ax1, cfad_ws, bins, level, label, vmax=vmax, log_scale=log_scale)
        ax1.set_title('Weak Echo + Stratiform')
        p2 = _plot_cfad(ax2, cfad_cv, bins, level, label, vmax=vmax, log_scale=log_scale)
        ax2.set_title('Convective (Shallow/Moderate/Deep)')
        ax2.set_ylabel('')
        fig.colorbar(p2, ax=[ax1, ax2], label='Probability', extend='max')

        suffix = component if component is not None else var.lower()
        fname = f'cfad_{suffix}_all_passes.png'
        plt.savefig(path_figures + "2d-dists/" + fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')

    # ---- per-leg figures: one variable per figure, one panel per pass
    # (3 panels for REFLECTIVITY, 2 for wind variables since Pass 1 is
    # excluded), classes combined (weak/strat + convective together) ------
    if per_leg:
        all_codes = WEAK_STRAT_CODES + CONVECTIVE_CODES
        for var, component, label, bins, comp_kw in variables:
            pass_data_use = pass_data if var == 'REFLECTIVITY' else pass_data[1:]
            labels_use     = PASS_LABELS if var == 'REFLECTIVITY' else PASS_LABELS[1:]

            fig, axs = plt.subplots(1, len(pass_data_use), figsize=(5 * len(pass_data_use), 5),
                                    sharey=True)
            axs = np.atleast_1d(axs)
            cfads = []
            for pd_ in pass_data_use:
                values, level = _grouped_values([pd_], var, all_codes,
                                                convective_axis_deg, comp_kw)
                if var != 'REFLECTIVITY':
                    values = values * knots
                cfads.append(_compute_cfad(values, level, bins))
            vmax = np.nanmax([np.nanmax(c) for c in cfads])

            for ax, cfad, plabel in zip(axs, cfads, labels_use):
                p = _plot_cfad(ax, cfad, bins, level, label, vmax=vmax, log_scale=log_scale)
                ax.set_title(plabel)
            for ax in axs[1:]:
                ax.set_ylabel('')
            fig.colorbar(p, ax=list(axs), label='Probability', extend='max')

            suffix = component if component is not None else var.lower()
            fname = f'cfad_{suffix}_per_leg.png'
            plt.savefig(path_figures + "2d-dists/" + fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {fname}')


# proxy legend entries describing the percentile-contour linestyles, shared
# by every _plot_cfad_contour call
_CONTOUR_LEGEND = [
    Line2D([0], [0], color='k', lw=2.2,        label='Mean'),
    Line2D([0], [0], color='k', lw=1.4, ls='--', label='25/75%'),
    Line2D([0], [0], color='k', lw=1.0, ls=':',  label='10/90%'),
    Line2D([0], [0], color='k', lw=0.7, ls='-.', label='1/99%'),
]


def _color_key_legend(names_colors):
    """Build proxy legend entries mapping a color to what it represents
    (e.g. class or pass), for combined/overlaid contour plots."""
    return [Line2D([0], [0], color=color, lw=2.2, label=name)
            for name, color in names_colors]


def _dual_legend(ax, color_handles, loc='upper right', fontsize=8):
    """
    Two-column legend for overlaid contour plots: one column for the
    percentile/linestyle key (_CONTOUR_LEGEND), one for the color key
    (class or pass). Drawn as two side-by-side opaque legend boxes rather
    than a single ncol=2 legend, so each column stays a clean, distinct
    group instead of interleaving.
    """
    leg1 = ax.legend(handles=_CONTOUR_LEGEND, fontsize=fontsize, loc=loc,
                     framealpha=1., bbox_to_anchor=(1., 1.))
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, fontsize=fontsize, loc=loc,
             framealpha=1., bbox_to_anchor=(0.72, 1.))


def _plot_cfad_contour(ax, values, level, label, color, name=None):
    """
    Percentile-contour alternative to _plot_cfad: instead of a shaded 2-D
    probability field, draws per-height summary curves (all in ``color``)
    of the value distribution: mean (solid, thick), 25th/75th percentile
    (dashed), 10th/90th (dotted), 1st/99th (dash-dot).

    values : (n_points, n_level) array (may contain NaN)
    """
    n_level = values.shape[1]
    mean = np.full(n_level, np.nan)
    p1, p10, p25, p75, p90, p99 = (np.full(n_level, np.nan) for _ in range(6))
    for li in range(n_level):
        if level[li] > DATA_HEIGHT_CUTOFF_KM:
            continue
        col = values[:, li]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        mean[li] = np.mean(col)
        p1[li], p10[li], p25[li], p75[li], p90[li], p99[li] = np.percentile(
            col, [1, 10, 25, 75, 90, 99])

    ax.plot(mean, level, color=color, lw=2.2, label=name)
    ax.plot(p25, level, color=color, lw=1.4, ls='--')
    ax.plot(p75, level, color=color, lw=1.4, ls='--')
    ax.plot(p10, level, color=color, lw=1.0, ls=':')
    ax.plot(p90, level, color=color, lw=1.0, ls=':')
    ax.plot(p1, level, color=color, lw=0.7, ls='-.')
    ax.plot(p99, level, color=color, lw=0.7, ls='-.')

    ax.set_xlabel(label)
    ax.set_ylabel('Height (km)')
    ax.set_ylim([0, PLOT_YLIM_KM])


def complete_partitions_contour(convective_axis_deg=None, per_leg=False, wind_units='kt'):
    """
    Percentile-contour version of complete_partitions(): same data,
    grouping, variable list, and subplot layout, but each CFAD panel is
    drawn as mean/25-75%/10-90%/1-99% contour lines (via
    _plot_cfad_contour) instead of a shaded 2-D probability field. Saved to
    the 2d-dists-contour folder.

    convective_axis_deg, per_leg, wind_units : see complete_partitions().

    NOTE: Pass 1 (1804 UTC) is excluded from every wind variable (U, V, W,
    and the rotated along/across-axis components) due to sparse/improper
    sampling that pass — it is still included in the REFLECTIVITY CFADs.
    """
    os.makedirs(path_figures + "2d-dists-contour/", exist_ok=True)

    class_ds = xr.open_dataset(CLASS_PATH + CLASS_FILE)
    pass_data = [_load_pass(f, class_ds) for f in PASS_FILES]

    knots = 1.944

    variables = list(_base_variables())
    if convective_axis_deg is not None:
        variables += _rotated_variables(convective_axis_deg)

    # ---- combined-passes figure: weak/strat vs convective plotted atop one
    # another on a single panel, plus the same two classes plotted in
    # separate panels for a cleaner look at each individually --------------
    for var, component, label, bins, comp_kw in variables:
        # Pass 1 excluded from wind variables (sparse/improper sampling)
        pass_data_use = pass_data if var == 'REFLECTIVITY' else pass_data[1:]

        values_ws, level = _grouped_values(pass_data_use, var, WEAK_STRAT_CODES,
                                           convective_axis_deg, comp_kw)
        values_cv, _     = _grouped_values(pass_data_use, var, CONVECTIVE_CODES,
                                           convective_axis_deg, comp_kw)

        if var != 'REFLECTIVITY' and wind_units == 'kt':
            values_ws = values_ws * knots
            values_cv = values_cv * knots

        # -- overlaid: both classes atop one another on a single panel -----
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_cfad_contour(ax, values_ws, level, label, color='tab:blue',
                           name='Weak Echo + Stratiform')
        _plot_cfad_contour(ax, values_cv, level, label, color='tab:red',
                           name='Convective')
        class_legend = _color_key_legend([('Weak Echo + Stratiform', 'tab:blue'),
                                          ('Convective', 'tab:red')])
        _dual_legend(ax, class_legend)

        suffix = component if component is not None else var.lower()
        fname = f'cfad_{suffix}_all_passes_overlay.png'
        plt.savefig(path_figures + "2d-dists-contour/" + fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')

        # -- separate: each class in its own panel, as before --------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        _plot_cfad_contour(ax1, values_ws, level, label, color='tab:blue')
        ax1.set_title('Weak Echo + Stratiform')
        ax1.legend(handles=_CONTOUR_LEGEND, fontsize=8, loc='upper right', framealpha=1.)
        _plot_cfad_contour(ax2, values_cv, level, label, color='tab:red')
        ax2.set_title('Convective (Shallow/Moderate/Deep)')
        ax2.set_ylabel('')

        fname = f'cfad_{suffix}_all_passes.png'
        plt.savefig(path_figures + "2d-dists-contour/" + fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')

    # ---- per-leg figures: classes combined (weak/strat + convective
    # together — leg-to-leg comparison matters more here than class here),
    # colored per pass using the same palette as plot_tdr_statistics()
    # (Pass 1/2/3 -> steelblue/darkorange/firebrick). Saved both overlaid
    # (all passes atop one another on a single panel) and separately (one
    # panel per pass, as in the original 2d CFADs' per-leg breakdown). Pass
    # 1 is excluded for wind variables, so those get 2 passes instead of 3 --
    if per_leg:
        all_codes = WEAK_STRAT_CODES + CONVECTIVE_CODES
        all_pass_colors = ['steelblue', 'darkorange', 'firebrick']
        for var, component, label, bins, comp_kw in variables:
            pass_data_use   = pass_data   if var == 'REFLECTIVITY' else pass_data[1:]
            labels_use      = PASS_LABELS if var == 'REFLECTIVITY' else PASS_LABELS[1:]
            pass_colors_use = all_pass_colors if var == 'REFLECTIVITY' else all_pass_colors[1:]

            pass_values = []
            for pd_ in pass_data_use:
                values, level = _grouped_values([pd_], var, all_codes,
                                                convective_axis_deg, comp_kw)
                if var != 'REFLECTIVITY':
                    values = values * knots
                pass_values.append(values)

            suffix = component if component is not None else var.lower()

            # -- overlaid: all passes atop one another on a single panel ----
            fig, ax = plt.subplots(figsize=(6, 5))
            for values, plabel, color in zip(pass_values, labels_use, pass_colors_use):
                _plot_cfad_contour(ax, values, level, label, color=color, name=plabel)
            pass_legend = _color_key_legend(list(zip(labels_use, pass_colors_use)))
            _dual_legend(ax, pass_legend)

            fname = f'cfad_{suffix}_per_leg_overlay.png'
            plt.savefig(path_figures + "2d-dists-contour/" + fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {fname}')

            # -- separate: each pass in its own panel, as before ------------
            fig, axs = plt.subplots(1, len(pass_data_use), figsize=(5 * len(pass_data_use), 5),
                                    sharey=True)
            axs = np.atleast_1d(axs)
            for ax, values, plabel, color in zip(axs, pass_values, labels_use, pass_colors_use):
                _plot_cfad_contour(ax, values, level, label, color=color)
                ax.set_title(plabel)
            for ax in axs[1:]:
                ax.set_ylabel('')
            axs[0].legend(handles=_CONTOUR_LEGEND, fontsize=8, loc='upper right', framealpha=1.)

            fname = f'cfad_{suffix}_per_leg.png'
            plt.savefig(path_figures + "2d-dists-contour/" + fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {fname}')


def complete_partitions_contour_nice():
    """
    Figure 9: per-leg percentile-contour CFADs of (a) TDR reflectivity and
    (b) vertical velocity W, side by side, all passes overlaid in each panel
    (colored per pass). Pass 1 is excluded from W (sparse sampling), as in
    complete_partitions_contour(). Saved as figure9.png in the main figures dir.
    """
    os.makedirs(path_figures, exist_ok=True)

    class_ds  = xr.open_dataset(CLASS_PATH + CLASS_FILE)
    pass_data = [_load_pass(f, class_ds) for f in PASS_FILES]

    knots       = 1.944
    all_codes   = WEAK_STRAT_CODES + CONVECTIVE_CODES
    all_colors  = ['steelblue', 'darkorange', 'firebrick']
    label_fs    = plt.rcParams['axes.labelsize']

    panels = [('REFLECTIVITY', 'TDR Reflectivity (dBZ)', 'a'),
              ('W',            'Vertical Velocity W (m s$^{-1}$)', 'b')]

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (var, label, letter) in zip(axs, panels):
        pass_data_use = pass_data   if var == 'REFLECTIVITY' else pass_data[1:]
        labels_use    = PASS_LABELS if var == 'REFLECTIVITY' else PASS_LABELS[1:]
        colors_use    = all_colors  if var == 'REFLECTIVITY' else all_colors[1:]

        for pd_, plabel, color in zip(pass_data_use, labels_use, colors_use):
            values, level = _grouped_values([pd_], var, all_codes, None, None)
            # W stays in m/s (raw TDR units); no knots conversion
            _plot_cfad_contour(ax, values, level, label, color=color, name=plabel)

        if letter == 'a':   # legend (percentile key + passes) on panel (a) only
            _dual_legend(ax, _color_key_legend(list(zip(labels_use, colors_use))))
        ax.text(0.03, 0.96, f'({letter})', transform=ax.transAxes,
                ha='left', va='top', fontsize=label_fs, zorder=20,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                          linewidth=1.2, pad=4))
        if letter == 'b':
            ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(path_figures + 'figure9.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved figure9.png')


# ---------------------------------------------------------------------------
# Flight-level distributions by TDR precipitation classification
# ---------------------------------------------------------------------------

# flight-level variables to build distributions for: (key, label)
FLIGHT_VARIABLES = [
    ('HUM_REL.d', 'Relative Humidity (%)'),
    ('UWZ.d',     'Vertical Velocity W (m s$^{-1}$)'),
    ('UWX.d',     'Zonal Wind U (m s$^{-1}$)'),
    ('UWY.d',     'Meridional Wind V (m s$^{-1}$)'),
    ('MR.d',      'Mixing Ratio (g kg$^{-1}$)'),
    ('TA.d',      'Air Temperature (°C)'),
]


def _nearest_classification(fl_lat, fl_lon, pd_):
    """
    For each flight-level point (fl_lat, fl_lon), find the nearest TDR grid
    cell in pd_['lat']/pd_['lon'] and return the rainfall_classification
    code there (same nearest-neighbor approach as build_cross_curtain).
    """
    tdr_lat = pd_['lat']   # (y, x)
    tdr_lon = pd_['lon']
    flat_lat = tdr_lat.ravel()
    flat_lon = tdr_lon.ravel()

    # manual edit: data need to be flipped before analysis!
    flat_class = pd_['class'].transpose().ravel()

    codes = np.full(fl_lat.shape, np.nan)
    for i, (la, lo) in enumerate(zip(fl_lat, fl_lon)):
        dist2 = (flat_lat - la)**2 + (flat_lon - lo)**2
        if np.all(np.isnan(dist2)):
            continue
        codes[i] = flat_class[np.nanargmin(dist2)]
    return codes


def _plot_distribution(ax, values_ws, values_cv, bins, label):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for values, color, name in [(values_ws, 'tab:blue', 'Weak Echo + Stratiform'),
                                (values_cv, 'tab:red', 'Convective')]:
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        hist, _ = np.histogram(values, bins=bins)
        prob = hist / hist.sum()
        ax.plot(bin_centers, prob, color=color, lw=1.8, label=name)
    ax.set_xlabel(label, fontsize=13, labelpad=10)
    ax.set_ylabel('Probability', fontsize=13, labelpad=10)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(0, 0.35)


def _curtain_classification():
    """(time, rainfall_classification) from the pre-built along-track
    curtain — already correctly matched to the flight path."""
    curtain = xr.open_dataset(path_data + '/tdr/tdr_curtain_2160m.nc')
    return curtain['time'].values, curtain['rainfall_classification'].values


def _manual_classification(fl_time, fl_lat, fl_lon):
    """(time, rainfall_classification) via direct nearest-neighbor lat/lon
    matching against the raw TDR volumes, restricted to each pass's time
    window. Only returns points that fell within a window."""
    class_ds = xr.open_dataset(CLASS_PATH + CLASS_FILE)
    pass_data = [_load_pass(f, class_ds) for f in PASS_FILES]

    all_codes = np.full(fl_time.shape, np.nan)
    for (t0, t1), pd_ in zip(PASS_TIME_WINDOWS, pass_data):
        seg = (fl_time >= t0) & (fl_time <= t1)
        if not np.any(seg):
            continue
        all_codes[seg] = _nearest_classification(fl_lat[seg], fl_lon[seg], pd_)

    in_window = np.isfinite(all_codes)
    return fl_time[in_window], all_codes[in_window]


def _upscale_to_flight_resolution(class_time, class_codes, fl_time):
    """
    Broadcast a coarser (time, classification) series up to every native
    1 Hz flight-level sample, via nearest-neighbor in time. Flight samples
    farther than half the classification series' own point spacing from
    any classification point are left unmatched (NaN) rather than
    extrapolated.
    """
    order = np.argsort(class_time)
    class_time, class_codes = class_time[order], class_codes[order]
    spacing = np.median(np.diff(class_time))

    idx = np.clip(np.searchsorted(class_time, fl_time), 1, len(class_time) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(fl_time - class_time[left]) <= np.abs(fl_time - class_time[right])
    nearest = np.where(use_left, left, right)
    within_tol = np.abs(fl_time - class_time[nearest]) <= spacing / 2.

    codes = np.where(within_tol, class_codes[nearest], np.nan)
    return fl_time, codes


def complete_flightlevel_partitions(match_method='curtain', resolution='downscale'):
    """
    Build probability-distribution plots of flight-level variables
    (HUM_REL.d, UWZ.d, UWX.d, UWY.d, MR.d, TA.d), partitioned into weak
    echo + stratiform vs. shallow/moderate/deep convective (same
    classification codes used in complete_partitions). All 3 flight legs
    are pooled into a single distribution per group — no per-leg breakdown,
    unlike the TDR CFADs.

    match_method : {'curtain', 'manual'}
        'curtain' (recommended): classification comes from the pre-built
        tdr_curtain_2160m.nc curtain, already sampled along the flight path
        with its own time axis.
        'manual': classification is looked up directly by nearest-neighbor
        lat/lon matching against the raw TDR volumes (see _load_pass for
        the lat/lon-orientation fix).
    resolution : {'downscale', 'upscale'}
        'downscale' (recommended): pair flight-level variables at the
        classification series' own (coarser) time resolution — the curtain's
        ~2160 m along-track spacing, or the manual method's native flight
        1 Hz spacing within each pass window (already as fine as it gets).
        'upscale': broadcast the classification up to every native 1 Hz
        flight-level sample (only meaningful for match_method='curtain',
        since 'manual' is already at 1 Hz).
    """
    os.makedirs(path_figures, exist_ok=True)

    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')  # hours UTC

    if match_method == 'curtain':
        class_time, class_codes = _curtain_classification()
    elif match_method == 'manual':
        fl_lat = fl['LATref'].values
        fl_lon = fl['LONref'].values
        class_time, class_codes = _manual_classification(fl_time, fl_lat, fl_lon)
    else:
        raise ValueError("match_method must be 'curtain' or 'manual'")

    if resolution == 'downscale':
        sample_time = class_time
        ws_mask = np.isin(class_codes, WEAK_STRAT_CODES)
        cv_mask = np.isin(class_codes, CONVECTIVE_CODES)
        get_var = lambda var: np.interp(sample_time, fl_time, fl[var].values)
    elif resolution == 'upscale':
        sample_time, codes_upscaled = _upscale_to_flight_resolution(
            class_time, class_codes, fl_time)
        ws_mask = np.isin(codes_upscaled, WEAK_STRAT_CODES)
        cv_mask = np.isin(codes_upscaled, CONVECTIVE_CODES)
        get_var = lambda var: fl[var].values   # already at native resolution
    else:
        raise ValueError("resolution must be 'downscale' or 'upscale'")

    print(f'[{match_method}/{resolution}] points: {ws_mask.sum()} weak/stratiform, '
          f'{cv_mask.sum()} convective (of {len(sample_time)} candidate samples)')

    for var, label in FLIGHT_VARIABLES:
        data = get_var(var)
        if var == 'HUM_REL.d':
            # sensor noise pushes some readings slightly above 100% —
            # physically impossible, so cap (round down) at 100%
            data = np.clip(data, None, 100.)
        values_ws = data[ws_mask]
        values_cv = data[cv_mask]

        finite = data[ws_mask | cv_mask]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            print(f'No finite {var} values; skipping.')
            continue
        bins = np.linspace(np.nanmin(finite), np.nanmax(finite), 40)

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_distribution(ax, values_ws, values_cv, bins, label)
        ax.set_title(f'{match_method} matching, {resolution} resolution')
        ax.legend(framealpha=1.)

        fname = f'flightlevel_dist_{var.replace(".", "")}_{match_method}_{resolution}.png'
        plt.savefig(path_figures + "1d-dists/" + fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')


# fixed histogram bin widths for the flight-level distribution figures
FL_BIN_WIDTHS = {
    'UWZ.d':    0.25,   # W (m/s)
    'MR.d':     0.2,    # mixing ratio (g/kg)
    'TA.d':     0.1,    # temperature (deg C)
    'THETAE.d': 1.0,    # theta-e (K)
    'RH':       2.0,    # relative humidity (%)
}


def _fixed_bins(values, var):
    """Bin edges at multiples of the variable's fixed width, spanning the data."""
    step = FL_BIN_WIDTHS[var]
    lo = np.floor(np.nanmin(values) / step) * step
    hi = np.ceil(np.nanmax(values) / step) * step
    return np.arange(lo, hi + step / 2, step)


# ---------------------------------------------------------------------------
# Buffer zone vs. SAL flight-level analysis
# ---------------------------------------------------------------------------

# decimal-hour UTC windows (placeholders; adjust as needed)
BUFFER_WINDOWS = [(19.42, 19.505), (19.835, 19.9)]
SAL_WINDOWS    = [(19.505, 19.835)]
FL_PRES_VAR    = 'PS.c'   # corrected flight-level static pressure (mb); only in the full file
FL_FULL_FILE   = '20240715I1_A.nc'
BUFFER_COLOR   = 'gold'
SAL_COLOR      = 'orangered'


def flightlevel_rh_thetae(fl):
    """
    Recompute (theta_e [K], RH [%]) from flight-level T (TA.d, degC), water vapor
    mixing ratio (MR.d, g/kg) and static pressure (FL_PRES_VAR, hPa), rather than
    using the file's HUM_REL.d / THETAE.d. RH is capped at 100%.
    """
    import metpy.calc as mpcalc
    from metpy.units import units

    if FL_PRES_VAR in fl.variables:
        p_vals = fl[FL_PRES_VAR].values
    else:
        # the _small file drops flight-level pressure; pull it from the full file
        # (same 1 Hz time axis, so arrays line up sample-for-sample)
        full = xr.open_dataset(FLIGHT_LEVEL_PATH + FL_FULL_FILE, decode_times=False)
        p_vals = full[FL_PRES_VAR].values
        assert p_vals.shape == fl['TA.d'].values.shape, 'full/small file lengths differ'
    p = p_vals * units.hPa
    T = fl['TA.d'].values * units.degC
    w = (fl['MR.d'].values / 1000.) * units.dimensionless

    rh  = mpcalc.relative_humidity_from_mixing_ratio(p, T, w).to('percent').m
    Td  = mpcalc.dewpoint(mpcalc.vapor_pressure(p, w))
    the = mpcalc.equivalent_potential_temperature(p, T, Td).to('K').m
    return the, np.clip(rh, None, 100.)


def _window_mask(fl_time, windows):
    mask = np.zeros(fl_time.shape, dtype=bool)
    for t0, t1 in windows:
        mask |= (fl_time >= t0) & (fl_time < t1)
    return mask


def flightlevel_region_dists(show_east_west=False):
    """
    Figure 12: flight-level distributions of the four sampled regions
    (stratiform, convective, buffer zone, SAL) plotted atop one another.
    show_east_west : if True, also plots the airmasses ahead of (East of MCS,
    18.35-18.9 UTC; light grey) and behind (West of MCS, 19.9-20.25 UTC; dark
    grey) the MCS, in the distributions, legend, and map.
    2 rows x 3 columns: (a) W, (b) mixing ratio, (c) T, (d) theta-e, (e) RH,
    plus (f) a GOES map of the flight track color-coded by region.
    theta-e and RH are recomputed from T, MR, p. Saves figure12.png.
    """
    import cartopy.crs as ccrs
    import synoptic_overview

    os.makedirs(path_figures, exist_ok=True)

    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')

    # corrected classification: nearest TDR cell within a max distance, per pass window
    codes_upscaled, _ = _masked_classification(fl_time, fl['LATref'].values,
                                               fl['LONref'].values)
    ws_mask  = np.isin(codes_upscaled, WEAK_STRAT_CODES)
    cv_mask  = np.isin(codes_upscaled, CONVECTIVE_CODES)
    buf_mask = _window_mask(fl_time, BUFFER_WINDOWS)
    sal_mask = _window_mask(fl_time, SAL_WINDOWS)

    # (mask, color, name, linestyle); same line width for all; later entries on top
    regions = [
        (ws_mask,  'lightskyblue', 'Weak Echo + Stratiform', (0, (1, 1.2))),   # closely dotted
        (cv_mask,  'navy',         'Convective',             (0, (1, 1.2))),
        (buf_mask, BUFFER_COLOR,   'Buffer zone',            '-'),
        (sal_mask, SAL_COLOR,      'SAL',                    '-'),
    ]
    if show_east_west:
        east_mask = _window_mask(fl_time, [(18.35, 18.9)])
        west_mask = _window_mask(fl_time, [(19.9, 20.2)])
        regions += [(east_mask, '#b8b8b8', 'East of MCS', '-'),
                    (west_mask, '#505050', 'West of MCS', '-')]
    # bottom-to-top stacking order of the distribution lines (legend order unchanged)
    draw_order = ['West of MCS', 'East of MCS', 'SAL', 'Buffer zone',
                  'Weak Echo + Stratiform', 'Convective']
    any_mask = np.zeros_like(ws_mask, dtype=bool)
    for m, *_ in regions:
        any_mask |= m

    variables = [
        ('UWZ.d',     'Vertical Velocity W (m s$^{-1}$)'),
        ('MR.d',      'Mixing Ratio (g kg$^{-1}$)'),
        ('TA.d',      'Air Temperature (°C)'),
        ('THETAE.d',  'Equivalent Potential Temperature (K)'),
        ('RH',        'Relative Humidity (%)'),
    ]
    panel_letters = ['a', 'b', 'c', 'd', 'e']
    the_recalc, rh_recalc = flightlevel_rh_thetae(fl)

    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (var, label), letter in zip(axs.flat, variables, panel_letters):
        data = {'THETAE.d': the_recalc, 'RH': rh_recalc}.get(var)
        if data is None:
            data = fl[var].values
        finite = data[any_mask]
        finite = finite[np.isfinite(finite)]
        bins = _fixed_bins(finite, var)
        centers = 0.5 * (bins[:-1] + bins[1:])

        for mask, color, name, ls in regions:
            vals = data[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            hist, _ = np.histogram(vals, bins=bins)
            z = draw_order.index(name) + 2   # stacking: West of MCS first (bottom) ... convective last (top)
            if var == 'RH':
                # cumulative distribution (evaluated at upper bin edges)
                ax.plot(bins[1:], np.cumsum(hist) / hist.sum(), color=color,
                        lw=1.8, ls=ls, label=name, zorder=z)
            else:
                ax.plot(centers, hist / hist.sum(), color=color, lw=1.8, ls=ls,
                        label=name, zorder=z)

        ax.set_xlabel(label, fontsize=13, labelpad=10)
        ax.set_ylabel('Probability', fontsize=13, labelpad=10)
        ax.tick_params(axis='both', labelsize=11)
        ax.text(0.03, 0.96, f'({letter})', transform=ax.transAxes,
                ha='left', va='top', fontsize=13, zorder=20,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                          linewidth=1.2, pad=4))
        if letter == 'e':
            ax.set_ylim(0, 1.02)   # RH shown as a CDF (piles up near 100% in convection)
            ax.set_ylabel('CDF Probability', fontsize=13, labelpad=10)
            ax.set_xlim(0, 100)   # show the full RH range (empty values below ~20%)
        if letter in ('b', 'c'):
            ax.set_ylabel('')

    # legend lives inside panel (a)
    axs.flat[0].legend(loc='upper right', fontsize=10, framealpha=1.)

    # ---- panel (f): GOES map with flight track color-coded by region --------
    axs[1, 2].remove()
    pc = ccrs.PlateCarree()
    ax_map = fig.add_subplot(2, 3, 6, projection=pc)
    lons_map, lats_map = (-55, -53), (9, 12.75)
    # same GOES scene as figures 13 and 14 (goes-level2-tdr index 1, ~19:00 UTC)
    path_goes = str(path_project) + "/data/goes-level2-tdr/"
    goes_files = synoptic_overview.pull_files_helper(path_goes)
    ds_goes = synoptic_overview.data_subset(path_goes, goes_files[1], lats_map, lons_map)
    synoptic_overview.test_plot(ds_goes, channel='rgb', lats=lats_map, lons=lons_map,
                                ax=ax_map, fig=fig, show_colorbar=False,
                                show_lat=True, show_lon=True,
                                show_title_banner=False, show_time=False,
                                panel_label='f',
                                label_x=0.05, label_y=0.95)
    ax_map.set_xlabel("Longitude ($\\degree$)", fontsize=13)

    lon_fl, lat_fl = fl['LONref'].values, fl['LATref'].values
    ax_map.plot(lon_fl, lat_fl, color='w', lw=0.6, transform=pc, zorder=5)   # full track
    for mask, color, name, ls in regions:
        # NaN-out other samples so separate segments aren't joined by a line
        ax_map.plot(np.where(mask, lon_fl, np.nan), np.where(mask, lat_fl, np.nan),
                    color=color, lw=2.5, transform=pc, zorder=10)

    plt.tight_layout()
    plt.savefig(path_figures + 'figure12.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved figure12.png')



PLAN_VIEW_PATH = path_data + "tdr/plan-view/"


def _masked_classification(fl_time, fl_lat, fl_lon, max_dist_km=2.5, pad_hr=0.):
    """
    Flight-track rainfall classification matched directly to the raw TDR
    plan-view volumes, with a maximum-distance rejection.

    For each pass (PASS_FILES / PASS_TIME_WINDOWS, widened by pad_hr), every
    flight sample is matched to the nearest TDR grid cell that has a finite
    classification. Samples farther than max_dist_km from that cell (i.e.
    outside the scan swath) are left unclassified (NaN) instead of inheriting
    the class of the nearest edge cell. lat/lon/class use the same alignment
    as the figure-7 plotting code (no transposes).

    Returns (codes, dist_km), each the length of fl_time; NaN outside the
    pass windows.
    """
    from scipy.spatial import cKDTree

    class_ds = xr.open_dataset(CLASS_PATH + CLASS_FILE)
    codes = np.full(fl_time.shape, np.nan)
    dist  = np.full(fl_time.shape, np.nan)
    kx = 111.2 * np.cos(np.deg2rad(np.nanmean(fl_lat)))   # km per deg lon
    ky = 111.2                                             # km per deg lat

    for fname, (t0, t1) in zip(PASS_FILES, PASS_TIME_WINDOWS):
        seg = ((fl_time >= t0 - pad_hr) & (fl_time <= t1 + pad_hr)
               & np.isfinite(fl_lat) & np.isfinite(fl_lon))
        if not np.any(seg):
            continue
        ds = xr.open_dataset(PLAN_VIEW_PATH + fname)
        lat2d = ds['LATITUDE'].isel(time=0).values
        lon2d = ds['LONGITUDE'].isel(time=0).values
        cls = class_ds['rainfall_classification'].isel(
            num_cases=FILE_NAMES.index(fname)).values
        valid = np.isfinite(cls) & np.isfinite(lat2d) & np.isfinite(lon2d)

        tree = cKDTree(np.c_[lon2d[valid] * kx, lat2d[valid] * ky])
        d, idx = tree.query(np.c_[fl_lon[seg] * kx, fl_lat[seg] * ky])
        c = cls[valid][idx].astype(float)
        c[d > max_dist_km] = np.nan
        codes[seg] = c
        dist[seg]  = d
    return codes, dist


def tdr_class_check(pad_hr=0.25, max_dist_km=2.5):
    """
    Check of the flight-track classification used in figure 12.
    3 rows (TDR passes 1-3, as in figure 7) x 4 columns:
      1: figure-7 TDR rainfall classification with the P-3 track atop
      2: classes pulled from the curtain file (original method)
      3: distance from each flight sample to the nearest classified TDR cell
      4: corrected classes (raw volumes, rejected beyond max_dist_km, pass windows only)
    Tracks are shown for each pass window +/- pad_hr hours.
    Saves to figures-v-temp-2026-11-09-tdr-class-check/tdr_class_check.png
    """
    import cartopy.crs as ccrs
    import synoptic_overview
    import tdr_analysis
    from matplotlib.gridspec import GridSpec

    out_dir = str(path_project) + "/figures-v-temp-2026-11-09-tdr-class-check/"
    os.makedirs(out_dir, exist_ok=True)

    lons, lats = (-55., -53.), (9., 12.)
    pc = ccrs.PlateCarree()
    path_goes = path_data + "goes-level2-tdr/"
    pv_files  = ['240715I1_1804_xy.nc', '240715I1_1919_xy.nc', '240715I1_2030_xy.nc']
    row_titles = ['Pass 1: 18:04 UTC', 'Pass 2: 19:19 UTC', 'Pass 3: 20:30 UTC']
    class_indices = [0, 2, 4]

    goes_files = synoptic_overview.pull_files_helper(path_goes)
    tdr_datasets  = [xr.open_dataset(PLAN_VIEW_PATH + f) for f in pv_files]
    class_ds      = xr.open_dataset(CLASS_PATH + CLASS_FILE)
    goes_datasets = [synoptic_overview.data_subset(path_goes, f, lats, lons)
                     for f in goes_files]
    rain_cmap = tdr_analysis._rain_cmap()

    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')
    fl_lat, fl_lon = fl['LATref'].values, fl['LONref'].values
    class_time, class_codes = _curtain_classification()
    _, codes_curtain = _upscale_to_flight_resolution(class_time, class_codes, fl_time)
    codes_fixed, _ = _masked_classification(fl_time, fl_lat, fl_lon,
                                            max_dist_km=max_dist_km)
    _, dist_pad = _masked_classification(fl_time, fl_lat, fl_lon,
                                         max_dist_km=max_dist_km, pad_hr=pad_hr)

    asp = synoptic_overview._panel_aspect(lons, lats)
    ph = 3.5
    pw = ph * asp
    cbar_w = 0.2
    fig = plt.figure(figsize=(pw * 4 + cbar_w * 2, ph * 3))
    gs = GridSpec(3, 6, figure=fig,
                  width_ratios=[pw, pw, pw, cbar_w, pw, cbar_w],
                  wspace=0.15, hspace=0.08)
    label_fs = plt.rcParams['axes.labelsize']
    col_titles = ['TDR classification', 'Curtain-pulled classes',
                  'Distance to nearest TDR cell', 'Corrected classes']
    gs_cols = [0, 1, 2, 4]
    letters = 'abcdefghijkl'

    sc_cls = sc_dist = None
    for row, ((t0, t1), tdr_ds, goes_ds, ci, rtitle) in enumerate(
            zip(PASS_TIME_WINDOWS, tdr_datasets, goes_datasets,
                class_indices, row_titles)):
        seg = (fl_time >= t0 - pad_hr) & (fl_time <= t1 + pad_hr)

        for col in range(4):
            ax = fig.add_subplot(gs[row, gs_cols[col]], projection=pc)
            lab = letters[row * 4 + col]
            title = col_titles[col] if row == 0 else None
            if col == 0:
                tdr_analysis._plot_class_panel(ax, class_ds, tdr_ds, ci, lons, lats,
                                               rain_cmap, goes_ds=goes_ds,
                                               panel_label=lab, title=title)
                ax.plot(fl_lon[seg], fl_lat[seg], color='w', lw=1.5, zorder=6,
                        transform=pc)
            else:
                tdr_analysis._plot_goes_background(ax, goes_ds)
                tdr_analysis._decorate_ax(ax, lons, lats, panel_label=lab, title=title)
                if col == 2:
                    ok = seg & np.isfinite(dist_pad)
                    sc_dist = ax.scatter(fl_lon[ok], fl_lat[ok], c=dist_pad[ok],
                                         cmap='magma_r', vmin=0, vmax=10, s=8,
                                         zorder=6, transform=pc, edgecolors='none')
                else:
                    codes = codes_curtain if col == 1 else codes_fixed
                    nocls = seg & ~np.isfinite(codes)
                    ax.scatter(fl_lon[nocls], fl_lat[nocls], c='0.6', s=4,
                               zorder=5, transform=pc)
                    cls = seg & np.isfinite(codes)
                    sc_cls = ax.scatter(fl_lon[cls], fl_lat[cls], c=codes[cls],
                                        cmap=rain_cmap, vmin=-0.5, vmax=4.5, s=8,
                                        zorder=6, transform=pc, edgecolors='none')

            ax.set_xlabel('')
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Longitude (°)', fontsize=label_fs)
            if col == 0:
                ax.set_ylabel(f'{rtitle}\nLatitude (°)', fontsize=label_fs)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

    cax_d = fig.add_subplot(gs[:, 3])
    cb_d = fig.colorbar(sc_dist, cax=cax_d, orientation='vertical', extend='max')
    cb_d.set_label('Distance to nearest TDR cell (km)', fontsize=label_fs)

    cax_c = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(sc_cls, cax=cax_c, orientation='vertical', ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(['Weak Echo', 'Stratiform', 'Shallow Conv.',
                           'Mod. Conv.', 'Strong Conv.'])
    cb.set_label('Rainfall Classification', fontsize=label_fs)

    plt.savefig(out_dir + 'tdr_class_check.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved tdr_class_check.png')


def tdr_offmcs_mask_check(buffer_km=10., dbz_thresh=10., lons=(-56.5, -51.5),
                          lats=(8., 13.)):
    """
    Test figure: mask the MCS out of the two off-MCS TDR volumes.
    2 rows x 3 columns (2 km reflectivity):
      row 1: 1919 volume -> mask applied to the 1838 volume
      row 2: 2030 volume -> mask applied to the 1957 volume
      col 1: original-pass reflectivity (MCS mask outline in white)
      col 2: off-MCS pass reflectivity with the MCS mask (solid) and the
             buffer (dashed) atop it
      col 3: off-MCS reflectivity remaining after masking MCS + buffer
    MCS mask = largest contiguous region of 2 km reflectivity >= dbz_thresh
    in the original-pass volume. Saves to
    figures-v-temp-2026-11-09-tdr-class-check/tdr_offmcs_mask_check.png
    """
    import cartopy.crs as ccrs
    import tdr_analysis
    import synoptic_overview
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from matplotlib.gridspec import GridSpec

    out_dir = str(path_project) + "/figures-v-temp-2026-11-09-tdr-class-check/"
    os.makedirs(out_dir, exist_ok=True)
    pc = ccrs.PlateCarree()
    pairs = [('240715I1_1919_xy.nc', '240715I1_1838_xy.nc'),
             ('240715I1_2030_xy.nc', '240715I1_1957_xy.nc')]
    names = [('1919 UTC', '1838 UTC'), ('2030 UTC', '1957 UTC')]
    kx = 111.2 * np.cos(np.deg2rad(0.5 * (lats[0] + lats[1])))
    ky = 111.2

    def _load2km(fname):
        ds = xr.open_dataset(PLAN_VIEW_PATH + fname)
        lev = int(np.argmin(np.abs(ds['level'].values - 2.)))
        lat2d = ds['LATITUDE'].isel(time=0).values
        lon2d = ds['LONGITUDE'].isel(time=0).values
        dbz = ds['REFLECTIVITY'].isel(time=0, level=lev).values
        dbz = np.where(dbz < -998., np.nan, dbz)
        in_dom = ((lon2d >= lons[0]) & (lon2d <= lons[1]) &
                  (lat2d >= lats[0]) & (lat2d <= lats[1]))
        return lat2d, lon2d, np.where(in_dom, dbz, np.nan)

    asp = synoptic_overview._panel_aspect(lons, lats)
    ph = 3.5
    pw = ph * asp
    cbar_w = 0.2
    fig = plt.figure(figsize=(pw * 3 + cbar_w, ph * 2))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[pw, pw, pw, cbar_w],
                  wspace=0.12, hspace=0.08)
    label_fs = plt.rcParams['axes.labelsize']
    col_titles = ['Original pass', 'Off-MCS pass + MCS mask',
                  f'Off-MCS pass, MCS + {buffer_km:.0f} km buffer removed']
    levels_dbz = np.arange(0, 46, 2.5)
    letters = 'abcdef'
    cf = None

    for row, ((f_orig, f_new), (n_orig, n_new)) in enumerate(zip(pairs, names)):
        lat_o, lon_o, dbz_o = _load2km(f_orig)
        lat_n, lon_n, dbz_n = _load2km(f_new)

        # MCS mask: largest contiguous region of echo >= dbz_thresh
        lab, nlab = ndimage.label(np.nan_to_num(dbz_o, nan=-999.) >= dbz_thresh,
                                  structure=np.ones((3, 3)))
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        mcs = lab == (1 + int(np.argmax(sizes)))

        # distance (km) from every off-MCS cell to the nearest MCS cell
        tree = cKDTree(np.c_[lon_o[mcs] * kx, lat_o[mcs] * ky])
        d_new, _ = tree.query(np.c_[lon_n.ravel() * kx, lat_n.ravel() * ky])
        d_new = d_new.reshape(lat_n.shape)
        remaining = np.where(d_new > buffer_km, dbz_n, np.nan)

        panels = [(lat_o, lon_o, dbz_o, n_orig), (lat_n, lon_n, dbz_n, n_new),
                  (lat_n, lon_n, remaining, n_new)]
        for col, (la, lo, dbz, nm) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col], projection=pc)
            cf = ax.contourf(lo, la, dbz, levels=levels_dbz, cmap='RdYlBu_r',
                             extend='both', transform=pc)
            tdr_analysis._decorate_ax(ax, lons, lats, panel_label=letters[row * 3 + col],
                                      title=col_titles[col] if row == 0 else None)
            if col < 2:   # MCS mask outline (from the original pass)
                ax.contour(lon_o, lat_o, mcs.astype(float), levels=[0.5],
                           colors='k', linewidths=1.2, transform=pc, zorder=8)
            if col == 1:  # buffer edge on the off-MCS grid
                ax.contour(lon_n, lat_n, d_new, levels=[buffer_km], colors='k',
                           linewidths=1.2, linestyles='--', transform=pc, zorder=8)
            ax.set_xlabel('')
            if row < 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Longitude (°)', fontsize=label_fs)
            if col == 0:
                ax.set_ylabel(f'{n_orig} mask -> {n_new}\nLatitude (°)', fontsize=label_fs)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(cf, cax=cax, orientation='vertical', extend='both')
    cb.set_label('2 km TDR Reflectivity (dBZ)', fontsize=label_fs)

    plt.savefig(out_dir + 'tdr_offmcs_mask_check.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved tdr_offmcs_mask_check.png')


def flightlevel_sal_buffer_spectra(nperseg_max=256):
    """
    Test plots: spectral analysis of 1 Hz flight-level mixing ratio, RH (recomputed)
    and vertical velocity within the buffer zone / SAL, looking for a dominant
    periodicity (gravity waves, ABL rolls, ...). Three figures, saved to
    figures-v-temp-2026-11-09-tdr-class-check/:
      spectra_buffer_sal.png : one contiguous span covering buffer zone + SAL
      spectra_buffer.png     : the two buffer-zone segments, analysed separately
      spectra_sal.png        : SAL only
    Each figure is 3 rows (MR, RH, W) x 2 columns: detrended time series (left)
    and Welch power spectrum (right, log-log) with a -5/3 reference slope, the
    peak of the variance-preserving spectrum (f*PSD) marked, and a top axis
    converting frequency to along-track wavelength using the mean ground speed.
    NaN gaps are linearly interpolated; the series is linearly detrended.
    """
    from scipy import signal

    out_dir = str(path_project) + "/figures-v-temp-2026-11-09-tdr-class-check/"
    os.makedirs(out_dir, exist_ok=True)

    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')
    _, rh = flightlevel_rh_thetae(fl)
    lat, lon = fl['LATref'].values, fl['LONref'].values
    dt = np.median(np.diff(fl_time)) * 3600.   # s
    fs = 1. / dt

    variables = [(fl['MR.d'].values, 'Mixing ratio (g kg$^{-1}$)'),
                 (rh,                'RH (%)'),
                 (fl['UWZ.d'].values, 'W (m s$^{-1}$)')]

    bufmin = min(w[0] for w in BUFFER_WINDOWS)
    # (file name, title, [(segment label, color, (t0, t1))], shade windows?)
    cases = [
        ('spectra_buffer_sal.png', 'Buffer zone + SAL',
         [('Buffer + SAL', 'k', (bufmin, BUFFER_WINDOWS[-1][1]))], True),
        ('spectra_buffer.png', 'Buffer zone',
         [(f'Buffer {i+1} ({w[0]:.2f}-{w[1]:.2f})', c, w)
          for i, (w, c) in enumerate(zip(BUFFER_WINDOWS, ['goldenrod', 'peru']))], False),
        ('spectra_sal.png', 'SAL',
         [('SAL', SAL_COLOR, SAL_WINDOWS[0])], False),
    ]

    label_fs = 12
    for fname, title, segments, shade in cases:
        # mean ground speed over the whole case span -> wavelength axis
        span = ((fl_time >= min(s[2][0] for s in segments)) &
                (fl_time <= max(s[2][1] for s in segments)))
        kx = 111.2e3 * np.cos(np.deg2rad(np.nanmean(lat[span])))
        step = np.hypot(np.diff(lon[span]) * kx, np.diff(lat[span]) * 111.2e3)
        speed = np.nansum(step) / (step.size * dt)   # m/s

        fig, axs = plt.subplots(3, 2, figsize=(13, 9),
                                gridspec_kw=dict(width_ratios=[1.3, 1], hspace=0.35,
                                                 wspace=0.22))
        for row, (data, vlabel) in enumerate(variables):
            ax_t, ax_s = axs[row]
            peak_txt = []
            for seg_i, (slabel, color, (t0, t1)) in enumerate(segments):
                m = (fl_time >= t0) & (fl_time <= t1)
                t, x = fl_time[m], data[m].astype(float)
                bad = ~np.isfinite(x)
                if bad.all():
                    continue
                x = np.interp(t, t[~bad], x[~bad])
                x = signal.detrend(x)

                ax_t.plot(t, x, color=color, lw=0.9, label=slabel)
                nper = min(nperseg_max, x.size)
                f, psd = signal.welch(x, fs=fs, window='hann', nperseg=nper,
                                      detrend='linear')
                f, psd = f[1:], psd[1:]
                ax_s.loglog(f, psd, color=color, lw=1.4, label=slabel)

                # dominant period: peak of f*PSD, skipping the 2 lowest resolved bins
                ok = f >= 2. * fs / nper
                if ok.any():
                    fpk = f[ok][np.argmax((f * psd)[ok])]
                    ax_s.axvline(fpk, color=color, ls=':', lw=1.)
                    peak_txt.append((color, f'{slabel}: {1./fpk:.0f} s '
                                     f'({speed / fpk / 1e3:.1f} km)'))
                    print(f'{title} | {vlabel} | {slabel}: peak period {1./fpk:.1f} s, '
                          f'wavelength {speed / fpk / 1e3:.2f} km '
                          f'(NaN fraction {bad.mean():.2f})')
                if seg_i == 0:   # -5/3 reference slope anchored to this spectrum
                    c53 = np.median(psd * f ** (5. / 3.))
                    ax_s.loglog(f, c53 * f ** (-5. / 3.), color='0.5', ls='--', lw=1.,
                                label='$f^{-5/3}$')

            if shade:
                for w in BUFFER_WINDOWS:
                    ax_t.axvspan(*w, color=BUFFER_COLOR, alpha=0.35, lw=0)
                for w in SAL_WINDOWS:
                    ax_t.axvspan(*w, color=SAL_COLOR, alpha=0.2, lw=0)

            for k, (c, txt) in enumerate(peak_txt):
                ax_s.text(0.03, 0.05 + 0.09 * k, txt, transform=ax_s.transAxes,
                          color=c, fontsize=9, va='bottom')
            ax_t.set_ylabel(f'{vlabel}\n(detrended)', fontsize=label_fs)
            ax_s.set_ylabel('PSD', fontsize=label_fs)
            ax_s.set_xlabel('Frequency (Hz)', fontsize=label_fs)
            ax_t.set_xlabel('Time (decimal UTC hours)', fontsize=label_fs)
            for ax, lab in zip((ax_t, ax_s), ('abcdefgh'[2 * row], 'abcdefgh'[2 * row + 1])):
                ax.text(0.02, 0.95, f'({lab})', transform=ax.transAxes, ha='left',
                        va='top', fontsize=12, zorder=20,
                        bbox=dict(facecolor='white', edgecolor='k', linewidth=1., pad=3))
            if row == 0:
                ax_t.set_title(title, fontsize=13)
                ax_t.legend(loc='upper right', fontsize=8, framealpha=1.)
                ax_s.legend(loc='upper right', fontsize=8, framealpha=1.)
                sec = ax_s.secondary_xaxis(
                    'top', functions=(lambda f_: speed / f_ / 1e3,
                                      lambda L_: speed / L_ / 1e3))
                sec.set_xlabel(f'Wavelength (km), ground speed {speed:.0f} m s$^{{-1}}$',
                               fontsize=10)

        plt.savefig(out_dir + fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')
