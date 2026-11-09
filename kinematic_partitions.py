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
              ('W',            'Vertical Velocity W (kt)', 'b')]

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (var, label, letter) in zip(axs, panels):
        pass_data_use = pass_data   if var == 'REFLECTIVITY' else pass_data[1:]
        labels_use    = PASS_LABELS if var == 'REFLECTIVITY' else PASS_LABELS[1:]
        colors_use    = all_colors  if var == 'REFLECTIVITY' else all_colors[1:]

        for pd_, plabel, color in zip(pass_data_use, labels_use, colors_use):
            values, level = _grouped_values([pd_], var, all_codes, None, None)
            if var != 'REFLECTIVITY':
                values = values * knots
            _plot_cfad_contour(ax, values, level, label, color=color, name=plabel)

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


def complete_flightlevel_partitions_nice():
    """
    2x3 publication-style figure: W, mixing ratio, temperature,
    theta-e, and relative humidity distributions (legend in 6th cell) (weak/stratiform vs. convective),
    using curtain matching + upscaled (native 1 Hz) resolution. Saved
    directly to the main figures directory as flightlevel_dists_nice.png.
    """
    os.makedirs(path_figures, exist_ok=True)

    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')  # hours UTC

    class_time, class_codes = _curtain_classification()
    sample_time, codes_upscaled = _upscale_to_flight_resolution(
        class_time, class_codes, fl_time)
    ws_mask = np.isin(codes_upscaled, WEAK_STRAT_CODES)
    cv_mask = np.isin(codes_upscaled, CONVECTIVE_CODES)

    variables = [
        ('UWZ.d',     'Vertical Velocity W (m s$^{-1}$)'),
        ('MR.d',      'Mixing Ratio (g kg$^{-1}$)'),
        ('TA.d',      'Air Temperature (°C)'),
        ('THETAE.d',  'Equivalent Potential Temperature (K)'),
        ('RH',        'Relative Humidity (%)'),
    ]
    panel_letters = ['c', 'd', 'e', 'f', 'g']
    the_recalc, rh_recalc = flightlevel_rh_thetae(fl)   # recomputed from T, MR, p

    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (var, label), letter in zip(axs.flat, variables, panel_letters):
        data = {'THETAE.d': the_recalc, 'RH': rh_recalc}.get(var)
        if data is None:
            data = fl[var].values
        if var == 'HUM_REL.d':
            # sensor noise pushes some readings slightly above 100% —
            # physically impossible, so cap (round down) at 100%
            data = np.clip(data, None, 100.)

        finite = data[ws_mask | cv_mask]
        finite = finite[np.isfinite(finite)]
        bins = np.linspace(np.nanmin(finite), np.nanmax(finite), 40)

        _plot_distribution(ax, data[ws_mask], data[cv_mask], bins, label)
        ax.text(0.03, 0.96, f'({letter})', transform=ax.transAxes,
               ha='left', va='top', fontsize=13, zorder=20,
               bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                         linewidth=1.2, pad=4))
        if letter == 'g':
            ax.set_ylim(0, 1.)   # RH piles up near 100% in convection
        if letter in ('d', 'e', 'g'):
            ax.set_ylabel('')

    # 6th cell: legend only
    ax_leg = axs.flat[5]
    ax_leg.axis('off')
    handles, labels = axs.flat[0].get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc='center', fontsize=13, framealpha=1.)
    plt.tight_layout()
    plt.savefig(path_figures + 'figure11_fl_dists.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved flightlevel_dists_nice.png')


# ---------------------------------------------------------------------------
# Buffer zone vs. SAL flight-level analysis
# ---------------------------------------------------------------------------

# decimal-hour UTC windows (placeholders; adjust as needed)
BUFFER_WINDOWS = [(19.4, 19.52), (19.825, 19.9)]
SAL_WINDOWS    = [(19.52, 19.825)]
FL_PRES_VAR    = 'PS.c'   # corrected flight-level static pressure (mb); only in the full file
FL_FULL_FILE   = '20240715I1_A.nc'
BUFFER_COLOR   = 'tab:orange'
SAL_COLOR      = 'saddlebrown'


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


def flightlevel_buffer_sal_dists():
    """
    Same layout as complete_flightlevel_partitions_nice(), but comparing
    buffer-zone and SAL distributions (bold), with the convective
    distribution as a faint reference line. Saves flightlevel_dists_nice_sal.png.
    """
    os.makedirs(path_figures, exist_ok=True)

    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')

    class_time, class_codes = _curtain_classification()
    _, codes_upscaled = _upscale_to_flight_resolution(class_time, class_codes, fl_time)
    cv_mask  = np.isin(codes_upscaled, CONVECTIVE_CODES)
    buf_mask = _window_mask(fl_time, BUFFER_WINDOWS)
    sal_mask = _window_mask(fl_time, SAL_WINDOWS)

    variables = [
        ('UWZ.d',     'Vertical Velocity W (m s$^{-1}$)'),
        ('MR.d',      'Mixing Ratio (g kg$^{-1}$)'),
        ('TA.d',      'Air Temperature (°C)'),
        ('THETAE.d',  'Equivalent Potential Temperature (K)'),
        ('RH',        'Relative Humidity (%)'),
    ]
    panel_letters = ['c', 'd', 'e', 'f', 'g']
    the_recalc, rh_recalc = flightlevel_rh_thetae(fl)

    # 2 rows x 3 columns: 5 distributions + legend in the 6th cell
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (var, label), letter in zip(axs.flat, variables, panel_letters):
        data = {'THETAE.d': the_recalc, 'RH': rh_recalc}.get(var)   # recomputed theta_e / RH
        if data is None:
            data = fl[var].values
        finite = data[buf_mask | sal_mask | cv_mask]
        finite = finite[np.isfinite(finite)]
        bins = np.linspace(np.nanmin(finite), np.nanmax(finite), 40)
        centers = 0.5 * (bins[:-1] + bins[1:])

        for mask, color, name, lw, alpha in [
                (cv_mask,  'dimgray',    'Convective (reference)', 0.8, 1.),
                (buf_mask, BUFFER_COLOR, 'Buffer zone',           1.8, 1.),
                (sal_mask, SAL_COLOR,    'SAL',                   1.8, 1.)]:
            vals = data[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            hist, _ = np.histogram(vals, bins=bins)
            ax.plot(centers, hist / hist.sum(), color=color, lw=lw,
                    alpha=alpha, label=name)

        ax.set_xlabel(label, fontsize=13, labelpad=10)
        ax.set_ylabel('Probability', fontsize=13, labelpad=10)
        ax.tick_params(axis='both', labelsize=11)
        ax.text(0.03, 0.96, f'({letter})', transform=ax.transAxes,
                ha='left', va='top', fontsize=13, zorder=20,
                bbox=dict(facecolor='white', alpha=1., edgecolor='k',
                          linewidth=1.2, pad=4))
        if letter in ('d', 'e', 'g'):
            ax.set_ylabel('')

    # 6th cell: legend only
    ax_leg = axs.flat[5]
    ax_leg.axis('off')
    handles, labels = axs.flat[0].get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc='center', fontsize=13, framealpha=1.)
    plt.tight_layout()
    plt.savefig(path_figures + 'figure12_fl_dists.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved flightlevel_dists_nice_sal.png')


def flightlevel_buffer_sal_timeseries(t_start=19 + 20 / 60, t_end=20.0):
    """
    Flight-level time series in the style of convective_mixing(): (h) U/V/W,
    (i) WVMR + temperature, (j) theta-e + RH. Buffer-zone and SAL boundaries
    are marked with colored vertical lines. Only dropsondes 17, 18, 21 are shown.
    Saves flightlevel_timeseries_sal.png.
    """
    from dropsonde_profiles import _plot_fl_timeseries

    os.makedirs(path_figures, exist_ok=True)
    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')

    w, u, v = fl['UWZ.d'].values, fl['UWX.d'].values, fl['UWY.d'].values
    mr, ta  = fl['MR.d'].values,  fl['TA.d'].values
    the, rh = flightlevel_rh_thetae(fl)   # recomputed from T, MR, p
    sondes  = (17, 18, 21)

    label_fs = plt.rcParams['axes.labelsize']
    fig, (ax_g, ax_h, ax_i) = plt.subplots(3, 1, figsize=(10.5, 8))

    _plot_fl_timeseries(
        ax_g, fl_time, var_left=u,
        label_left='Wind components (m s$^{-1}$)', legend_label_left='U',
        color_left='steelblue',
        extra_vars=[(v, 'V', 'tomato'), (w, 'W', 'k')],
        t_start=t_start, t_end=t_end, panel_label='h', sonde_ids=sondes)
    ax_g.set_ylim([-6, 12])

    _plot_fl_timeseries(
        ax_h, fl_time, var_left=mr,
        label_left='WVMR (g kg$^{-1}$)', color_left='royalblue',
        var_right=ta, label_right='Temperature (°C)', color_right='firebrick',
        t_start=t_start, t_end=t_end, panel_label='i', add_leg=False,
        sonde_ids=sondes)

    _plot_fl_timeseries(
        ax_i, fl_time, var_left=the,
        label_left='$\\theta_e$ (K)', color_left='darkgreen',
        var_right=rh, label_right='Relative Humidity (%)', color_right='purple',
        t_start=t_start, t_end=t_end, panel_label='j', add_leg=False,
        sonde_ids=sondes)
    ax_i.set_ylim([320, 350])
    ax_i.set_xlabel('Time (Hours, UTC)', fontsize=label_fs)

    # boundaries of buffer-zone / SAL segments (SAL drawn last so it sits on top)
    for ax in (ax_g, ax_h, ax_i):
        for windows, color in [(BUFFER_WINDOWS, BUFFER_COLOR), (SAL_WINDOWS, SAL_COLOR)]:
            for t0, t1 in windows:
                for tb in (t0, t1):
                    ax.axvline(tb, color=color, lw=2, zorder=1)   # behind data lines

    plt.tight_layout()
    plt.savefig(path_figures + 'figure12_fl_ts.png', dpi=300,
                bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print('Saved flightlevel_timeseries_sal.png')


def flightlevel_dists_spatial():
    """
    Flight track (lon/lat) colored by segment: buffer zone, SAL, and other.
    Saves flightlevel_dists_spatial.png.
    """
    os.makedirs(path_figures, exist_ok=True)
    fl = xr.open_dataset(FLIGHT_LEVEL_PATH + FLIGHT_LEVEL_FILE, decode_times=False)
    fl_time = build_time_array(fl, axis_type='decimal')
    lon, lat = fl['LONref'].values, fl['LATref'].values

    buf_mask = _window_mask(fl_time, BUFFER_WINDOWS)
    sal_mask = _window_mask(fl_time, SAL_WINDOWS)
    in_range = (fl_time >= 19.2) & (fl_time <= 20.0)
    other    = in_range & ~buf_mask & ~sal_mask

    fig, ax = plt.subplots(figsize=(6, 6))
    for mask, color, name, lw in [(other,    'lightgray',  'Other (19.2-20 UTC)', 1.5),
                                  (buf_mask, BUFFER_COLOR, 'Buffer zone',         2.5),
                                  (sal_mask, SAL_COLOR,    'SAL',                 2.5)]:
        # NaN-out other segments so separate legs don't get joined by a line
        ax.plot(np.where(mask, lon, np.nan), np.where(mask, lat, np.nan),
                color=color, lw=lw, label=name)
    ax.set_xlabel('Longitude ($\\degree$)', fontsize=13)
    ax.set_ylabel('Latitude ($\\degree$)', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(framealpha=1.)
    plt.tight_layout()
    plt.savefig(path_figures + 'flightlevel_dists_spatial.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved flightlevel_dists_spatial.png')
