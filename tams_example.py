#%%
import os
import warnings

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from xrframes import Frames

import tams

warnings.filterwarnings("ignore", category=cartopy.io.DownloadWarning)

xr.set_options(display_expand_data=False)

#%%
# Download example data if not already present
# tams.data.download_examples()

#%%

# Load data
tb = tams.load_example_tb().isel(time=slice(4))

tb.isel(time=0).plot(x="lon", y="lat", size=5, aspect=2.5)

#%%
# Identify cloud elements (CEs)
times = tb.time
contour_sets, contour_sets_219 = tams.identify(tb)

#%%
# Simple plot to test 219 matching
m, n = 0, 1  # time, contour #
fig, ax = plt.subplots()
c = contour_sets[m].iloc[[n]]
c.plot(ax=ax)
c.cs219.plot(color="red", ax=ax, alpha=0.4)

# Track CE groups between times
cs = tams.track(contour_sets, times, u_projection=-5).reset_index(drop=True)

tams.plot_tracked(cs)

#%%
size = 2.5
cx = cy = 3
vmin, vmax = 190, 300

if cx > 1 or cy > 1:
    tb_ = tb.coarsen(x=cx, y=cy, boundary="trim").mean()
else:
    tb_ = tb

x0, x1 = tb_.lon.min().item(), tb_.lon.max().item()
y0, y1 = tb_.lat.min().item(), tb_.lat.max().item()
extent = [-40, 50, 0, 20]

aspect = (x1 - x0) / (y1 - y0)
proj = ccrs.Mercator()
tran = ccrs.PlateCarree()

def plot(tb_i):
    fig = plt.figure(figsize=(size * aspect, size + 1))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(1, 1), height_ratios=(aspect * 2 + 1, 1),
        left=0.1, right=0.9, bottom=0.1, top=0.9,
        wspace=0.05, hspace=0.18,
    )

    ax = fig.add_subplot(gs[0, :], projection=proj)
    ax.set_extent(extent, crs=tran)
    ax.gridlines(draw_labels=True)
    ax.coastlines(color="orange", alpha=0.5)

    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    t = pd.Timestamp(tb_i.time.item())

    # Background -- CTT
    tb_i.plot(
        x="lon", y="lat",
        cmap="gray_r", ax=ax, cbar_ax=ax2,
        transform=tran,
        cbar_kwargs=dict(orientation="horizontal"),
        vmin=vmin, vmax=vmax, extend="both",
    )

    # CEs with colored precip (currently Tb)
    shapes = cs.query("time == @t")[["geometry"]]
    regions = regionmask.from_geopandas(shapes, overlap=False)
    mask = regions.mask(tb_i)
    masked = tb_i.where(mask >= 0)
    masked.plot.pcolormesh(
        x="lon", y="lat",
        ax=ax, cbar_ax=ax3, transform=tran, alpha=0.6,
        cbar_kwargs=dict(orientation="horizontal"),
        vmin=vmin, vmax=vmax, extend="both",
    )

    # Tracks up to this time
    for _, g in cs.groupby("mcs_id"):
        g_ = g[g.time <= t].dissolve("itime")
        c = g_.to_crs("EPSG:32663").centroid.to_crs("EPSG:4326")
        ax.plot(c.x, c.y, ".-", c="r", lw=2, alpha=0.4, transform=tran)
        c_t = c[g_.time == t]
        if not c_t.empty:
            ax.plot(c_t.x, c_t.y, ".", c="r", ms=8, transform=tran)

    ax.set_title("")
    ax.set_title(f"{t:%Y-%m-%d %HZ}", loc="left", size=11)

frames = Frames(tb_, plot, dim="time")
frames.write(dpi=120)
frames.to_gif("./tb.gif", fps=1, magick="READTHEDOCS" not in os.environ)

frames.display()


#%%
# Classify
cs = tams.classify(cs)
cs.head()




# %%
import matplotlib.pyplot as plt
import xarray as xr

import tams

xr.set_options(display_expand_data=False)

#%%
tb = tams.load_example_tb().isel(time=0)

tb.plot(x="lon", y="lat", size=2.3, aspect=6, cmap="gist_gray_r")

ax = plt.gca()
ax.set(xlim=(-40, 50), ylim=(0, 20))
ax.set_aspect("equal", "box")

#%%
fig, ax = plt.subplots(figsize=(10, 3))

for i, (thresh, color, ls) in enumerate(
    [
        (250, "firebrick", "--"),
        (235, "rebeccapurple", "-"),  # default
        (225, "mediumblue", ":"),
    ]
):
    ce = tams.identify(tb, ctt_threshold=thresh)[0][0]
    ce.plot(ax=ax, ec=color, fc="none", ls=ls)
    ax.text(0.005, 0.98 - (2 - i) * 0.1, len(ce), color=color, size=12, ha="left", va="top", transform=ax.transAxes)

ax.set_title("$n$ CEs", loc="left", size=10)
ax.set(xlabel="Longitude", ylabel="Latitude");

# %%
%%time

cases = [10, 100, 200, 500, 1000, 2000, 4000, 10_000]

fig, axs = plt.subplots(len(cases), 1, sharex=True, sharey=True, figsize=(5, 8), constrained_layout=True)

for ax, thresh in zip(axs.flat, cases):
    ce = tams.identify(tb, size_threshold=thresh)[0][0]
    ce.plot(ax=ax, ec="0.2", fc="none")
    ax.text(0.005, 0.97, f"{len(ce)}", size=10, ha="left", va="top", transform=ax.transAxes)
    ax.text(0.005, 0.03, f"≥{thresh}km²", size=8, ha="left", va="bottom", transform=ax.transAxes)

axs[0].set_title("$n$ CEs", loc="left", size=10)
fig.supxlabel("Longitude")
fig.supylabel("Latitude");

#%%
(
    tams.identify(tb, size_filter=False)[0][0]
    .plot(ec="0.2", fc="none")
    .set(xlabel="Longitude", ylabel="Latitude")
);
# %%
