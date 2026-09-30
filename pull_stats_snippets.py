# goal: use this notebook to pull specific stats on dropsonde etc
# features for paper / table details.

#%%
# import...
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import synoptic_overview
from dropsonde_profiles import path_project

#%%
# differences between drop 4 and 6 temps, analysis here

path_drops   = str(path_project) + "/data/dropsondes/"
drop_files   = synoptic_overview.pull_files_helper(path_drops)
d4 = xr.open_dataset(path_drops + drop_files[4])
d6 = xr.open_dataset(path_drops + drop_files[6])

# drop NaNs and sort by pressure (descending = surface first)
def _profile(d):
    p, z, T = d.pres.values, d.gpsalt.values, d.tdry.values
    ok = ~np.isnan(p) & ~np.isnan(T)
    p, z, T = p[ok], z[ok], T[ok]
    o = np.argsort(p)[::-1]
    return p[o], z[o], T[o]

p4, z4, T4 = _profile(d4)
p6, z6, T6 = _profile(d6)

# interpolate drop 4 onto drop 6 pressure levels (np.interp needs ascending x)
T4_on_6 = np.interp(p6, p4[::-1], T4[::-1], left=np.nan, right=np.nan)
dT = T6 - T4_on_6   # drop 6 minus drop 4

print(f'Drop 4: {drop_files[4]}\nDrop 6: {drop_files[6]}')
for pp, t6, t4, dd in list(zip(p6, T6, T4_on_6, dT))[::50]:
    print(f'p={pp:7.1f} hPa  T6={t6:6.2f}  T4={t4:6.2f}  dT={dd:+6.2f}')
print(f'mean dT = {np.nanmean(dT):+.2f} C, min = {np.nanmin(dT):+.2f}, max = {np.nanmax(dT):+.2f}')

fig, ax = plt.subplots(figsize=(4, 6))
ax.plot(dT, z6, 'k', lw=2)
ax.axvline(0, color='gray', ls='--')
# ax.set_ylim(1020, 400)
ax.set_xlim([-1.75, 1.75])
ax.set_xlabel('T$_6$ - T$_4$ (°C)')
ax.set_ylabel('Pressure (hPa)')
ax.axhline(y=250)
plt.show()


# %%
