"""
Create north pole Lambert azimuthal equal area (NPLAEA) grids for
interpolating ERA-Interim lat-lon data.
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import xarray as xr

import grpy.geo as geo

# -----------------------------------------------------------------------------
savedir = '../../data/era_interim/grids/'

# Resolution for equal-area grid
grid_res = {'100km' : 100000, '200km' : 200000}

# Map projection parameters
map_opts = {'projection' : 'nplaea', 'ellps' : None, 'boundinglat' : 35,
            'lon_0' : -40, 'resolution' : None, 'round' : False}

# Domain parameters
# --- Approximate centre of Greenland
lon0, lat0 = -40, 73
# --- Domain width and height (in metres)
width, height = 8e6, 8e6

# File paths for saving output
savefiles = {'map' : savedir + 'map.csv'}
for res in grid_res:
    savefiles[res] = savedir + 'grid_' + res + '.nc'

# -----------------------------------------------------------------------------
# Create map for coordinate conversions
m = Basemap(**map_opts)

# Define domain
x0, y0 = np.round(np.array(m(lon0, lat0)) / 1e6, decimals=1) * 1e6
xmin, xmax = (x0 - width/2, x0 + width/2)
ymin, ymax = (y0 - height/2, y0 + height/2)

# Create equal-area polar grids
grids = {}
for nm, res in grid_res.items():
    grids[nm] = geo.equal_area_grid(xmin, xmax, ymin, ymax, m=m, res=res)

# Save map, domains and grid
print('Saving to' + savefiles['map'])
df = pd.DataFrame(map_opts, index=[0])
df.to_csv(savefiles['map'], index=False)
for nm in grids:
    print('Saving to ' + savefiles[nm])
    grids[nm].to_netcdf(savefiles[nm])
